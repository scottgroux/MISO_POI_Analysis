"""
summarize.py
------------
Reads all Parquet partitions from the store (via local cache → R2 fallback),
computes pre-aggregated JSON summaries, and uploads them to R2 under
summaries/<node>/<type>.json for direct browser consumption.

Summary types (per node, all 109 Indiana nodes):
  seasonal.json     — avg, P10, P90 LMP by (month × 5-min slot) across all years
  rolling_7d.json   — last 30 days of raw + 7-day rolling average per interval
  monthly_avg.json  — avg, P25, P75 LMP by 5-min slot for the current month
  recent_actuals.json — last 7 days of raw 5-min LMP + congestion

Global:
  summaries/meta.json — node list, date range, last_updated timestamp
  summaries/geo_seasonal.json — avg LMP by (month × 5-min slot) for every node,
    used by the Geo-View map

Scheduled to run nightly ~2am via Render cron job.
"""

import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import available_dates, load_date
from r2 import upload_json, r2_enabled

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# 288 5-minute slot labels for a 24-hour day
SLOT_LABELS = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
MONTH_ABBR  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

RECENT_DAYS   = 35   # days to load for rolling/monthly/actuals


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_one(date_str: str) -> pd.DataFrame:
    df = load_date(date_str)
    if df.empty:
        return df
    # Keep only the columns we need
    keep = [c for c in ["node", "interval_utc", "lmp", "congestion"] if c in df.columns]
    return df[keep]


def load_all(dates: list[str], workers: int = 10) -> pd.DataFrame:
    """Download and concat all partitions in parallel."""
    log.info("Loading %d partitions (up to %d parallel)…", len(dates), workers)
    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_one, d): d for d in dates}
        for i, future in enumerate(as_completed(futures), 1):
            df = future.result()
            if not df.empty:
                frames.append(df)
            if i % 100 == 0:
                log.info("  Loaded %d / %d partitions", i, len(dates))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    log.info("  Total rows loaded: %d", len(combined))
    return combined


# ── Slot helpers ──────────────────────────────────────────────────────────────

def add_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer slot index (0–287) from interval_utc."""
    if "interval_utc" not in df.columns:
        return df
    dt = pd.to_datetime(df["interval_utc"], utc=True)
    df = df.copy()
    df["slot"] = (dt.dt.hour * 60 + dt.dt.minute) // 5
    return df


# ── Summary computers ────────────────────────────────────────────────────────

def compute_seasonal(df: pd.DataFrame) -> dict:
    """For each (month, slot): mean, P10, P90 LMP across all available years."""
    df = add_slot(df)
    df["month"] = pd.to_datetime(df["interval_utc"], utc=True).dt.month

    out = {}
    for month_num in range(1, 13):
        m_df = df[df["month"] == month_num]
        if m_df.empty:
            agg = {"avg": [None]*288, "p10": [None]*288, "p90": [None]*288}
        else:
            grp = m_df.groupby("slot")["lmp"]
            avg = grp.mean().reindex(range(288))
            p10 = grp.quantile(0.10).reindex(range(288))
            p90 = grp.quantile(0.90).reindex(range(288))
            agg = {
                "avg": _round_list(avg),
                "p10": _round_list(p10),
                "p90": _round_list(p90),
            }
        out[MONTH_ABBR[month_num - 1]] = agg

    return {"slots": SLOT_LABELS, "months": out}


def compute_rolling_7d(df: pd.DataFrame) -> dict:
    """Per-interval 7-day rolling average for the last 30 days of data."""
    df = df.copy()
    df["interval_utc"] = pd.to_datetime(df["interval_utc"], utc=True)
    df = df.sort_values("interval_utc")

    # Time-based rolling window: "trailing 7 days" by timestamp, not row count.
    # This stays correct even when there are gaps in the 5-minute series.
    df["rolling_7d"] = (
        df.rolling("7D", on="interval_utc", min_periods=1)["lmp"]
        .mean()
    )

    return {
        "timestamps": df["interval_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "lmp":        _round_list(df["lmp"]),
        "rolling_7d": _round_list(df["rolling_7d"]),
    }


def compute_monthly_avg(df: pd.DataFrame) -> dict:
    """Average, P25, P75 LMP by 5-min slot for the current calendar month."""
    now = datetime.now(timezone.utc)
    month_str = now.strftime("%Y-%m")

    df = df.copy()
    df["interval_utc"] = pd.to_datetime(df["interval_utc"], utc=True)
    month_df = df[df["interval_utc"].dt.strftime("%Y-%m") == month_str]
    month_df = add_slot(month_df)

    grp = month_df.groupby("slot")["lmp"]
    avg = grp.mean().reindex(range(288))
    p25 = grp.quantile(0.25).reindex(range(288))
    p75 = grp.quantile(0.75).reindex(range(288))

    return {
        "month":  month_str,
        "slots":  SLOT_LABELS,
        "avg":    _round_list(avg),
        "p25":    _round_list(p25),
        "p75":    _round_list(p75),
    }


def compute_recent_actuals(df: pd.DataFrame) -> dict:
    """Last 7 days of raw 5-min LMP and congestion."""
    df = df.copy()
    df["interval_utc"] = pd.to_datetime(df["interval_utc"], utc=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    recent = df[df["interval_utc"] >= cutoff].sort_values("interval_utc")

    result = {
        "timestamps": recent["interval_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "lmp":        _round_list(recent["lmp"]),
    }
    if "congestion" in recent.columns:
        result["congestion"] = _round_list(recent["congestion"])
    return result


def _round_list(series: pd.Series, decimals: int = 2) -> list:
    return [round(float(v), decimals) if pd.notna(v) else None for v in series]


def compute_geo_seasonal(seasonal_by_node: dict[str, dict]) -> dict:
    """Combine per-node seasonal averages into a single payload for the
    Geo-View map: avg LMP by (month x slot) for every node, keyed by node."""
    nodes = {
        node: {m: data["months"][m]["avg"] for m in MONTH_ABBR}
        for node, data in seasonal_by_node.items()
    }
    return {"slots": SLOT_LABELS, "nodes": nodes}


# ── Streaming seasonal aggregation ───────────────────────────────────────────

def compute_seasonal_streaming(all_dates: list[str]) -> dict[str, dict]:
    """Compute seasonal avg/P10/P90 per node by streaming one partition at a time.

    For each (node, month), accumulates one 288-slot row per day (a small
    float32 array) rather than every raw reading individually — this keeps
    memory roughly proportional to (nodes x months x days), not total rows,
    which matters once the store spans years of 5-minute data."""
    from collections import defaultdict

    # acc[node][month] = [row, ...] where each row is a 288-slot float32 array
    acc: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    log.info("Streaming %d partitions for seasonal aggregation…", len(all_dates))
    for i, date_str in enumerate(all_dates, 1):
        df = load_date(date_str)
        if df.empty:
            continue

        month = int(date_str[5:7])
        df = df.copy()
        df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce").astype("float32")
        df = df.dropna(subset=["lmp"])

        dt = pd.to_datetime(df["interval_utc"], utc=True)
        df["slot"] = ((dt.dt.hour * 60 + dt.dt.minute) // 5).astype("int16")

        for node_val, grp in df.groupby("node"):
            row = np.full(288, np.nan, dtype="float32")
            row[grp["slot"].to_numpy()] = grp["lmp"].to_numpy()
            acc[node_val][month].append(row)

        if i % 100 == 0:
            log.info("  Streamed %d / %d partitions", i, len(all_dates))

    log.info("Seasonal accumulation done. Computing stats…")

    # Convert to per-node seasonal dicts
    results: dict[str, dict] = {}
    for node, months_data in acc.items():
        month_out = {}
        for month_num in range(1, 13):
            rows = months_data.get(month_num)
            if not rows:
                month_out[MONTH_ABBR[month_num - 1]] = {
                    "avg": [None] * 288, "p10": [None] * 288, "p90": [None] * 288,
                }
                continue
            mat = np.vstack(rows)  # shape: (n_days_in_month, 288)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                # Some slots may have no data across every day in this month
                # (all-NaN column) — that's expected and yields null below.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg = np.nanmean(mat, axis=0)
                p10 = np.nanpercentile(mat, 10, axis=0)
                p90 = np.nanpercentile(mat, 90, axis=0)
            month_out[MONTH_ABBR[month_num - 1]] = {
                "avg": _round_list(avg),
                "p10": _round_list(p10),
                "p90": _round_list(p90),
            }
        results[node] = {"slots": SLOT_LABELS, "months": month_out}

    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    if not r2_enabled():
        log.error("R2 env vars not set — aborting.")
        sys.exit(1)

    all_dates    = available_dates()
    recent_dates = all_dates[-RECENT_DAYS:]

    log.info("Store: %d total dates, using last %d for recent summaries",
             len(all_dates), len(recent_dates))

    # ── Discover nodes from a sample ─────────────────────────────────────────
    sample = load_date(all_dates[-1])
    nodes = sorted(sample["node"].unique().tolist()) if not sample.empty else []
    log.info("Nodes: %d", len(nodes))

    # ── Seasonal: stream all partitions, accumulate per (node,month,slot) ────
    seasonal_by_node = compute_seasonal_streaming(all_dates)

    # ── Upload seasonal + geo + meta now, before the lighter "recent" phase.
    #    This is the slow, memory-heavy step — get its results (and a fresh
    #    last_updated) onto R2 immediately so the site reflects them even if
    #    the recent phase below fails. ────────────────────────────────────────
    for node in nodes:
        if node in seasonal_by_node:
            upload_json(seasonal_by_node[node], f"summaries/{node}/seasonal.json")

    upload_json(compute_geo_seasonal(seasonal_by_node), "summaries/geo_seasonal.json")

    meta = {
        "earliest":    all_dates[0],
        "latest":      all_dates[-1],
        "total_dates": len(all_dates),
        "nodes":       nodes,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    upload_json(meta, "summaries/meta.json")
    log.info("Seasonal/geo/meta uploaded for %d nodes.", len(nodes))

    # Free the large seasonal accumulation before loading recent data.
    del seasonal_by_node

    # ── Recent: load last 35 days (small, fits easily in memory) ─────────────
    try:
        log.info("Loading last %d partitions for recent summaries…", len(recent_dates))
        recent_df = load_all(recent_dates)
        if "lmp" in recent_df.columns:
            recent_df["lmp"] = recent_df["lmp"].astype("float32")

        for i, node in enumerate(nodes, 1):
            node_recent = recent_df[recent_df["node"] == node] if not recent_df.empty else pd.DataFrame()
            if not node_recent.empty:
                prefix = f"summaries/{node}"
                upload_json(compute_rolling_7d(node_recent),      f"{prefix}/rolling_7d.json")
                upload_json(compute_monthly_avg(node_recent),     f"{prefix}/monthly_avg.json")
                upload_json(compute_recent_actuals(node_recent),  f"{prefix}/recent_actuals.json")
            if i % 25 == 0:
                log.info("  Recent summaries uploaded for %d / %d nodes", i, len(nodes))

        log.info("Done. Recent summaries uploaded for %d nodes.", len(nodes))
    except Exception:
        log.exception("Recent-data summary phase failed; seasonal/geo/meta were already updated.")


if __name__ == "__main__":
    main()
