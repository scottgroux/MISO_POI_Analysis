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

Scheduled to run nightly ~2am via Render cron job.
"""

import logging
import sys
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
ROLLING_WINDOW = 7 * 288  # 7 days × 288 intervals = 2016 intervals


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

    # Rolling mean over a sorted time series
    df["rolling_7d"] = (
        df["lmp"]
        .rolling(window=ROLLING_WINDOW, min_periods=1)
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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    if not r2_enabled():
        log.error("R2 env vars not set — aborting.")
        sys.exit(1)

    all_dates    = available_dates()
    recent_dates = all_dates[-RECENT_DAYS:]

    log.info("Store: %d total dates, using last %d for recent summaries",
             len(all_dates), len(recent_dates))

    # ── Load all data for seasonal computation ────────────────────────────────
    log.info("Loading ALL partitions for seasonal summaries…")
    all_df = load_all(all_dates)
    if all_df.empty:
        log.error("No data loaded — aborting.")
        sys.exit(1)

    # Use float32 to cut memory roughly in half
    all_df["lmp"] = all_df["lmp"].astype("float32")
    nodes = sorted(all_df["node"].unique().tolist())
    log.info("Nodes found: %d", len(nodes))

    # ── Load recent data for rolling/monthly/actuals ──────────────────────────
    log.info("Loading last %d partitions for recent summaries…", len(recent_dates))
    recent_df = load_all(recent_dates)
    if "lmp" in recent_df.columns:
        recent_df["lmp"] = recent_df["lmp"].astype("float32")

    # ── Per-node summaries ────────────────────────────────────────────────────
    for i, node in enumerate(nodes, 1):
        log.info("[%d/%d] Summarizing %s", i, len(nodes), node)
        node_all    = all_df[all_df["node"] == node]
        node_recent = recent_df[recent_df["node"] == node] if not recent_df.empty else pd.DataFrame()

        prefix = f"summaries/{node}"

        upload_json(compute_seasonal(node_all),        f"{prefix}/seasonal.json")
        if not node_recent.empty:
            upload_json(compute_rolling_7d(node_recent),   f"{prefix}/rolling_7d.json")
            upload_json(compute_monthly_avg(node_recent),  f"{prefix}/monthly_avg.json")
            upload_json(compute_recent_actuals(node_recent), f"{prefix}/recent_actuals.json")

    # ── Global meta ───────────────────────────────────────────────────────────
    meta = {
        "earliest":    all_dates[0],
        "latest":      all_dates[-1],
        "total_dates": len(all_dates),
        "nodes":       nodes,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    upload_json(meta, "summaries/meta.json")
    log.info("Done. Uploaded summaries for %d nodes.", len(nodes))


if __name__ == "__main__":
    main()
