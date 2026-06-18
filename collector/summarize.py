"""
summarize.py
------------
Reads all Parquet partitions from the store (via local cache → R2 fallback),
computes pre-aggregated JSON summaries, and uploads them to R2 under
summaries/<node>/<type>.json for direct browser consumption.

Summary types (per node, all 109 Indiana nodes):
  seasonal.json          — avg, P10, P90 LMP by (month × 5-min slot) across all years
  rolling_7d.json        — last 30 days of raw + 7-day rolling average per interval
  monthly_by_period.json — avg, P25, P75 LMP by 5-min slot for every (year, month)
    that has data, so the site can let users pick a specific month and year
  recent_actuals.json    — last 7 days of raw 5-min LMP + congestion

Global:
  summaries/meta.json — node list, date range, last_updated timestamp
  summaries/geo_seasonal.json — avg LMP by (month × 5-min slot) for every node,
    used by the Geo-View map

Scheduled to run nightly ~2am via Render cron job.
"""

import gc
import logging
import sys
import warnings
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

RECENT_DAYS           = 35   # days of data shown for rolling/monthly/actuals
ROLLING_LOOKBACK_DAYS = 7     # extra trailing days loaded just to seed the
                              # 7-day rolling average so it has full context
                              # at the left edge of the displayed window
                              # instead of tapering toward the raw value


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_one(date_str: str) -> pd.DataFrame:
    df = load_date(date_str)
    if df.empty:
        return df
    # Keep only the columns we need
    keep = [c for c in ["node", "interval_utc", "lmp", "congestion"] if c in df.columns]
    return df[keep]


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


def compute_rolling_7d(df: pd.DataFrame, display_days: int = RECENT_DAYS) -> dict:
    """Per-interval 7-day rolling average for the last `display_days` days.

    `df` is expected to include ROLLING_LOOKBACK_DAYS of data *before* the
    display window. Without that lookback, the rolling average for the
    earliest displayed days would only have 1-6 days of history to average
    over instead of a true trailing 7 days — visible as the line tapering
    toward the raw value at the left edge of the chart.
    """
    df = df.copy()
    df["interval_utc"] = pd.to_datetime(df["interval_utc"], utc=True)
    df = df.sort_values("interval_utc")

    # Time-based rolling window: "trailing 7 days" by timestamp, not row count.
    # This stays correct even when there are gaps in the 5-minute series.
    df["rolling_7d"] = (
        df.rolling("7D", on="interval_utc", min_periods=1)["lmp"]
        .mean()
    )

    # Drop the lookback buffer now that it's done its job of seeding the
    # rolling window — only the display window actually gets plotted.
    cutoff = df["interval_utc"].max() - pd.Timedelta(days=display_days)
    df = df[df["interval_utc"] > cutoff]

    return {
        "timestamps": df["interval_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "lmp":        _round_list(df["lmp"]),
        "rolling_7d": _round_list(df["rolling_7d"]),
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


# ── Streaming seasonal aggregation ───────────────────────────────────────────

def compute_seasonal_streaming(all_dates: list[str], valid_nodes: set[str]) -> dict[str, dict]:
    """Compute seasonal avg/P10/P90 per node by streaming one partition at a
    time, uploading each node's seasonal.json as soon as it's computed.

    For each (node, month), accumulates one 288-slot row per day (a small
    float32 array) rather than every raw reading individually — this keeps
    memory roughly proportional to (nodes x months x days), not total rows,
    which matters once the store spans years of 5-minute data.

    `valid_nodes` restricts output to the current node set (discovered from
    the latest partition in main()) — older partitions can contain retired
    or since-renamed node names that would otherwise leave orphaned summary
    files in R2 with no corresponding entry in meta.json or node_coords.json.

    Returns the much smaller avg-only geo_nodes dict (used for the combined
    Geo-View payload) rather than the full per-node results — the caller
    never needs the full per-month avg/P10/P90 data again once it's uploaded.
    """
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
            if node_val not in valid_nodes:
                continue
            row = np.full(288, np.nan, dtype="float32")
            row[grp["slot"].to_numpy()] = grp["lmp"].to_numpy()
            acc[node_val][month].append(row)

        if i % 100 == 0:
            log.info("  Streamed %d / %d partitions", i, len(all_dates))

    log.info("Seasonal accumulation done. Computing stats…")

    # Convert to per-node seasonal dicts and upload immediately, node by
    # node, rather than building a full results dict first — that dict
    # would otherwise sit in memory alongside the (still-shrinking) acc
    # for the whole conversion pass. Also build the much smaller geo_nodes
    # (avg only) dict to return for the combined Geo-View payload.
    geo_nodes: dict[str, dict] = {}
    node_keys = list(acc.keys())
    for node in node_keys:
        months_data = acc.pop(node)
        month_out = {}
        for month_num in range(1, 13):
            rows = months_data.pop(month_num, None)
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
        upload_json({"slots": SLOT_LABELS, "months": month_out}, f"summaries/{node}/seasonal.json")
        geo_nodes[node] = {m: month_out[m]["avg"] for m in MONTH_ABBR}

    return geo_nodes


# ── Streaming monthly-by-period aggregation ──────────────────────────────────

def compute_monthly_by_period_streaming(all_dates: list[str], valid_nodes: set[str]) -> None:
    """Compute avg/P25/P75 per node for every distinct calendar (year, month)
    that has data, streaming one partition at a time, and upload each node's
    result to R2 as soon as it's computed (see note below on why).

    `valid_nodes` restricts output to the current node set — see
    compute_seasonal_streaming's docstring for why that filter matters.

    Same memory rationale as compute_seasonal_streaming above for the
    accumulation phase, just bucketed by "YYYY-MM" instead of
    month-across-all-years. Powers the Monthly Profile chart's year/month
    picker. Must run as its own pass, after the seasonal accumulator has
    been freed, rather than alongside it — holding both full-history
    accumulators in memory at once would roughly double peak RSS."""
    from collections import defaultdict

    # acc[node][period] = [row, ...] where period is "YYYY-MM" and each row
    # is a 288-slot float32 array
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    log.info("Streaming %d partitions for monthly-by-period aggregation…", len(all_dates))
    for i, date_str in enumerate(all_dates, 1):
        df = load_date(date_str)
        if df.empty:
            continue

        period = date_str[:7]  # "YYYY-MM"
        df = df.copy()
        df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce").astype("float32")
        df = df.dropna(subset=["lmp"])

        dt = pd.to_datetime(df["interval_utc"], utc=True)
        df["slot"] = ((dt.dt.hour * 60 + dt.dt.minute) // 5).astype("int16")

        for node_val, grp in df.groupby("node"):
            if node_val not in valid_nodes:
                continue
            row = np.full(288, np.nan, dtype="float32")
            row[grp["slot"].to_numpy()] = grp["lmp"].to_numpy()
            acc[node_val][period].append(row)

        if i % 100 == 0:
            log.info("  Streamed %d / %d partitions", i, len(all_dates))

    log.info("Monthly-by-period accumulation done. Computing stats…")

    # Upload each node's payload immediately rather than building one big
    # results dict first — this one has ~3.5x more (node, period) entries
    # than the seasonal equivalent (every year-month vs. just 12 months),
    # so holding it all in memory at once is the difference between this
    # job fitting in Render's 512MB budget and OOMing.
    node_keys = list(acc.keys())
    for node in node_keys:
        periods_data = acc.pop(node)
        by_period = {}
        for period in sorted(periods_data.keys()):
            rows = periods_data.pop(period)
            mat = np.vstack(rows)  # shape: (n_days_in_period, 288)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg = np.nanmean(mat, axis=0)
                p25 = np.nanpercentile(mat, 25, axis=0)
                p75 = np.nanpercentile(mat, 75, axis=0)
            by_period[period] = {
                "avg": _round_list(avg),
                "p25": _round_list(p25),
                "p75": _round_list(p75),
            }
        upload_json({
            "slots": SLOT_LABELS,
            "periods": sorted(by_period.keys()),
            "by_period": by_period,
        }, f"summaries/{node}/monthly_by_period.json")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    if not r2_enabled():
        log.error("R2 env vars not set — aborting.")
        sys.exit(1)

    all_dates    = available_dates()
    recent_dates = all_dates[-(RECENT_DAYS + ROLLING_LOOKBACK_DAYS):]

    log.info("Store: %d total dates, using last %d (%d display + %d rolling lookback)",
             len(all_dates), len(recent_dates), RECENT_DAYS, ROLLING_LOOKBACK_DAYS)

    # ── Discover nodes from a sample ─────────────────────────────────────────
    sample = load_date(all_dates[-1])
    nodes = sorted(sample["node"].unique().tolist()) if not sample.empty else []
    log.info("Nodes: %d", len(nodes))

    # ── Seasonal: stream all partitions, accumulate per (node,month,slot),
    # and upload each node's seasonal.json as soon as it's computed (inside
    # the function) rather than collecting a full results dict here first.
    # Returns just the small avg-only geo_nodes dict for the combined
    # Geo-View payload. ───────────────────────────────────────────────────
    geo_nodes = compute_seasonal_streaming(all_dates, set(nodes))
    upload_json({"slots": SLOT_LABELS, "nodes": geo_nodes}, "summaries/geo_seasonal.json")
    del geo_nodes
    gc.collect()

    # meta.json goes up now too, before the lighter "recent" phase, so the
    # site reflects a fresh last_updated even if the recent phase fails.
    meta = {
        "earliest":    all_dates[0],
        "latest":      all_dates[-1],
        "total_dates": len(all_dates),
        "nodes":       nodes,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    upload_json(meta, "summaries/meta.json")
    log.info("Seasonal/geo/meta uploaded for %d nodes.", len(nodes))

    # ── Monthly-by-period: a second full streaming pass, run only after the
    # seasonal accumulator above is freed so the two never overlap in memory.
    # Partitions are already cached on local disk from the seasonal pass, so
    # this doesn't re-hit R2 — just re-reads local Parquet files. Uploads
    # per node internally (same reasoning as compute_seasonal_streaming) —
    # this one has ~3.5x more (node, period) entries than seasonal's, so
    # collecting a full results dict here first would be the difference
    # between fitting in Render's 512MB budget and OOMing.
    compute_monthly_by_period_streaming(all_dates, set(nodes))
    log.info("Monthly-by-period uploaded for %d nodes.", len(nodes))
    gc.collect()

    # ── Recent: stream one partition at a time into per-node buffers ─────────
    # load_all() used to download all 35 days in parallel and pd.concat them
    # into one big DataFrame — that holds the full 35-day x all-nodes dataset
    # twice (the per-date frames list, then the concat result) at once, which
    # is what pushed this job over Render's 512MB limit once the store grew to
    # 1000+ partitions. Streaming bounds memory to ~(nodes x recent_days)
    # small per-node slices instead, and these partitions are already cached
    # on local disk from the seasonal phase above, so no extra R2 downloads.
    try:
        log.info("Streaming %d recent partitions for per-node summaries…", len(recent_dates))
        node_set = set(nodes)
        node_frames: dict[str, list] = {n: [] for n in nodes}

        for j, date_str in enumerate(recent_dates, 1):
            df = _load_one(date_str)
            if df.empty:
                continue
            if "lmp" in df.columns:
                df = df.copy()
                df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce").astype("float32")
            keep = [c for c in ["interval_utc", "lmp", "congestion"] if c in df.columns]
            for node_val, grp in df.groupby("node"):
                if node_val in node_set:
                    node_frames[node_val].append(grp[keep].reset_index(drop=True))
            if j % 10 == 0:
                log.info("  Streamed %d / %d recent partitions", j, len(recent_dates))

        for i, node in enumerate(nodes, 1):
            frames = node_frames.pop(node, [])  # pop to free as we go
            if frames:
                node_recent = pd.concat(frames, ignore_index=True)
                prefix = f"summaries/{node}"
                upload_json(compute_rolling_7d(node_recent),      f"{prefix}/rolling_7d.json")
                upload_json(compute_recent_actuals(node_recent),  f"{prefix}/recent_actuals.json")
            if i % 25 == 0:
                log.info("  Recent summaries uploaded for %d / %d nodes", i, len(nodes))

        log.info("Done. Recent summaries uploaded for %d nodes.", len(nodes))
    except Exception:
        log.exception("Recent-data summary phase failed; seasonal/geo/meta were already updated.")


if __name__ == "__main__":
    main()
