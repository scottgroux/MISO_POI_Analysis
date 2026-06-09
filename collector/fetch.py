"""
fetch.py
--------
Polls the MISO public real-time 5-minute ExPost LMP API and appends
new intervals to a local Parquet store, partitioned by date.

Run manually:       python collector/fetch.py
Run on a schedule:  use cron, APScheduler, or Render Cron Job
"""

import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; env vars can be set by other means

# ── Configuration ────────────────────────────────────────────────────────────

# MISO rolling market day endpoint (new API, Dec 2025+)
# Returns all approved 5-min intervals for the current market day.
ROLLING_URL = (
    "https://public-api.misoenergy.org/api/MarketPricing"
    "/GetRealTimeFiveMinExPost/Rolling"
)

# Where partitioned Parquet files are written.
# Structure: data/lmp/date=YYYY-MM-DD/part.parquet
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "lmp"

# Column mapping — API returns integer-indexed columns (0..4).
# Positional order confirmed 2026-06-02: timestamp, node, lmp, congestion, loss.
RENAME_MAP = {
    0: "interval_est",
    1: "node",
    2: "lmp",
    3: "congestion",
    4: "loss",
}

# Only store nodes belonging to Indiana utilities/zones.
# Filters from ~2,600 MISO-wide nodes down to ~108 Indiana nodes.
INDIANA_PREFIXES = {"INDIANA", "INDN", "IPL", "NIPS", "SIGE", "PSI_GEN"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Core fetch logic ─────────────────────────────────────────────────────────

def fetch_rolling() -> pd.DataFrame:
    """Download the current rolling market day from MISO and return a clean DataFrame."""
    log.info("Fetching rolling LMP data from MISO …")
    try:
        resp = requests.get(ROLLING_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("Request failed: %s", e)
        raise

    payload = resp.json()

    # MISO wraps the data in different keys depending on the endpoint version.
    # Try common wrapper keys; fall back to treating the root as a list.
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("data", "Data", "items", "Items", "result", "Result"):
            if key in payload:
                records = payload[key]
                break
        else:
            # If we can't find a list, surface the raw response for inspection.
            log.warning("Unexpected response shape. Keys: %s", list(payload.keys()))
            records = []
    else:
        records = []

    if not records:
        log.warning("No records returned. The API response may have changed.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    log.info("Raw response: %d rows, columns: %s", len(df), list(df.columns))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, parse timestamps, cast numerics."""
    if df.empty:
        return df

    # Rename only the columns that exist in this response
    rename = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Parse interval timestamp → UTC-aware datetime.
    # MISO's timestamp is a fixed UTC-5 offset year-round (no DST observed —
    # verified empirically: no gap on spring-forward days, no duplicated hour
    # on fall-back days). A plain +5h shift is correct and never ambiguous,
    # unlike tz_localize("America/New_York") which misconverts ~half the year.
    if "interval_est" in df.columns:
        df["interval_est"] = pd.to_datetime(df["interval_est"])
        df["interval_utc"] = (df["interval_est"] + pd.Timedelta(hours=5)).dt.tz_localize("UTC")

        df["date"] = df["interval_utc"].dt.date.astype(str)

    # Cast price columns to float
    for col in ["lmp", "congestion", "loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only the columns we've mapped (ignore anything extra)
    keep = [c for c in ["node", "interval_est", "interval_utc", "date",
                         "lmp", "congestion", "loss"]
            if c in df.columns]
    return df[keep]


# ── Storage ──────────────────────────────────────────────────────────────────

def load_existing(date_str: str) -> pd.DataFrame:
    """Load the Parquet file for a given date.
    Checks local cache first; falls back to R2 download if absent (needed on
    ephemeral runtimes like Render where there is no persistent local cache)."""
    path = DATA_DIR / f"date={date_str}" / "part.parquet"
    if path.exists():
        return pd.read_parquet(path)
    try:
        from r2 import download, r2_enabled
        if r2_enabled():
            path.parent.mkdir(parents=True, exist_ok=True)
            if download(date_str, path):
                return pd.read_parquet(path)
    except Exception as e:
        log.warning("R2 prefetch failed for %s: %s", date_str, e)
    return pd.DataFrame()


def save(df: pd.DataFrame, date_str: str) -> None:
    """Append-write (deduplicated) to the Parquet partition for this date."""
    path = DATA_DIR / f"date={date_str}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing(date_str)
    combined = pd.concat([existing, df], ignore_index=True)

    # Deduplicate on (node, interval_utc) — the natural primary key
    dedup_cols = [c for c in ["node", "interval_utc"] if c in combined.columns]
    if dedup_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        log.info("Dedup: %d → %d rows for %s", before, len(combined), date_str)

    combined.to_parquet(path, index=False, compression="snappy")
    log.info("Saved %d rows to %s", len(combined), path)

    # Write-through to R2 (no-op if env vars not set)
    try:
        from r2 import upload, r2_enabled
        if r2_enabled():
            upload(path, date_str)
    except Exception as e:
        log.warning("R2 upload skipped: %s", e)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(inspect: bool = False) -> None:
    """Main collection cycle: fetch → clean → save."""
    raw = fetch_rolling()

    if inspect or raw.empty:
        print("\n── Raw API response (first 3 rows) ──")
        print(raw.head(3).to_string())
        print("\n── Columns ──")
        print(list(raw.columns))
        if inspect:
            return

    df = clean(raw)
    if df.empty:
        log.warning("Nothing to save after cleaning.")
        return

    # Filter to Indiana nodes only
    df = df[df["node"].str.split(".").str[0].isin(INDIANA_PREFIXES)]
    if df.empty:
        log.warning("No Indiana nodes found in this fetch.")
        return
    log.info("Indiana filter: %d rows retained", len(df))

    # Data can span two calendar dates around midnight — save each separately
    for date_str, group in df.groupby("date"):
        save(group.drop(columns=["date"]), date_str)

    log.info("Collection cycle complete. Total new rows processed: %d", len(df))


if __name__ == "__main__":
    inspect_mode = "--inspect" in sys.argv
    run(inspect=inspect_mode)
