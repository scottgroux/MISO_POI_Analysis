"""
backfill.py
-----------
Downloads and processes MISO's weekly 5-minute LMP ZIP archives,
converting them into the same Parquet partition format used by fetch.py.

MISO publishes final (corrected) RT 5-min LMP as weekly ZIPs every Monday.
The file date is the Monday publish date (~7-8 days after the data week ends).
Available files are discovered via the MISO market reports search API.

URL pattern:
  https://docs.misoenergy.org/marketreports/YYYYMMDD_5MIN_LMP.zip

Usage:
  # Pull everything available (slow — years of data)
  python collector/backfill.py --all

  # Pull a specific date range
  python collector/backfill.py --start 2024-01-01 --end 2024-06-30

  # Pull just the last N weeks
  python collector/backfill.py --weeks 8

  # Dry run: print which ZIPs would be fetched without downloading
  python collector/backfill.py --weeks 4 --dry-run
"""

import argparse
import io
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://docs.misoenergy.org/marketreports/{date}_5MIN_LMP.zip"

# MISO market reports Elasticsearch API — used to discover actual file publish dates.
MISO_SEARCH_URL = (
    "https://www.misoenergy.org/api/find/Optics_Models_Find_MarketReportItem/_search"
)

EARLIEST_DATE    = date(2013, 1, 1)   # approx. first available file

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "lmp"

# Column mapping from MISO ZIP CSVs to our schema (named columns in ZIP CSVs)
RENAME_MAP = {
    "PNODENAME":   "node",
    "MKTHOUR_EST": "interval_est",
    "LMP":         "lmp",
    "CON_LMP":     "congestion",
    "LOSS_LMP":    "loss",
    "MLC_LMP":     "mlc",
}

SLEEP_BETWEEN_REQUESTS = 1.5   # seconds — be polite to MISO's servers

# Only store nodes belonging to Indiana utilities/zones.
# Filters from ~2,600 MISO-wide nodes down to ~108 Indiana nodes.
INDIANA_PREFIXES = {"INDIANA", "INDN", "IPL", "NIPS", "SIGE", "PSI_GEN"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Download & parse ──────────────────────────────────────────────────────────

CHUNK_SIZE = 50_000   # rows per chunk — keeps peak memory low on 5M-row ZIPs


def _fetch_zip_url(url: str) -> pd.DataFrame | None:
    """Download one weekly ZIP and return only the Indiana rows as a DataFrame.
    Streams the CSV in chunks so we never hold all 5M rows in memory.
    Returns None if the file doesn't exist (404) or on network error."""
    log.info("Fetching %s", url)

    try:
        resp = requests.get(url, timeout=120)
    except requests.RequestException as e:
        log.warning("Network error for %s: %s", url, e)
        return None

    if resp.status_code == 404:
        log.info("Not found (404) — skipping %s", url)
        return None

    if not resp.ok:
        log.warning("HTTP %d for %s", resp.status_code, url)
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.upper().endswith(".CSV")]
            if not csv_names:
                log.warning("No CSV found inside %s", url)
                return None

            kept = []
            total_rows = 0
            with zf.open(csv_names[0]) as f:
                # Read header row manually so we can use chunksize with c engine
                # MISO CSVs: 4 header rows (rows 0-3 are metadata), row 4 is column names
                for _ in range(4):
                    f.readline()
                reader = pd.read_csv(f, chunksize=CHUNK_SIZE, on_bad_lines="skip")
                for chunk in reader:
                    total_rows += len(chunk)
                    # Drop the footer row MISO appends ("*END*" in first column)
                    chunk = chunk[~chunk.iloc[:, 0].astype(str).str.startswith("*")]
                    if "PNODENAME" in chunk.columns:
                        chunk = chunk[
                            chunk["PNODENAME"].str.split(".").str[0].isin(INDIANA_PREFIXES)
                        ]
                    if not chunk.empty:
                        kept.append(chunk)

    except (zipfile.BadZipFile, Exception) as e:
        log.warning("Could not parse ZIP for %s: %s", url, e)
        return None

    if not kept:
        log.info("  → %d raw rows scanned, 0 Indiana rows found", total_rows)
        return pd.DataFrame()

    df = pd.concat(kept, ignore_index=True)
    log.info("  → %d raw rows scanned, %d Indiana rows retained, columns: %s",
             total_rows, len(df), list(df.columns))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw MISO CSV DataFrame into our storage schema."""
    if df is None or df.empty:
        return pd.DataFrame()

    rename = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    if "interval_est" not in df.columns:
        log.warning("Expected column 'MKTHOUR_EST' not found. Columns: %s", list(df.columns))
        return pd.DataFrame()

    # Parse timestamp and convert to UTC
    df["interval_est"] = pd.to_datetime(df["interval_est"], errors="coerce")
    df = df.dropna(subset=["interval_est"])

    try:
        df["interval_utc"] = (
            df["interval_est"]
            .dt.tz_localize("America/New_York", ambiguous="infer")
            .dt.tz_convert("UTC")
        )
    except Exception as e:
        log.warning("Timezone conversion failed: %s", e)
        df["interval_utc"] = df["interval_est"]

    df["date"] = df["interval_utc"].dt.date.astype(str)

    for col in ["lmp", "congestion", "loss", "mlc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [c for c in ["node", "interval_est", "interval_utc", "date",
                         "lmp", "congestion", "loss", "mlc"]
            if c in df.columns]
    return df[keep]


# ── Storage ───────────────────────────────────────────────────────────────────

def save_partition(df: pd.DataFrame, date_str: str) -> None:
    """Write (or merge) one day's data into its Parquet partition."""
    path = DATA_DIR / f"date={date_str}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)

    dedup_cols = [c for c in ["node", "interval_utc"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")

    df.to_parquet(path, index=False, compression="snappy")


def save_weekly_data(df: pd.DataFrame) -> None:
    """Split a week's DataFrame by date and save each partition."""
    if df.empty:
        return
    total = 0
    for date_str, group in df.groupby("date"):
        save_partition(group.drop(columns=["date"]), date_str)
        total += len(group)
    log.info("  → Saved %d rows across %d date partitions",
             total, df["date"].nunique())


# ── Date range logic ──────────────────────────────────────────────────────────

def fetch_available_urls(start: date, end: date) -> list[str]:
    """Query the MISO market reports API for all weekly 5-min LMP ZIPs whose
    publish date falls in the window [start - 14 days, end + 14 days].
    Returns a sorted list of download URLs.
    """
    # Add a 14-day buffer on each side so we don't miss files at the edges.
    window_start = (start - timedelta(days=14)).isoformat()
    window_end   = (end   + timedelta(days=14)).isoformat()

    query = {
        "query": {
            "filtered": {
                "query": {
                    "term": {"MarketReportName$$string": "Weekly Real-Time 5-Min LMP (zip)"}
                },
                "filter": {
                    "and": [
                        {"range": {"MarketReportPublished$$string": {
                            "gte": window_start,
                            "lte": window_end,
                        }}}
                    ]
                }
            }
        },
        "sort": [{"MarketReportPublished$$string": "asc"}],
        "size": 200,
    }

    try:
        resp = requests.post(MISO_SEARCH_URL, json=query, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        urls = [
            h["_source"]["SearchHitUrl$$string"]
            for h in hits
            if "SearchHitUrl$$string" in h.get("_source", {})
        ]
        log.info("API returned %d matching weekly ZIP files", len(urls))
        return urls
    except Exception as e:
        log.warning("Could not query MISO search API (%s) — falling back to date guessing", e)
        return _fallback_urls(start, end)


def _fallback_urls(start: date, end: date) -> list[str]:
    """Generate candidate URLs by stepping weekly through the date window.
    Used if the MISO search API is unavailable.
    """
    candidates = []
    # MISO publishes on Mondays ~7-8 days after data week; add a small buffer.
    d = start + timedelta(days=6)
    end_window = end + timedelta(days=15)
    while d <= end_window:
        candidates.append(BASE_URL.format(date=d.strftime("%Y%m%d")))
        d += timedelta(days=7)
    return candidates


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill MISO 5-min LMP from weekly ZIPs")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",   action="store_true",
                       help="Pull all available history (from 2013 to today)")
    group.add_argument("--weeks", type=int, metavar="N",
                       help="Pull the last N weeks of data")
    group.add_argument("--start", type=str, metavar="YYYY-MM-DD",
                       help="Start date (use with --end)")
    p.add_argument("--end", type=str, metavar="YYYY-MM-DD",
                   help="End date (default: today). Required if --start is used.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print URLs without downloading")
    p.add_argument("--nodes", nargs="+", metavar="NODE",
                   help="Filter to specific node names (e.g. --nodes AMIL.AMIL1 CWLP.CWLP1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    today = date.today()

    # Determine data date range
    if args.all:
        start, end = EARLIEST_DATE, today
    elif args.weeks:
        end   = today
        start = today - timedelta(weeks=args.weeks)
    else:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end) if args.end else today

    log.info("Backfill range: %s → %s", start, end)

    urls = fetch_available_urls(start, end)
    log.info("ZIP files to download: %d", len(urls))

    if args.dry_run:
        for url in urls:
            print(url)
        return

    success, skipped, errors = 0, 0, 0

    for url in urls:
        raw = _fetch_zip_url(url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if raw is None:
            skipped += 1
            continue

        df = clean(raw)
        if df.empty:
            errors += 1
            continue

        # Filter to Indiana nodes (always applied)
        df = df[df["node"].str.split(".").str[0].isin(INDIANA_PREFIXES)]
        if df.empty:
            log.info("  → No Indiana nodes in this file, skipping save.")
            skipped += 1
            continue

        # Optional additional node filter
        if args.nodes:
            df = df[df["node"].isin(args.nodes)]
            if df.empty:
                log.info("  → No matching nodes in this file, skipping save.")
                skipped += 1
                continue

        # Clip to the requested data range
        if "date" in df.columns:
            df = df[
                (df["date"] >= str(start)) &
                (df["date"] <= str(end))
            ]

        save_weekly_data(df)
        success += 1

    log.info(
        "Backfill complete. Success: %d  |  Skipped/404: %d  |  Errors: %d",
        success, skipped, errors,
    )


if __name__ == "__main__":
    main()
