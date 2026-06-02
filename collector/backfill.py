"""
backfill.py
-----------
Downloads and processes MISO's weekly 5-minute LMP ZIP archives,
converting them into the same Parquet partition format used by fetch.py.

MISO publishes final (corrected) RT 5-min LMP as weekly ZIPs.
The filename date is ~2 weeks AFTER the data's actual operating week.

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

# MISO publishes weekly files; the file date is ~14 days after the data week.
# We step through candidate dates weekly and skip 404s gracefully.
PUBLISH_LAG_DAYS = 14       # how far ahead the filename date is vs. data date
STEP_DAYS        = 7        # weekly files
EARLIEST_DATE    = date(2013, 1, 1)   # approx. first available file

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "lmp"

# Column mapping from MISO ZIP CSVs to our schema
RENAME_MAP = {
    "PNODENAME":   "node",
    "MKTHOUR_EST": "interval_est",
    "LMP":         "lmp",
    "CON_LMP":     "congestion",
    "LOSS_LMP":    "loss",
    "MLC_LMP":     "mlc",
}

SLEEP_BETWEEN_REQUESTS = 1.5   # seconds — be polite to MISO's servers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Download & parse ──────────────────────────────────────────────────────────

def build_url(file_date: date) -> str:
    return BASE_URL.format(date=file_date.strftime("%Y%m%d"))


def fetch_zip(file_date: date) -> pd.DataFrame | None:
    """Download one weekly ZIP and return its contents as a DataFrame.
    Returns None if the file doesn't exist (404) or on network error."""
    url = build_url(file_date)
    log.info("Fetching %s", url)

    try:
        resp = requests.get(url, timeout=60)
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
            # Find the CSV inside (there should be exactly one)
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                log.warning("No CSV found inside %s", url)
                return None

            # MISO CSVs have 4 header rows and 1 footer row
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(
                    f,
                    skiprows=4,
                    skipfooter=1,
                    engine="python",
                )
    except (zipfile.BadZipFile, Exception) as e:
        log.warning("Could not parse ZIP for %s: %s", url, e)
        return None

    log.info("  → %d raw rows, columns: %s", len(df), list(df.columns))
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

def candidate_file_dates(start: date, end: date) -> list[date]:
    """Return the list of weekly file-dates whose ZIPs likely cover [start, end].

    Because the file date is ~14 days after the data, we add the lag to both
    ends and then step weekly through that window.
    """
    window_start = start + timedelta(days=PUBLISH_LAG_DAYS)
    window_end   = end   + timedelta(days=PUBLISH_LAG_DAYS + STEP_DAYS)

    # Align to the nearest Monday (MISO tends to publish on Mondays/Tuesdays)
    # A simple weekly step from window_start is good enough in practice.
    candidates = []
    d = window_start
    while d <= window_end:
        candidates.append(d)
        d += timedelta(days=STEP_DAYS)
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

    file_dates = candidate_file_dates(start, end)
    log.info("Candidate ZIP file dates: %d files to try", len(file_dates))

    if args.dry_run:
        for fd in file_dates:
            print(build_url(fd))
        return

    success, skipped, errors = 0, 0, 0

    for file_date in file_dates:
        raw = fetch_zip(file_date)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if raw is None:
            skipped += 1
            continue

        df = clean(raw)
        if df.empty:
            errors += 1
            continue

        # Optional node filter
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
