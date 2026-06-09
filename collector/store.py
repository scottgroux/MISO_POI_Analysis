"""
store.py
--------
Shared helpers for reading from the Parquet data store.
Used by both analysis scripts and the site generator.

Storage layout (local and R2 mirror the same structure):
  data/lmp/date=YYYY-MM-DD/part.parquet

Read path (when R2 env vars are set):
  1. Check local cache  → return immediately if present
  2. Download from R2   → cache locally → return
  3. If neither exists  → return empty DataFrame

When R2 env vars are absent, reads from local cache only.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "lmp"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _local_path(date_str: str) -> Path:
    return DATA_DIR / f"date={date_str}" / "part.parquet"


def _try_r2_download(date_str: str, path: Path) -> bool:
    """Attempt to pull a partition from R2 into the local cache.
    Returns True if successful, False if R2 is disabled or the object is absent."""
    try:
        from r2 import download, r2_enabled
    except ImportError:
        return False
    if not r2_enabled():
        return False
    return download(date_str, path)


# ── Public API ────────────────────────────────────────────────────────────────

def available_dates() -> list[str]:
    """Return a sorted list of all date partitions.

    Sources (in priority order):
      1. R2 listing (authoritative when env vars are set)
      2. Local cache listing (fallback / offline mode)
    """
    try:
        from r2 import list_dates, r2_enabled
        if r2_enabled():
            dates = list_dates()
            if dates:
                return dates
    except ImportError:
        pass

    return sorted(
        p.name.replace("date=", "")
        for p in DATA_DIR.glob("date=*")
        if p.is_dir()
    )


def load_date(date_str: str) -> pd.DataFrame:
    """Load a single date partition.

    Checks local cache first; if absent, attempts R2 download.
    Returns empty DataFrame if data is not found anywhere.
    """
    path = _local_path(date_str)
    if path.exists():
        return pd.read_parquet(path)

    if _try_r2_download(date_str, path):
        return pd.read_parquet(path)

    return pd.DataFrame()


def load_range(
    start: date | str,
    end: date | str,
    nodes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load all partitions between start and end (inclusive).

    Args:
        start:  Start date as date object or 'YYYY-MM-DD' string.
        end:    End date as date object or 'YYYY-MM-DD' string.
        nodes:  Optional list of node names to filter to.

    Returns:
        Combined DataFrame sorted by (node, interval_utc).
    """
    start_str = str(start)
    end_str   = str(end)

    partitions = [
        d for d in available_dates()
        if start_str <= d <= end_str
    ]

    if not partitions:
        return pd.DataFrame()

    frames = []
    for date_str in partitions:
        df = load_date(date_str)
        if df.empty:
            continue
        if nodes:
            df = df[df["node"].isin(nodes)]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if "interval_utc" in combined.columns:
        combined = combined.sort_values(["node", "interval_utc"])

    return combined


def available_nodes(sample_dates: int = 5) -> list[str]:
    """Return all unique node names by sampling recent partitions."""
    dates = available_dates()[-sample_dates:]
    frames = [load_date(d) for d in dates]
    combined = pd.concat(frames, ignore_index=True)
    if "node" not in combined.columns:
        return []
    return sorted(combined["node"].unique().tolist())


def store_summary() -> dict:
    """Return a quick summary of what's in the store."""
    dates = available_dates()
    if not dates:
        return {"dates": 0, "earliest": None, "latest": None, "nodes": 0}

    sample = load_date(dates[-1])
    n_nodes = sample["node"].nunique() if "node" in sample.columns else "?"

    return {
        "dates":    len(dates),
        "earliest": dates[0],
        "latest":   dates[-1],
        "nodes":    n_nodes,
    }


if __name__ == "__main__":
    summary = store_summary()
    print("── Store summary ──────────────────────")
    for k, v in summary.items():
        print(f"  {k:<12} {v}")
    print()
    print("Recent dates:", available_dates()[-5:])
