"""
scheduler.py
------------
Runs the collector on a 5-minute loop.

  Local dev:   python collector/scheduler.py
  Render:      Set this as the start command, OR use a Render Cron Job
               pointing at `python collector/fetch.py` (preferred for Render).

On Render's free Cron Job tier, you don't need this file — Render itself
handles the schedule. Use scheduler.py only for local continuous runs.
"""

import logging
import time
import traceback
from datetime import datetime

import schedule

from fetch import run as collect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

INTERVAL_MINUTES = 5


def safe_collect() -> None:
    """Wrap collect() so a single failure doesn't kill the scheduler."""
    try:
        collect()
    except Exception:
        log.error("Collection failed:\n%s", traceback.format_exc())


def main() -> None:
    log.info("Scheduler starting — collecting every %d minutes.", INTERVAL_MINUTES)

    # Run immediately on startup to get the first data point right away
    safe_collect()

    schedule.every(INTERVAL_MINUTES).minutes.do(safe_collect)

    while True:
        schedule.run_pending()
        time.sleep(30)   # wake every 30s to check for pending jobs


if __name__ == "__main__":
    main()
