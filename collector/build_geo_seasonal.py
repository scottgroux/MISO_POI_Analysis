"""
build_geo_seasonal.py
----------------------
One-off helper: builds summaries/geo_seasonal.json from the per-node
seasonal.json files already on R2, without re-streaming all partitions.

Going forward, summarize.py generates geo_seasonal.json directly during its
nightly run (see compute_geo_seasonal). Use this script only to backfill
geo_seasonal.json for an existing deployment.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2 import download_json, upload_json, r2_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def main() -> None:
    if not r2_enabled():
        log.error("R2 env vars not set — aborting.")
        sys.exit(1)

    meta = download_json("summaries/meta.json")
    nodes = meta["nodes"]
    log.info("Building geo_seasonal.json for %d nodes…", len(nodes))

    slots = None
    geo_nodes = {}
    for i, node in enumerate(nodes, 1):
        seasonal = download_json(f"summaries/{node}/seasonal.json")
        if seasonal is None:
            log.warning("  [%d/%d] %s: no seasonal.json found, skipping", i, len(nodes), node)
            continue
        if slots is None:
            slots = seasonal["slots"]
        geo_nodes[node] = {m: seasonal["months"][m]["avg"] for m in MONTH_ABBR}
        log.info("  [%d/%d] %s done", i, len(nodes), node)

    upload_json({"slots": slots, "nodes": geo_nodes}, "summaries/geo_seasonal.json")
    log.info("Done. geo_seasonal.json covers %d nodes.", len(geo_nodes))


if __name__ == "__main__":
    main()
