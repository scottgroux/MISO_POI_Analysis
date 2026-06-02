"""
generate.py
-----------
Reads the Parquet data store and generates static Plotly HTML charts
into docs/ for GitHub Pages.

Run manually:   python site/generate.py
Auto-run:       Called by GitHub Actions after each collection cycle.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collector"))
from store import load_range, store_summary, available_nodes

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

PLOTLY_CONFIG = dict(responsive=True)

# Indiana hub and aggregation nodes — shown first in the dashboard
DEFAULT_HUB_NODES = [
    "INDIANA.HUB",
    "INDN",
    "IPL.IPL",
    "NIPS.NIPS",
    "SIGE.SIGW",
    "PSI_GEN.AGG",
]


def last_N_days(n: int = 7) -> tuple[date, date]:
    end   = date.today()
    start = end - timedelta(days=n)
    return start, end


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_hub_lmp_timeseries(df: pd.DataFrame, node: str) -> go.Figure:
    """Line chart of LMP over time for a single node."""
    node_df = df[df["node"] == node].copy()
    if node_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=node_df["interval_utc"],
        y=node_df["lmp"],
        mode="lines",
        name="LMP ($/MWh)",
        line=dict(width=1.5, color="#1f77b4"),
    ))

    if "congestion" in node_df.columns:
        fig.add_trace(go.Scatter(
            x=node_df["interval_utc"],
            y=node_df["congestion"],
            mode="lines",
            name="Congestion",
            line=dict(width=1, color="#ff7f0e", dash="dot"),
        ))

    fig.update_layout(
        title=f"RT 5-min LMP — {node}",
        xaxis_title="Interval (UTC)",
        yaxis_title="$/MWh",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
    )
    return fig


def chart_lmp_distribution(df: pd.DataFrame, node: str) -> go.Figure:
    """Histogram of LMP distribution for a single node."""
    node_df = df[df["node"] == node]
    if node_df.empty:
        return go.Figure()

    fig = go.Figure(go.Histogram(
        x=node_df["lmp"],
        nbinsx=60,
        marker_color="#1f77b4",
        opacity=0.8,
        name="LMP distribution",
    ))
    fig.update_layout(
        title=f"LMP Distribution — {node}",
        xaxis_title="$/MWh",
        yaxis_title="Interval count",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def chart_daily_avg_heatmap(df: pd.DataFrame, node: str) -> go.Figure:
    """Hour-of-day × Day-of-week average LMP heatmap."""
    node_df = df[df["node"] == node].copy()
    if node_df.empty or "interval_utc" not in node_df.columns:
        return go.Figure()

    node_df["hour"] = node_df["interval_utc"].dt.hour
    node_df["dow"]  = node_df["interval_utc"].dt.day_name()

    pivot = (
        node_df.groupby(["dow", "hour"])["lmp"]
        .mean()
        .unstack(fill_value=0)
    )
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values.round(2),
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        colorbar=dict(title="$/MWh"),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg LMP: $%{z:.2f}/MWh<extra></extra>",
    ))
    fig.update_layout(
        title=f"Avg LMP by Hour & Day — {node}",
        xaxis_title="Hour (UTC)",
        margin=dict(l=100, r=20, t=50, b=50),
    )
    return fig


# ── Page builder ──────────────────────────────────────────────────────────────

def build_index(summary: dict, nodes_available: list[str]) -> str:
    """Build the index.html landing page."""
    node_links = "\n".join(
        f'      <li><a href="node_{n.replace(".", "_")}.html">{n}</a></li>'
        for n in nodes_available[:50]   # cap at 50 for readability
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MISO POI Analysis</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    h1   {{ font-size: 1.6rem; }}
    .stats {{ display: flex; gap: 2rem; margin: 1.5rem 0; }}
    .stat {{ background: #f5f5f5; padding: 1rem 1.5rem; border-radius: 8px; }}
    .stat strong {{ display: block; font-size: 1.4rem; }}
    ul {{ column-count: 3; column-gap: 2rem; }}
    a  {{ color: #1a6fc4; }}
  </style>
</head>
<body>
  <h1>MISO POI LMP Analysis</h1>
  <p>Real-time 5-minute Locational Marginal Pricing — updated every 5 minutes.</p>

  <div class="stats">
    <div class="stat"><strong>{summary.get('dates', '—')}</strong>Days collected</div>
    <div class="stat"><strong>{summary.get('earliest', '—')}</strong>Earliest date</div>
    <div class="stat"><strong>{summary.get('latest', '—')}</strong>Latest date</div>
    <div class="stat"><strong>{summary.get('nodes', '—')}</strong>Nodes (latest day)</div>
  </div>

  <h2>Nodes</h2>
  <ul>
{node_links}
  </ul>
</body>
</html>
"""


def build_node_page(fig_ts: go.Figure, fig_hist: go.Figure,
                    fig_heat: go.Figure, node: str) -> str:
    """Build a per-node HTML page embedding three Plotly charts."""
    ts_html   = pio.to_html(fig_ts,   full_html=False, config=PLOTLY_CONFIG)
    hist_html = pio.to_html(fig_hist, full_html=False, config=PLOTLY_CONFIG)
    heat_html = pio.to_html(fig_heat, full_html=False, config=PLOTLY_CONFIG)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MISO LMP — {node}</title>
  <style>
    body   {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 1rem auto; padding: 0 1rem; }}
    h1     {{ font-size: 1.4rem; }}
    .chart {{ margin-bottom: 2rem; }}
    a      {{ color: #1a6fc4; }}
  </style>
</head>
<body>
  <p><a href="index.html">← All nodes</a></p>
  <h1>{node}</h1>
  <p>Last 7 days of RT 5-minute ExPost LMP data.</p>

  <div class="chart">{ts_html}</div>
  <div class="chart">{hist_html}</div>
  <div class="chart">{heat_html}</div>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Site generator starting …")
    summary = store_summary()
    print(f"  Store: {summary}")

    if summary["dates"] == 0:
        print("  No data in store yet. Run fetch.py or backfill.py first.")
        # Write a placeholder index so GitHub Pages has something to show
        (DOCS_DIR / "index.html").write_text(
            "<h1>MISO POI Analysis</h1><p>Data collection in progress…</p>"
        )
        return

    start, end = last_N_days(7)
    nodes_to_render = available_nodes(sample_dates=3)

    # Prefer hub nodes first; fall back to whatever is available
    priority = [n for n in DEFAULT_HUB_NODES if n in nodes_to_render]
    others   = [n for n in nodes_to_render if n not in priority]
    render_nodes = (priority + others)[:50]   # cap to keep CI fast

    print(f"  Loading {start} → {end} for {len(render_nodes)} nodes …")
    df = load_range(start, end, nodes=render_nodes if render_nodes else None)

    if df.empty:
        print("  DataFrame is empty — nothing to chart.")
        return

    # Ensure interval_utc is tz-aware for Plotly
    if "interval_utc" in df.columns and df["interval_utc"].dt.tz is None:
        df["interval_utc"] = pd.to_datetime(df["interval_utc"], utc=True)

    # Render per-node pages
    for node in render_nodes:
        if df[df["node"] == node].empty:
            continue
        slug = node.replace(".", "_")
        fig_ts   = chart_hub_lmp_timeseries(df, node)
        fig_hist = chart_lmp_distribution(df, node)
        fig_heat = chart_daily_avg_heatmap(df, node)
        page     = build_node_page(fig_ts, fig_hist, fig_heat, node)
        (DOCS_DIR / f"node_{slug}.html").write_text(page)
        print(f"  ✓  {node}")

    # Build index
    index_html = build_index(summary, render_nodes)
    (DOCS_DIR / "index.html").write_text(index_html)
    print(f"\nSite written to {DOCS_DIR}/")


if __name__ == "__main__":
    main()
