# MISO_POI_Analysis

Real-time and historical MISO 5-minute Locational Marginal Price (LMP) tracker
with automated collection, Parquet storage, and a static GitHub Pages dashboard.

---

## Project structure

```
MISO_POI_Analysis/
├── collector/
│   ├── fetch.py        # polls MISO RT API every 5 min
│   ├── backfill.py     # downloads historical weekly ZIP archives
│   ├── scheduler.py    # local continuous runner (not needed on Render)
│   └── store.py        # shared Parquet read helpers
├── analysis/           # ad-hoc notebooks and scripts (add your own)
├── site/
│   └── generate.py     # builds static HTML charts → docs/
├── data/lmp/           # Parquet store (gitignored by default)
├── docs/               # GitHub Pages output
├── .github/workflows/
│   └── update.yml      # GitHub Actions: collect → generate → deploy
└── requirements.txt
```

---

## Quick start (local)

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/YOUR_USER/MISO_POI_Analysis.git
cd MISO_POI_Analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Inspect the live API response (do this first!)
python collector/fetch.py --inspect

# 3. Run one collection cycle
python collector/fetch.py

# 4. Backfill the last 8 weeks of history
python collector/backfill.py --weeks 8

# 5. Check what's in the store
python collector/store.py

# 6. Generate the site locally
python site/generate.py
# Open docs/index.html in your browser
```

---

## Backfill options

```bash
# Last N weeks
python collector/backfill.py --weeks 12

# Specific date range
python collector/backfill.py --start 2024-01-01 --end 2024-12-31

# All available history (~2013 onwards — takes a while)
python collector/backfill.py --all

# Specific nodes only (much faster for targeted analysis)
python collector/backfill.py --weeks 8 --nodes AMIL.AMIL1 CWLP.CWLP1

# Dry run — print URLs without downloading
python collector/backfill.py --weeks 4 --dry-run
```

---

## GitHub Actions (automated updates)

1. Enable GitHub Pages in repo Settings → Pages → Source: **Deploy from branch**,
   branch: `main`, folder: `/docs`.

2. Create a Personal Access Token (PAT):
   GitHub → Settings → Developer settings → Personal access tokens → Fine-grained →
   New token → give it **Contents: Read & Write** on this repo.

3. Add it as a secret:
   Repo → Settings → Secrets and variables → Actions → New repository secret →
   Name: `GH_PAT`, Value: your token.

4. The workflow in `.github/workflows/update.yml` will run every 5 minutes,
   collect new data, regenerate charts, and push to `docs/`.

**Note:** GitHub Actions free tier gives 2,000 minutes/month. Running every
5 minutes = ~8,640 runs/month × ~30 seconds each ≈ 4,320 minutes — this
exceeds the free limit. Options:
- Use a 15-minute cron instead: `*/15 * * * *`
- Use Render's free Cron Job for collection, and only trigger site generation
  on push (much cheaper on Actions minutes).

---

## Render Cron Job (recommended for collection)

1. Create a new **Cron Job** service on Render.
2. Connect this GitHub repo.
3. Set the build command: `pip install -r requirements.txt`
4. Set the cron schedule: `*/5 * * * *`
5. Set the run command: `python collector/fetch.py`
6. Add environment variable `GITHUB_TOKEN` pointing to your PAT so the job
   can push collected data back to the repo.

---

## Important first step — inspect the API

MISO updated their RT API in December 2025. Before running the collector in
production, always run:

```bash
python collector/fetch.py --inspect
```

This prints the raw column names from the live API response. If the column names
differ from what's in `RENAME_MAP` inside `fetch.py`, update the mapping there.

---

## Data schema

Each Parquet partition (`data/lmp/date=YYYY-MM-DD/part.parquet`) contains:

| Column         | Type      | Description                              |
|----------------|-----------|------------------------------------------|
| `node`         | str       | Pricing node name (PNODENAME)            |
| `interval_est` | datetime  | Interval start in Eastern time           |
| `interval_utc` | datetime  | Interval start in UTC (primary key col)  |
| `lmp`          | float     | Total LMP ($/MWh)                        |
| `congestion`   | float     | Congestion component                     |
| `loss`         | float     | Loss component                           |
| `mlc`          | float     | Marginal loss component (if available)   |
