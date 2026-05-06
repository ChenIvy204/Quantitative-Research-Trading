# Quantitative-Research-Trading

## API Setup & Testing

This repository now includes a small setup for downloading full 2018-2024 data from market data and news sources and saving the raw CSV outputs locally:

- Yahoo Finance via the `yfinance` package
- News API for recent article sentiment inputs, with GDELT fallback for historical coverage
- FRED via the FRED REST API

### 1. Create a local environment file

Copy [.env.example](.env.example) to `.env` and fill in your API keys.

Required values:

- `NEWS_API_KEY`
- `FRED_API_KEY`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python scripts/pipeline.py
```

This is the only command you need to run locally. It executes the full workflow in order:

- Week 1 data downloads
- Week 2 preprocessing and feature engineering

The raw and processed CSV files are written to `data/raw/` and `data/processed/`.

`scripts/apis.py` and `scripts/preprocess.py` are internal modules used by the pipeline; you do not need to run them separately.

Week 1 downloads and saves:

- Yahoo Finance: daily bars for `JPM` from 2018-01-01 through 2024-12-31
- News API: JPMorgan Chase related articles from the recent rolling window the plan supports
- GDELT: JPMorgan Chase related articles from 2018-01-01 through 2024-12-31 when News API archive access is unavailable
- FRED: `DGS10` observations from 2018-01-01 through 2024-12-31
- FRED: `VIXCLS` observations from 2018-01-01 through 2024-12-31

### 4. Build Week 2 features

This creates `data/processed/week2_feature_dataset_2018_2024.csv` with aligned JPM, Treasury, VIX, and optional news sentiment features.

Week 2 features include:

- JPM daily returns and rolling volatility
- Treasury rate changes and momentum
- VIX changes and JPM-VIX rolling correlation
- News article counts and rolling sentiment scores when news CSV files are available
- IQR clipping on key return and change series to reduce outlier impact

### 5. Automated scheduling

The repository includes a GitHub Actions workflow at [.github/workflows/preprocess.yml](.github/workflows/preprocess.yml) that runs the full pipeline daily and on push to `main`.

It expects these GitHub Secrets:

- `NEWS_API_KEY`
- `FRED_API_KEY`

The workflow uploads the processed CSV as an artifact after a successful run.

### Notes on API keys

The FRED key you provided is meant for local use only. Do not commit it into the repository.

### News API limitation

NewsAPI's free or non-archive plans can return `426 Upgrade Required` for historical requests. If you see that message, the key is valid but the plan does not support the 2018-2024 archive range. In that case, the script uses GDELT for the historical sentiment set and still uses NewsAPI for recent articles when possible.

When that happens, the script falls back to GDELT automatically so the sentiment data download still completes.