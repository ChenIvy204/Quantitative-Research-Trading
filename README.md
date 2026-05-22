# Quantitative-Research-Trading

## API Setup & Testing

This repository now includes a small setup for downloading market data and Alpha Vantage news sentiment data and saving the raw CSV outputs locally:

- Yahoo Finance via the `yfinance` package
- Alpha Vantage `NEWS_SENTIMENT` for market news and sentiment
- FRED via the FRED REST API

### 1. Create a local environment file

Copy [.env.example](.env.example) to `.env` and fill in your API keys.

Required values:

- `ALPHA_VANTAGE_API_KEY`
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
- Alpha Vantage: recent JPM-related market news and sentiment articles
- FRED: `DGS10` observations from 2018-01-01 through 2024-12-31
- FRED: `VIXCLS` observations from 2018-01-01 through 2024-12-31

### 4. Build Week 2 features

This creates versioned outputs such as `data/processed/week2_feature_dataset_v1.0_YYYYMMDD.csv` and `data/processed/week2_feature_dataset_v1.0_YYYYMMDD.parquet` with the optimized feature set after correlation pruning and IC screening.
It also creates `data/processed/week2_data_quality_report_v1.0_YYYYMMDD.csv` and `data/processed/week2_data_quality_report_v1.0_YYYYMMDD.pdf`.
The feature engineering optimization outputs are `data/processed/week2_feature_optimization_report_v1.0_YYYYMMDD.csv`, `data/processed/week2_feature_ic_report_v1.0_YYYYMMDD.csv`, `data/processed/week2_feature_correlation_matrix_v1.0_YYYYMMDD.csv`, and `data/processed/week2_feature_optimization_report_v1.0_YYYYMMDD.pdf`.

Week 2 features include:

- JPM daily returns and rolling volatility
- Trailing dividend yield for JPM using TTM dividend divided by current price
- 5-day, 20-day, and 60-day historical volatility features
- Volatility change features based on the 20-day volatility series
- Treasury rate changes and momentum
- VIX changes and JPM-VIX rolling correlation
- News article counts and rolling 0-1 sentiment scores when news CSV files are available
- A data quality report with per-feature missing rate, min, max, mean, std, and outlier counts
- Boxplot-based outlier handling with median replacement on flagged values
- Explicit missing-value rules for price, macro, dividend, and news-derived features
- Pearson correlation pruning for features with absolute correlation above 0.8
- IC screening against future 1-week and 1-month JPM returns using a 0.03 absolute threshold

### 5. Automated scheduling

The repository includes a GitHub Actions workflow at [.github/workflows/preprocess.yml](.github/workflows/preprocess.yml) that runs the full pipeline daily and on push to `main`.

It expects these GitHub Secrets:

- `ALPHA_VANTAGE_API_KEY`
- `FRED_API_KEY`

The workflow uploads the processed CSV and Parquet files as artifacts after a successful run.

### 6. Week 3 chooser-option replication

Week 3 adds a reusable Black-Scholes chooser-option implementation and validation outputs based on the paper's JPM parameter table.

- Core code: [scripts/week3_bsm.py](scripts/week3_bsm.py)
- Paper parameters: [configs/week3_bsm_parameters.json](configs/week3_bsm_parameters.json)
- Replication notebook: [notebooks/week3_bsm_replication.ipynb](notebooks/week3_bsm_replication.ipynb)
- Validation notebook: [notebooks/week3_bsm_validation.ipynb](notebooks/week3_bsm_validation.ipynb)

Run it directly with:

```bash
python scripts/week3_bsm.py
```

The script writes a versioned CSV summary under `data/processed/` and a Markdown validation report under `data/reports/`.
It also writes a PDF version of the same report, with markdown tables rendered as standalone formatted tables instead of raw pipe text.
The Week 3 report now includes the Table 2 inputs, an explicit simulation setup section with the random seed and GBM drift assumption, a paper reference table, and split comparison tables so the row-by-row differences are easier to read.

### Feature engineering reference

See [docs/feature_engineering.md](docs/feature_engineering.md) for a concise description of the engineered features and preprocessing steps.

### Notes on API keys

The FRED key you provided is meant for local use only. Do not commit it into the repository.
Alpha Vantage provides the market news and sentiment feed used by the pipeline.