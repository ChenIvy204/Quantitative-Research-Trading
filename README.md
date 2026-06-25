# Quantitative Research Trading

This repository contains a multi-week quantitative research workflow built around JPMorgan market data, macro data, option pricing, feature engineering, and machine learning models.

The project is organized as a reproducible pipeline:

1. Week 1 collects raw market, macro, dividend, and news data.
2. Week 2 cleans the data, engineers features, and exports quality reports.
3. Week 3 replicates the chooser-option model from the paper.
4. Week 4 evaluates Black-Scholes pricing against a Monte Carlo benchmark.
5. Week 5 / Week 6 trains machine learning models for volatility prediction and chooser pricing.
6. Week 7 wraps the pricing workflow in a Streamlit app, FastAPI service, and report view.

## Repository Layout

- `scripts/` - main pipeline and analysis scripts
- `configs/` - configuration files used by the pricing experiments
- `data/raw/` - downloaded source data
- `data/processed/` - cleaned datasets and generated tables
- `data/models/` - trained model artifacts
- `data/reports/` - generated Markdown, PDF, and plot reports
- `notebooks/` - supporting notebooks for the chooser-option work
- `docs/` - documentation for feature engineering and preprocessing
- `streamlit_app.py` - thin launcher for the Week 7 Streamlit app
- `report_app.py` - standalone Streamlit report view for Week 7 outputs
- `requirements.txt` - Python dependencies

## Setup

The project is designed to run locally with Python 3.12, which is also what the GitHub Actions workflow uses.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file in the repository root and add the required API keys:

```env
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

The pipeline can still run partially if a feed is unavailable, but the Week 1 downloads and downstream sentiment features depend on those keys.
Yahoo Finance data is collected through `yfinance` and does not require an API key.

## Quick Start

If you are running the project for the first time:

1. Install dependencies with `pip install -r requirements.txt`.
2. Create a `.env` file with the API keys shown above.
3. Run the full pipeline with `python scripts/pipeline.py`.
4. Launch the Week 7 pricing tool with `streamlit run scripts/week7_app.py`.

The first pipeline run downloads source data automatically, then builds the processed datasets, reports, and model artifacts used by the later stages.

## Run the Full Pipeline

Run everything from the repository root:

```bash
python scripts/pipeline.py
```

This executes the workflow in order:

1. Week 1 data downloads
2. Week 2 preprocessing and feature engineering
3. Week 3 chooser-option replication
4. Week 4 BSM evaluation
5. Week 5 / Week 6 machine learning models

The pipeline writes versioned datasets, reports, and model artifacts into `data/raw/`, `data/processed/`, `data/models/`, and `data/reports/`.

## Run Individual Components

If you only need one stage, you can run it directly:

```bash
python scripts/apis.py
python scripts/preprocess.py
python scripts/week3_bsm.py
python scripts/week4_bsm_evaluation.py
python scripts/week5_ml_models.py
python scripts/week7_toolkit.py
```

Week 7 has a main UI and two convenience launchers:

```bash
streamlit run scripts/week7_app.py
streamlit run streamlit_app.py
streamlit run report_app.py
uvicorn scripts.week7_api:app --reload
```

`scripts/week7_app.py` is the main pricing UI. `streamlit_app.py` is a root-level convenience launcher for the same app, and `report_app.py` opens a lighter report view that focuses on Week 7 outputs.

## What Each Week Produces

### Week 1 - Data Collection

Downloads and caches source data such as:

- JPM daily prices from Yahoo Finance
- JPM-related news sentiment from Alpha Vantage
- `DGS10` and `VIXCLS` from FRED
- JPM dividend history

Raw files are written to `data/raw/`.

### Week 2 - Feature Engineering

Builds the modeling table and the associated quality/optimization reports.

Key outputs include the feature dataset and the quality / optimization reports:

- `data/processed/week2_feature_dataset_v1.0_YYYYMMDD.csv`
- `data/processed/week2_feature_dataset_v1.0_YYYYMMDD.parquet`
- `data/processed/week2_data_quality_report_v1.0_YYYYMMDD.csv`
- `data/processed/week2_data_quality_report_v1.0_YYYYMMDD.md`
- `data/processed/week2_data_quality_report_v1.0_YYYYMMDD.pdf`
- `data/processed/week2_feature_optimization_report_v1.0_YYYYMMDD.csv`
- `data/processed/week2_feature_optimization_report_v1.0_YYYYMMDD.pdf`

The full feature reference is documented in [docs/feature_engineering.md](docs/feature_engineering.md).

The engineered feature set includes rolling volatility, dividend yield, Treasury and VIX changes, correlation pruning, IC screening, and news sentiment features when the news feed is available.

### Week 3 - Chooser-Option Replication

Reimplements the Black-Scholes chooser-option setup and compares the simulation results against the paper reference values.

Inputs and supporting files:

- [scripts/week3_bsm.py](scripts/week3_bsm.py)
- [configs/week3_bsm_parameters.json](configs/week3_bsm_parameters.json)
- [notebooks/week3_bsm_replication.ipynb](notebooks/week3_bsm_replication.ipynb)
- [notebooks/week3_bsm_validation.ipynb](notebooks/week3_bsm_validation.ipynb)

Outputs are written to `data/processed/` and `data/reports/` as versioned CSV, Markdown, and PDF files.

### Week 4 - BSM Evaluation

Evaluates analytical Black-Scholes prices against Monte Carlo benchmark prices and runs additional error analysis.

Outputs include:

- `data/processed/week4_bsm_evaluation_daily_v1.0_YYYYMMDD.csv`
- `data/processed/week4_bsm_error_metrics_v1.0_YYYYMMDD.csv`
- `data/processed/week4_bsm_sentiment_gap_v1.0_YYYYMMDD.csv`
- `data/reports/week4_bsm_error_timeseries.png`
- `data/reports/week4_bsm_regime_boxplot.png`
- `data/reports/week4_bsm_sentiment_scatter.png`
- `data/reports/week4_bsm_validation_v1.0_YYYYMMDD.md`

### Week 5 / Week 6 - Machine Learning Models

Trains two model families:

- Approach 1: predict forward volatility and feed it into the chooser pricing formula
- Approach 2: predict chooser prices directly from market and contract features

Generated artifacts include:

- `data/processed/week6_feature_dataset_v1.0.csv`
- `data/processed/week6_vol_results_v1.0.csv`
- `data/processed/week6_pricing_results_v1.0.csv`
- `data/processed/week6_pricing_stratified_v1.0.csv`
- `data/processed/week6_model_comparison_v1.0.csv`
- `data/models/week6_*.joblib`
- `data/models/week6_*.keras`
- `data/reports/week6_ml_architecture_v1.0.md`
- `data/reports/week6_ml_architecture_v1.0.pdf`
- `data/reports/week6_feature_importance.png`
- `data/reports/week6_shap_app1_*.png`
- `data/reports/week6_shap_app2_*.png`
- `data/reports/week6_vol_prediction_comparison.png`
- `data/reports/week6_pricing_comparison.png`
- `data/reports/week6_model_performance.png`

The Week 6 script can optionally use XGBoost, TensorFlow, and SHAP. These are already listed in `requirements.txt`, but the script also degrades gracefully if a specific optional dependency cannot be imported.

### Week 7 - Pricing Tool and API

Adds a small tool layer around the Week 6 chooser model artifacts.

- [scripts/week7_toolkit.py](scripts/week7_toolkit.py) runs the analysis workflow
- [scripts/week7_app.py](scripts/week7_app.py) provides the Streamlit pricing UI
- [scripts/week7_api.py](scripts/week7_api.py) exposes a FastAPI service

The Week 7 workflow writes sensitivity, scenario stress, SHAP summary, and report outputs to `data/processed/` and `data/reports/`.

## Automation

There are two GitHub Actions workflows:

- [.github/workflows/preprocess.yml](.github/workflows/preprocess.yml) runs the full pipeline daily at 02:00 UTC and on pushes to `main`
- [.github/workflows/week7_refresh.yml](.github/workflows/week7_refresh.yml) refreshes market data and rebuilds the Week 7 outputs daily at 03:00 UTC and on pushes to `main`
- Both workflows can also be triggered manually from the GitHub Actions tab via `workflow_dispatch`

Both workflows expect these repository secrets:

- `ALPHA_VANTAGE_API_KEY`
- `FRED_API_KEY`

If a workflow run fails, the most common cause is a missing secret or an upstream data source issue. The pipeline is written to tolerate some data-source gaps, but it still needs the secrets configured to refresh the raw feeds.

## Configuration

Key configuration files and tuning locations:

- `configs/week3_bsm_parameters.json` - paper parameter table used by the Week 3 chooser replication
- `scripts/week5_ml_models.py` - model definitions, training settings, and feature handling for Weeks 5 and 6
- `scripts/week7_toolkit.py` - Week 7 pricing workflow, sensitivity analysis, and scenario settings

## Notes

- Keep API keys out of version control.
- Run commands from the repository root so the relative imports resolve correctly.
- Versioned outputs use the current date stamp, so reruns will create new files instead of overwriting historical results.