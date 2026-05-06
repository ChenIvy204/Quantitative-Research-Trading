# Quantitative-Research-Trading

## API Setup & Testing

This repository now includes a small setup for downloading full 2018-2024 data from two market data sources and saving the raw CSV outputs locally:

- Yahoo Finance via the `yfinance` package
- FRED via the FRED REST API

### 1. Create a local environment file

Copy [.env.example](.env.example) to `.env` and fill in your API keys.

Required values:

- `FRED_API_KEY`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API download script

```bash
python scripts/test_apis.py
```

The script downloads and saves:

- Yahoo Finance: daily bars for `JPM` from 2018-01-01 through 2024-12-31
- FRED: `DGS10` observations from 2018-01-01 through 2024-12-31
- FRED: `VIXCLS` observations from 2018-01-01 through 2024-12-31

The CSV files are written to `data/raw/`.

### Notes on API keys

The FRED key you provided is meant for local use only. Do not commit it into the repository.