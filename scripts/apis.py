from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
START_DATE = "2018-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
load_dotenv(ROOT / ".env")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def ensure_output_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, filename: str) -> Path:
    ensure_output_dir()
    path = DATA_DIR / filename
    frame.to_csv(path, index=True)
    return path


def retry_request(url: str, *, params: dict[str, str], retries: int = 3) -> requests.Response:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
    }
    last_error: Exception | None = None

    for _ in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except Exception as error:  # noqa: BLE001 - network failures are expected here
            last_error = error

    assert last_error is not None
    raise last_error


def test_yahoo_finance() -> bool:
    start_ts = int(datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.fromisoformat(END_DATE_EXCLUSIVE).replace(tzinfo=timezone.utc).timestamp())
    url = "https://query1.finance.yahoo.com/v8/finance/chart/JPM"
    params = {
        "period1": str(start_ts),
        "period2": str(end_ts),
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
        "lang": "en-US",
        "region": "US",
    }

    try:
        response = retry_request(url, params=params)
        payload = response.json()
    except Exception as error:  # noqa: BLE001 - network failures are expected here
        print(f"[WARN] Yahoo Finance unavailable for JPM: {error}")
        return False

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        fail(f"Yahoo Finance error: {error.get('description', error)}")

    result = chart.get("result", [])
    if not result:
        fail("Yahoo Finance returned no chart result for JPM")

    data = result[0]
    timestamps = data.get("timestamp", [])
    quote = data.get("indicators", {}).get("quote", [{}])[0]
    adjclose = data.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

    if not timestamps or not quote:
        fail("Yahoo Finance returned incomplete chart data for JPM")

    history = pd.DataFrame(quote)
    history.index = pd.to_datetime(timestamps, unit="s")
    history.index.name = "Date"

    if adjclose:
        history["Adj Close"] = adjclose

    history = history.loc[START_DATE:END_DATE_EXCLUSIVE]

    if history.empty:
        print("[WARN] Yahoo Finance returned no price history for JPM in 2018-2024")
        return False

    if not isinstance(history, pd.DataFrame):
        print("[WARN] Yahoo Finance did not return a DataFrame")
        return False

    path = save_csv(history, "yahoo_jpm_2018_2024.csv")
    ok(f"Yahoo Finance returned {len(history)} rows for JPM and saved {path.name}")
    return True


def test_fred(api_key: str, series_id: str, filename: str) -> None:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": "2024-12-31",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    observations = payload.get("observations", [])
    if not observations:
        fail(f"FRED returned no observations for {series_id} in 2018-2024")

    frame = pd.DataFrame(observations)
    if frame.empty:
        fail(f"FRED returned an empty DataFrame for {series_id}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date")
    frame = frame.sort_index()

    path = save_csv(frame, filename)
    ok(f"FRED returned {len(frame)} rows for {series_id} and saved {path.name}")


def main() -> None:
    fred_key = os.getenv("FRED_API_KEY", "").strip()

    if not fred_key:
        fail("Missing FRED_API_KEY")

    yahoo_ok = test_yahoo_finance()
    test_fred(fred_key, "DGS10", "fred_DGS10_2018_2024.csv")
    test_fred(fred_key, "VIXCLS", "fred_VIXCLS_2018_2024.csv")

    if yahoo_ok is False:
        print("[WARN] Yahoo Finance download did not complete, but FRED did.")
    else:
        ok("All API downloads completed successfully")


if __name__ == "__main__":
    main()