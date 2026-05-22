from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
import yfinance as yf


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


def sanitize_filename(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def iter_year_windows(start_date: str, end_date_exclusive: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date_exclusive).normalize()
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    current = start
    while current < end:
        next_year = pd.Timestamp(year=current.year + 1, month=1, day=1)
        window_end = min(next_year, end)
        windows.append((current, window_end))
        current = window_end

    return windows


def get_with_retry(
    url: str,
    *,
    params: dict[str, str],
    headers: dict[str, str] | None = None,
    retries: int = 4,
    base_sleep: float = 2.0,
) -> requests.Response | None:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return requests.get(url, params=params, headers=headers, timeout=30)
        except requests.exceptions.RequestException as error:
            last_error = error
            if attempt < retries:
                time.sleep(base_sleep * attempt)

    if last_error is not None:
        print(f"[WARN] Request failed after {retries} attempts: {last_error}")

    return None


def normalize_news_results(
    results: list[dict[str, object]],
    ticker: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    if frame.empty:
        return frame

    frame["publishedAt"] = pd.to_datetime(frame.get("time_published"), format="%Y%m%dT%H%M%S", errors="coerce", utc=True)
    frame["publishedAt"] = frame["publishedAt"].dt.tz_convert(None)

    frame["source_name"] = frame["source"] if "source" in frame.columns else None
    if "authors" in frame.columns:
        frame["author"] = frame["authors"].apply(lambda value: ", ".join(value) if isinstance(value, list) else value)
    else:
        frame["author"] = None

    frame["url"] = frame["url"] if "url" in frame.columns else None
    frame["description"] = frame["summary"] if "summary" in frame.columns else None
    frame["content"] = frame["summary"] if "summary" in frame.columns else None
    frame["title"] = frame["title"] if "title" in frame.columns else None

    frame["overall_sentiment_score"] = pd.to_numeric(frame.get("overall_sentiment_score"), errors="coerce")
    frame["overall_sentiment_label"] = frame.get("overall_sentiment_label")

    def extract_ticker_sentiment(value: object) -> tuple[float | None, str | None, float | None]:
        if isinstance(value, list):
            for entry in value:
                if str(entry.get("ticker", "")).upper() == ticker.upper():
                    score = pd.to_numeric(entry.get("ticker_sentiment_score"), errors="coerce")
                    label = entry.get("ticker_sentiment_label")
                    relevance = pd.to_numeric(entry.get("relevance_score"), errors="coerce")
                    return (
                        None if pd.isna(score) else float(score),
                        None if label is None else str(label),
                        None if pd.isna(relevance) else float(relevance),
                    )
        return None, None, None

    ticker_sentiment = frame.get("ticker_sentiment", pd.Series(dtype=object)).apply(extract_ticker_sentiment)
    ticker_sentiment_frame = pd.DataFrame(
        ticker_sentiment.tolist(),
        columns=["ticker_sentiment_score", "ticker_sentiment_label", "ticker_relevance_score"],
        index=frame.index,
    )
    frame = pd.concat([frame, ticker_sentiment_frame], axis=1)

    frame["sentiment_score"] = frame["ticker_sentiment_score"].combine_first(frame["overall_sentiment_score"])

    frame = frame.dropna(subset=["publishedAt"])
    frame = frame[(frame["publishedAt"] >= window_start) & (frame["publishedAt"] < window_end)]
    frame = frame.sort_values("publishedAt")

    if frame.empty:
        return frame

    frame = frame[
        [
            "publishedAt",
            "source_name",
            "author",
            "title",
            "description",
            "url",
            "content",
            "overall_sentiment_score",
            "overall_sentiment_label",
            "ticker_sentiment_score",
            "ticker_sentiment_label",
            "ticker_relevance_score",
            "sentiment_score",
        ]
    ].copy()
    frame["ticker"] = ticker
    frame["sentiment_window"] = f"{window_start:%Y-%m-%d}_to_{window_end:%Y-%m-%d}"
    return frame


def fetch_alpha_vantage_news(api_key: str) -> bool:
    existing_path = DATA_DIR / "alphavantage_news_jpm_2018_2024.csv"
    if existing_path.exists():
        ok(f"Alpha Vantage 2018-2024 news file already exists and will be reused: {existing_path.name}")
        return True

    url = "https://www.alphavantage.co/query"
    ticker = "JPM"
    all_articles: list[dict[str, object]] = []
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "sort": "LATEST",
        "limit": "1000",
        "apikey": api_key,
    }

    response = get_with_retry(url, params=params, retries=4, base_sleep=2.5)
    if response is None:
        print("[WARN] Skipping Alpha Vantage latest endpoint after repeated connection failures.")
        return False

    if not response.ok:
        snippet = response.text.strip().replace("\n", " ")[:300]
        print(f"[WARN] Alpha Vantage HTTP {response.status_code}: {snippet}")
        return False

    payload = response.json()
    results = payload.get("feed", [])
    if not results:
        print("[WARN] Alpha Vantage returned no news items for the latest request")
        return False

    normalized = normalize_news_results(
        results,
        ticker,
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("2100-01-01"),
    )
    if not normalized.empty:
        all_articles.extend(normalized.to_dict(orient="records"))

    if not all_articles:
        print("[WARN] Alpha Vantage returned no JPM news articles for the latest request")
        return False

    frame = pd.DataFrame(all_articles)
    frame = frame.sort_values("publishedAt")
    filename = "alphavantage_news_jpm_2018_2024.csv"
    path = save_csv(frame, filename)
    ok(f"Alpha Vantage NEWS_SENTIMENT returned {len(frame)} articles for JPM and saved {path.name}")
    return True


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

    yahoo_cache = DATA_DIR / "yahoo_jpm_2018_2024.csv"
    dividend_cache = DATA_DIR / "jpm_dividends_2018_2024.csv"

    try:
        response = retry_request(url, params=params)
        payload = response.json()
    except Exception as error:  # noqa: BLE001 - network failures are expected here
        print(f"[WARN] Yahoo Finance unavailable for JPM: {error}")
        if yahoo_cache.exists():
            ok(f"Yahoo Finance cache already exists and will be reused: {yahoo_cache.name}")
            if dividend_cache.exists():
                ok(f"JPM dividend cache already exists and will be reused: {dividend_cache.name}")
            return True
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

    try:
        history_with_actions = yf.Ticker("JPM").history(period="max", auto_adjust=False, actions=True)
    except Exception as error:  # noqa: BLE001 - network failures are expected here
        print(f"[WARN] Yahoo Finance dividend history unavailable for JPM: {error}")
        return True

    if history_with_actions is None or history_with_actions.empty or "Dividends" not in history_with_actions.columns:
        print("[WARN] Yahoo Finance returned no dividend history for JPM")
        return True

    dividend_frame = history_with_actions[["Dividends"]].rename(columns={"Dividends": "dividend"})
    dividend_frame.index.name = "date"
    dividend_frame = dividend_frame.loc[START_DATE:END_DATE_EXCLUSIVE]

    if dividend_frame.empty:
        print("[WARN] Yahoo Finance dividend history did not cover 2018-2024 for JPM")
        return True

    dividend_path = save_csv(dividend_frame, "jpm_dividends_2018_2024.csv")
    ok(f"Yahoo Finance returned {len(dividend_frame)} dividend rows for JPM and saved {dividend_path.name}")
    return True


def test_fred(api_key: str, series_id: str, filename: str) -> None:
    existing_path = DATA_DIR / filename
    if existing_path.exists():
        ok(f"FRED {series_id} file already exists and will be reused: {existing_path.name}")
        return

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": "2024-12-31",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as error:  # noqa: BLE001 - network failures are expected here
        print(f"[WARN] FRED {series_id} download failed (network unavailable): {error}")
        fail(f"FRED {series_id} download failed and no local cache exists at {existing_path}")
        return

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
    news_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

    if not fred_key:
        fail("Missing FRED_API_KEY")

    if not news_key:
        fail("Missing ALPHA_VANTAGE_API_KEY")

    yahoo_ok = test_yahoo_finance()
    news_ok = fetch_alpha_vantage_news(news_key)
    test_fred(fred_key, "DGS10", "fred_DGS10_2018_2024.csv")
    test_fred(fred_key, "VIXCLS", "fred_VIXCLS_2018_2024.csv")

    if yahoo_ok is False:
        print("[WARN] Yahoo Finance download did not complete, but FRED did.")
    if news_ok is False:
        print("[WARN] Alpha Vantage news download did not complete, but the other sources did.")
    else:
        ok("All API downloads completed successfully")


if __name__ == "__main__":
    main()