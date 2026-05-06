from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from datetime import timedelta

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


def sanitize_filename(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def recent_date_window(days: int = 30) -> tuple[str, str]:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    return start_date.isoformat(), end_date.isoformat()


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


def test_news_api(api_key: str) -> bool:
    url = "https://newsapi.org/v2/everything"
    query = "JPM OR JPMorgan Chase"
    page_size = 100
    max_pages = 10
    all_articles: list[dict[str, object]] = []
    from_date, to_date = recent_date_window(30)

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "language": "en",
            "sortBy": "popularity",
            "from": from_date,
            "to": to_date,
            "pageSize": str(page_size),
            "page": str(page),
            "apiKey": api_key,
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 426:
            print(
                "[WARN] News API returned 426 Upgrade Required. "
                "The current plan does not support this historical range; "
                "use a paid NewsAPI archive plan or another historical news source."
            )
            return False

        if not response.ok:
            snippet = response.text.strip().replace("\n", " ")[:300]
            print(f"[WARN] News API HTTP {response.status_code}: {snippet}")
            return False

        payload = response.json()

        if payload.get("status") != "ok":
            message = payload.get("message", "Unknown News API error")
            print(f"[WARN] News API response: {message}")
            return False

        articles = payload.get("articles", [])
        if not articles:
            break

        all_articles.extend(articles)

        total_results = int(payload.get("totalResults", 0))
        if page * page_size >= total_results:
            break

    if not all_articles:
        print("[WARN] News API returned no JPMorgan Chase articles in the 2018-2024 range")
        return False

    frame = pd.DataFrame(all_articles)
    frame["source_name"] = frame["source"].apply(
        lambda source: source.get("name") if isinstance(source, dict) else None
    )
    frame["publishedAt"] = pd.to_datetime(frame["publishedAt"], errors="coerce")
    frame = frame.dropna(subset=["publishedAt"])
    frame = frame[
        (frame["publishedAt"] >= pd.Timestamp(from_date))
        & (frame["publishedAt"] < pd.Timestamp(to_date) + pd.Timedelta(days=1))
    ]
    frame = frame.sort_values("publishedAt")

    if frame.empty:
        print("[WARN] News API returned articles, but none fell inside the recent sentiment window")
        return False

    frame = frame[
        [
            "publishedAt",
            "source_name",
            "author",
            "title",
            "description",
            "url",
            "content",
        ]
    ].copy()
    frame["query"] = query
    frame["sentiment_window"] = f"{from_date}_to_{to_date}"

    filename = f"news_{sanitize_filename(query).lower()}_recent.csv"
    path = save_csv(frame, filename)
    ok(f"News API returned {len(frame)} recent articles for JPM and saved {path.name}")
    return True


def test_gdelt_news() -> bool:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    query = "JPMorgan Chase"
    page_size = 250
    max_pages = 10
    all_articles: list[dict[str, object]] = []

    start_datetime = "20180101000000"
    end_datetime = "20241231235959"

    for page in range(1, max_pages + 1):
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": str(page_size),
            "startrecord": str((page - 1) * page_size + 1),
            "startdatetime": start_datetime,
            "enddatetime": end_datetime,
        }

        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429:
            print("[WARN] GDELT rate limited the request; waiting before retrying.")
            time.sleep(5.5)
            response = requests.get(url, params=params, timeout=30)

        if not response.ok:
            snippet = response.text.strip().replace("\n", " ")[:300]
            print(f"[WARN] GDELT HTTP {response.status_code}: {snippet}")
            return False

        payload = response.json()
        articles = payload.get("articles", [])
        if not articles:
            break

        all_articles.extend(articles)
        if len(articles) < page_size:
            break

        time.sleep(5.5)

    if not all_articles:
        print("[WARN] GDELT returned no JPMorgan Chase articles in the 2018-2024 range")
        return False

    frame = pd.DataFrame(all_articles)
    if "seendate" in frame.columns:
        frame["publishedAt"] = pd.to_datetime(frame["seendate"], errors="coerce")
    else:
        frame["publishedAt"] = pd.NaT

    frame = frame.dropna(subset=["publishedAt"])
    frame = frame[(frame["publishedAt"] >= pd.Timestamp(START_DATE)) & (frame["publishedAt"] < pd.Timestamp(END_DATE_EXCLUSIVE))]
    frame = frame.sort_values("publishedAt")

    if frame.empty:
        print("[WARN] GDELT returned articles, but none fell inside the 2018-2024 range")
        return False

    source_column = "sourceCountry" if "sourceCountry" in frame.columns else None
    frame["source_name"] = frame[source_column] if source_column else None
    frame = frame[[
        "publishedAt",
        "source_name",
        "title",
        "url",
        "domain",
        "language",
        "seendate",
    ]].copy()
    frame["query"] = query

    filename = f"gdelt_{sanitize_filename(query).lower()}_2018_2024.csv"
    path = save_csv(frame, filename)
    ok(f"GDELT returned {len(frame)} articles for JPMorgan Chase and saved {path.name}")
    return True


def main() -> None:
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    news_key = os.getenv("NEWS_API_KEY", "").strip()

    if not fred_key:
        fail("Missing FRED_API_KEY")

    if not news_key:
        fail("Missing NEWS_API_KEY")

    yahoo_ok = test_yahoo_finance()
    news_ok = test_news_api(news_key)
    if not news_ok:
        print("[WARN] Falling back to GDELT for historical news data.")
        news_ok = test_gdelt_news()
    test_fred(fred_key, "DGS10", "fred_DGS10_2018_2024.csv")
    test_fred(fred_key, "VIXCLS", "fred_VIXCLS_2018_2024.csv")

    if yahoo_ok is False:
        print("[WARN] Yahoo Finance download did not complete, but FRED did.")
    if news_ok is False:
        print("[WARN] News API download did not complete, but the other sources did.")
    else:
        ok("All API downloads completed successfully")


if __name__ == "__main__":
    main()