from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def ensure_output_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, filename: str) -> Path:
    ensure_output_dir()
    path = PROCESSED_DIR / filename
    frame.to_csv(path, index=True)
    return path


def cap_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series.clip(lower, upper)


def load_market_data() -> pd.DataFrame:
    # Load raw JPM, Treasury, and VIX data and normalize the column names.
    jpm = pd.read_csv(RAW_DIR / "yahoo_jpm_2018_2024.csv", parse_dates=["Date"])
    dgs10 = pd.read_csv(RAW_DIR / "fred_DGS10_2018_2024.csv", parse_dates=["date"])
    vix = pd.read_csv(RAW_DIR / "fred_VIXCLS_2018_2024.csv", parse_dates=["date"])

    jpm = jpm.rename(
        columns={
            "Date": "date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj close": "Adj Close",
            "Adj Close": "Adj Close",
            "volume": "Volume",
            "Volume": "Volume",
        }
    ).set_index("date").sort_index()
    dgs10 = dgs10.rename(columns={"value": "dgs10"}).set_index("date").sort_index()
    vix = vix.rename(columns={"value": "vix"}).set_index("date").sort_index()

    jpm = jpm[[column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in jpm.columns]]
    dgs10 = dgs10[["dgs10"]]
    vix = vix[["vix"]]

    frame = jpm.join(dgs10, how="outer").join(vix, how="outer")
    frame = frame.sort_index()
    frame = frame.loc["2018-01-01":"2024-12-31"]

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[["dgs10", "vix"]] = frame[["dgs10", "vix"]].ffill()

    price_columns = [column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in frame.columns]
    if price_columns:
        frame[price_columns] = frame[price_columns].interpolate(limit_direction="both")

    return frame


def score_text(text: str) -> float:
    positive_words = {
        "beat",
        "growth",
        "gain",
        "upgrade",
        "strong",
        "positive",
        "record",
        "improve",
        "surge",
        "rise",
    }
    negative_words = {
        "miss",
        "loss",
        "downgrade",
        "weak",
        "negative",
        "drop",
        "fall",
        "decline",
        "risk",
        "concern",
    }

    tokens = {token.strip(".,!?;:\"'()[]{}<>|/\\").lower() for token in str(text).split()}
    positive_hits = sum(1 for token in tokens if token in positive_words)
    negative_hits = sum(1 for token in tokens if token in negative_words)
    raw_score = positive_hits - negative_hits

    if raw_score >= 0:
        return min(1.0, 0.5 + raw_score * 0.1)
    return max(0.0, 0.5 + raw_score * 0.1)


def load_news_data() -> pd.DataFrame | None:
    # Load the available news source and convert article text into a simple sentiment proxy.
    candidate_files = sorted(list(RAW_DIR.glob("news_*.csv")) + list(RAW_DIR.glob("gdelt_*.csv")))
    if not candidate_files:
        return None

    news_frame = pd.read_csv(candidate_files[0])
    if "publishedAt" not in news_frame.columns:
        return None

    news_frame["publishedAt"] = pd.to_datetime(news_frame["publishedAt"], errors="coerce")
    news_frame = news_frame.dropna(subset=["publishedAt"])
    if news_frame.empty:
        return None

    for column in ["title", "description", "content"]:
        if column not in news_frame.columns:
            news_frame[column] = ""

    news_frame["text_for_sentiment"] = (
        news_frame["title"].fillna("").astype(str)
        + " "
        + news_frame["description"].fillna("").astype(str)
        + " "
        + news_frame["content"].fillna("").astype(str)
    )
    news_frame["sentiment_score"] = news_frame["text_for_sentiment"].apply(score_text)
    news_frame["news_date"] = news_frame["publishedAt"].dt.floor("D")

    daily_news = (
        news_frame.groupby("news_date")
        .agg(
            news_article_count=("sentiment_score", "size"),
            news_sentiment_mean=("sentiment_score", "mean"),
            news_sentiment_std=("sentiment_score", "std"),
        )
        .sort_index()
    )

    daily_news["news_sentiment_std"] = daily_news["news_sentiment_std"].fillna(0.0)
    return daily_news


def build_features() -> pd.DataFrame:
    # Build the daily modeling table used in Week 2.
    frame = load_market_data()

    if "Adj Close" not in frame.columns:
        raise ValueError("JPM raw data is missing an Adj Close column")
    if "Close" not in frame.columns:
        raise ValueError("JPM raw data is missing a Close column")
    if "Open" not in frame.columns:
        raise ValueError("JPM raw data is missing an Open column")
    if "High" not in frame.columns:
        raise ValueError("JPM raw data is missing a High column")
    if "Low" not in frame.columns:
        raise ValueError("JPM raw data is missing a Low column")
    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0

    frame["jpm_return_1d"] = frame["Adj Close"].pct_change()
    frame["jpm_return_5d"] = frame["Adj Close"].pct_change(5)
    frame["jpm_vol_20d"] = frame["jpm_return_1d"].rolling(20).std() * (252 ** 0.5)
    frame["jpm_ma_20d"] = frame["Adj Close"].rolling(20).mean()
    frame["jpm_price_to_ma_20d"] = frame["Adj Close"] / frame["jpm_ma_20d"]
    frame["dgs10_change_1d"] = frame["dgs10"].diff()
    frame["dgs10_momentum_5d"] = frame["dgs10"].diff(5)
    frame["vix_change_1d"] = frame["vix"].diff()
    frame["vix_jpm_corr_20d"] = frame["jpm_return_1d"].rolling(20).corr(frame["vix_change_1d"])
    frame["rolling_high_20d"] = frame["Adj Close"].rolling(20).max()
    frame["drawdown_20d"] = frame["Adj Close"] / frame["rolling_high_20d"] - 1.0

    # Week 2 cleanup: cap obvious outliers with IQR clipping.
    for column in ["jpm_return_1d", "jpm_return_5d", "dgs10_change_1d", "dgs10_momentum_5d", "vix_change_1d"]:
        frame[column] = cap_iqr(frame[column])

    news_frame = load_news_data()
    if news_frame is not None:
        frame = frame.join(news_frame, how="left")
        frame["news_article_count"] = frame["news_article_count"].fillna(0)
        frame["news_sentiment_mean"] = frame["news_sentiment_mean"].fillna(0.5)
        frame["news_sentiment_std"] = frame["news_sentiment_std"].fillna(0.0)
    else:
        frame["news_article_count"] = 0
        frame["news_sentiment_mean"] = 0.5
        frame["news_sentiment_std"] = 0.0

    frame["news_7d_article_count"] = frame["news_article_count"].rolling(7).sum().fillna(0)
    frame["news_7d_sentiment_mean"] = frame["news_sentiment_mean"].rolling(7).mean().fillna(0.5)
    frame["news_7d_sentiment_trend"] = frame["news_7d_sentiment_mean"].diff().fillna(0.0)

    frame = frame.dropna(subset=["Adj Close", "Close", "Open", "High", "Low"])
    frame = frame.reset_index().rename(columns={"index": "date"})

    return frame


def main() -> None:
    # Run the full Week 2 preprocessing pipeline and save the structured dataset.
    features = build_features()
    output_path = save_csv(features, "week2_feature_dataset_2018_2024.csv")
    ok_columns = [
        "date",
        "Adj Close",
        "jpm_return_1d",
        "jpm_return_5d",
        "jpm_vol_20d",
        "dgs10_change_1d",
        "dgs10_momentum_5d",
        "vix_change_1d",
        "vix_jpm_corr_20d",
        "news_7d_article_count",
        "news_7d_sentiment_mean",
        "news_7d_sentiment_trend",
    ]

    print(f"[OK] Week 2 feature dataset saved to {output_path.name}")
    print(f"[OK] Columns included: {', '.join(ok_columns)}")


if __name__ == "__main__":
    main()
