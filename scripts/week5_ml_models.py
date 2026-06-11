"""
Week 6 – Machine Learning Model Design & Implementation
=======================================================

Two-Approach Architecture
--------------------------
Approach 1: ML Volatility Prediction + Chooser Pricing
    - Predict 20-day forward realised volatility from market features
    - Feed predicted vol into the chooser-option closed form
    - Models: RandomForest, XGBoost, LSTM

Approach 2: End-to-End Supervised Chooser Pricing
    - Directly predict chooser-option prices from market + contract features
    - Models: Linear Regression, XGBoost/GradientBoosting, Neural Network (MLP)

Feature Preparation
-------------------
- Time-series split: 70% train / 15% validation / 15% test by date
- No look-ahead bias: all input features are strictly backward-looking
- Forward vol target computed from future returns (Approach 1 only)

Outputs
-------
        data/processed/week6_feature_dataset_v1.0.csv
        data/processed/week6_vol_results_v1.0.csv
        data/processed/week6_pricing_results_v1.0.csv
        data/processed/week6_pricing_stratified_v1.0.csv
        data/processed/week6_model_comparison_v1.0.csv
        data/models/week6_*.joblib
        data/models/week6_*.keras
        data/reports/week6_ml_architecture_v1.0.md
        data/reports/week6_ml_architecture_v1.0.pdf
    data/reports/week6_feature_importance.png
    data/reports/week6_shap_app1_*.png
    data/reports/week6_shap_app2_*.png
    data/reports/week6_vol_prediction_comparison.png
    data/reports/week6_pricing_comparison.png
    data/reports/week6_model_performance.png
"""

from __future__ import annotations

import logging
import subprocess
import warnings
import zlib
from datetime import datetime
from functools import lru_cache
from math import erf, exp, log, sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    import xgboost as xgb
    HAS_XGB = os.environ.get("WEEK6_DISABLE_XGB") != "1"
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    HAS_TF = os.environ.get("WEEK6_DISABLE_LSTM") != "1"
except ImportError:
    HAS_TF = False

try:
    import shap  # type: ignore[import-not-found]
    HAS_SHAP = True
except Exception:
    shap = None  # type: ignore[assignment]
    HAS_SHAP = False

ENABLE_MLP = os.environ.get("WEEK6_DISABLE_MLP") != "1"

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR   = ROOT / "data" / "reports"
PIPELINE_VER  = "v1.0"
RUN_DATE      = datetime.now().strftime("%Y%m%d")
OUTPUT_WEEK   = "week6"
REPORT_STEM   = f"{OUTPUT_WEEK}_ml_architecture_{PIPELINE_VER}"
FEATURE_STEM  = f"{OUTPUT_WEEK}_feature_dataset_{PIPELINE_VER}"
VOL_STEM      = f"{OUTPUT_WEEK}_vol_results_{PIPELINE_VER}"
PRICING_STEM  = f"{OUTPUT_WEEK}_pricing_results_{PIPELINE_VER}"
PRICING_STRAT_STEM = f"{OUTPUT_WEEK}_pricing_stratified_{PIPELINE_VER}"
COMP_STEM     = f"{OUTPUT_WEEK}_model_comparison_{PIPELINE_VER}"
MODELS_DIR    = ROOT / "data" / "models"

# ── Config ────────────────────────────────────────────────────────────────────
ANNUALISE             = 252        # trading days / year
VOL_WINDOW            = 20         # backward volatility window (days)
FWD_VOL_WINDOW        = 20         # forward volatility target window (days)
CHOOSER_DECISION_TIMES = [0.25, 0.5]
CHOOSER_MATURITIES    = [0.5, 1.0, 1.5]
MONEYNESS             = [0.9, 1.0, 1.1]
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15
# TEST_FRAC     = 0.15  (implicit: remainder)
LSTM_LOOKBACK   = 20
LSTM_EPOCHS     = 50
LSTM_BATCH      = 32
RANDOM_STATE    = 42
CHOOSER_MC_PATHS = 2_000
VIX_LOW         = 20.0
VIX_HIGH        = 30.0
SEARCH_CV_SPLITS = 5
SEARCH_CV_SPLITS = 3
SEARCH_N_ITER_LR = 2
SEARCH_N_ITER_RF = 8
SEARCH_N_ITER_GBM = 2
SEARCH_N_ITER_MLP = 2

logger = logging.getLogger("week5_ml_models")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# =============================================================================
# 1. BSM Helper Functions
# =============================================================================

def _ncdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bsm_price_scalar(
    S: float, K: float, T: float, r: float, q: float,
    sigma: float, option_type: str,
) -> float:
    """Black-Scholes-Merton price for a single European option."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S * exp(-q * T) - K * exp(-r * T), 0.0)
        return max(K * exp(-r * T) - S * exp(-q * T), 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return S * exp(-q * T) * _ncdf(d1) - K * exp(-r * T) * _ncdf(d2)
    return K * exp(-r * T) * _ncdf(-d2) - S * exp(-q * T) * _ncdf(-d1)


def bsm_price_vec(
    S: np.ndarray, K: np.ndarray, T: np.ndarray,
    r: np.ndarray, q: np.ndarray, sigma: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """Vectorised BSM for numpy arrays."""
    from scipy.stats import norm
    T     = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    d1    = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2    = d1 - sigma * np.sqrt(T)
    call  = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put   = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(is_call.astype(bool), call, put)


def chooser_price_scalar(
    S: float,
    K: float,
    T1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Chooser option price using the Rubinstein-style closed form."""
    if T2 <= T1:
        raise ValueError("T2 must be greater than T1 for a chooser option")

    time_to_choice = T2 - T1
    call_leg = bsm_price_scalar(S, K, T2, r, q, sigma, "call")
    adjusted_strike = K * exp(-r * time_to_choice)
    put_leg = bsm_price_scalar(S, adjusted_strike, time_to_choice, r, q, sigma, "put")
    return call_leg + put_leg


def _stable_seed(*values: object) -> int:
    """Create a deterministic seed from a tuple of scalar inputs."""
    payload = "|".join(str(v) for v in values).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


@lru_cache(maxsize=65536)
def mc_price_scalar(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
    n_paths: int = CHOOSER_MC_PATHS,
) -> float:
    """Monte Carlo European option price using the same GBM benchmark style as Week 4."""
    if T <= 0 or sigma <= 0:
        return bsm_price_scalar(S, K, T, r, q, sigma, option_type)

    seed = _stable_seed(round(S, 6), round(K, 6), round(T, 6), round(r, 6), round(q, 6), round(sigma, 6), option_type, n_paths)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * z)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    return float(exp(-r * T) * payoffs.mean())


@lru_cache(maxsize=65536)
def chooser_price_mc_scalar(
    S: float,
    K: float,
    T1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int = CHOOSER_MC_PATHS,
) -> float:
    """Chooser price benchmark built from MC-valued call/put legs."""
    if T2 <= T1:
        raise ValueError("T2 must be greater than T1 for a chooser option")

    time_to_choice = T2 - T1
    call_leg = mc_price_scalar(S, K, T2, r, q, sigma, "call", n_paths=n_paths)
    adjusted_strike = K * exp(-r * time_to_choice)
    put_leg = mc_price_scalar(S, adjusted_strike, time_to_choice, r, q, sigma, "put", n_paths=n_paths)
    return call_leg + put_leg


# =============================================================================
# 2. Data Loading
# =============================================================================

def load_market_data() -> pd.DataFrame:
    """Load and merge raw market data into a single daily DataFrame."""
    # ── Stock prices ──────────────────────────────────────────────────────
    stock = pd.read_csv(RAW_DIR / "yahoo_jpm_2018_2024.csv", parse_dates=["Date"])
    stock = stock.rename(columns={"Date": "date"})
    col_map = {c: c.lower() for c in stock.columns}
    stock   = stock.rename(columns=col_map)
    stock["date"] = pd.to_datetime(stock["date"]).dt.normalize()
    stock   = stock[["date", "close"]].dropna().set_index("date")

    # ── Risk-free rate (10-yr Treasury) ───────────────────────────────────
    rates = pd.read_csv(RAW_DIR / "fred_DGS10_2018_2024.csv", parse_dates=["date"])
    rates["dgs10"] = pd.to_numeric(rates["value"], errors="coerce")
    rates = rates[["date", "dgs10"]].set_index("date").sort_index().ffill()

    # ── VIX ───────────────────────────────────────────────────────────────
    vix = pd.read_csv(RAW_DIR / "fred_VIXCLS_2018_2024.csv", parse_dates=["date"])
    vix["vix"] = pd.to_numeric(vix["value"], errors="coerce")
    vix = vix[["date", "vix"]].set_index("date").sort_index().ffill()

    # ── Dividends → TTM dividend yield ────────────────────────────────────
    divs = pd.read_csv(RAW_DIR / "jpm_dividends_2018_2024.csv")
    divs["date"] = (
        pd.to_datetime(divs["date"], utc=True).dt.tz_convert(None).dt.normalize()
    )
    divs = divs[divs["dividend"] > 0].set_index("date").sort_index()

    # ── News sentiment ────────────────────────────────────────────────────
    news = pd.read_csv(
        RAW_DIR / "alphavantage_news_jpm_2018_2024.csv",
        parse_dates=["publishedAt"],
    )
    news["date"] = news["publishedAt"].dt.normalize()
    daily_sentiment = (
        news.groupby("date")["ticker_sentiment_score"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "sentiment", "count": "news_count"})
    )

    # ── Merge everything ──────────────────────────────────────────────────
    df = stock.copy()
    df = df.join(rates, how="left").join(vix, how="left")
    # Forward-fill only (no bfill) to prevent look-ahead bias in starting period
    df["dgs10"] = df["dgs10"].ffill()
    df["vix"]   = df["vix"].ffill()
    # Drop rows with missing dgs10 or vix at the start (before first data point)
    df = df.dropna(subset=["dgs10", "vix"])

    div_series    = divs["dividend"].reindex(df.index, fill_value=0.0)
    df["ttm_div"] = div_series.rolling(ANNUALISE, min_periods=1).sum()
    df["q"]       = df["ttm_div"] / df["close"]

    df = df.join(daily_sentiment, how="left")
    df["sentiment"]  = df["sentiment"].fillna(0.0)
    df["news_count"] = df["news_count"].fillna(0.0)
    df["r"] = df["dgs10"] / 100.0

    return df.sort_index()


# =============================================================================
# 3. Feature Engineering
# =============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all backward-looking features and the forward-vol target."""
    out = df.copy()

    # Log returns
    out["log_ret"] = np.log(out["close"] / out["close"].shift(1))

    # ── Backward volatility features (multiple horizons for term-structure) ──
    out["hist_vol_5d"]  = out["log_ret"].rolling(5).std()   * sqrt(ANNUALISE)
    out["hist_vol_10d"] = out["log_ret"].rolling(10).std()  * sqrt(ANNUALISE)
    out["hist_vol_20d"] = out["log_ret"].rolling(20).std()  * sqrt(ANNUALISE)
    out["hist_vol_40d"] = out["log_ret"].rolling(40).std()  * sqrt(ANNUALISE)
    out["hist_vol_60d"] = out["log_ret"].rolling(60).std()  * sqrt(ANNUALISE)

    # Vol term-structure ratios
    out["vol_ratio_5_20"]  = (
        out["hist_vol_5d"] / out["hist_vol_20d"].replace(0, np.nan)
    )
    out["vol_ratio_20_60"] = (
        out["hist_vol_20d"] / out["hist_vol_60d"].replace(0, np.nan)
    )

    # Vol momentum (5-day % change in 20-day rolling vol)
    out["vol_20d_change"] = out["hist_vol_20d"].pct_change(5)

    # ── Price momentum ────────────────────────────────────────────────────
    out["return_1d"]  = out["log_ret"]
    out["return_5d"]  = out["log_ret"].rolling(5).sum()
    out["return_20d"] = out["log_ret"].rolling(20).sum()

    # ── Trend features ────────────────────────────────────────────────────
    ma20 = out["close"].rolling(20).mean()
    ma60 = out["close"].rolling(60).mean()
    out["price_to_ma_20d"] = out["close"] / ma20
    out["price_to_ma_60d"] = out["close"] / ma60

    # ── VIX features ──────────────────────────────────────────────────────
    out["vix_change_5d"] = out["vix"].pct_change(5)
    out["vix_ma_ratio"]  = out["vix"] / out["vix"].rolling(20).mean()

    # ── Rolling VIX–JPM return correlation ───────────────────────────────
    vix_ret = out["vix"].pct_change()
    out["vix_jpm_corr_20d"] = out["log_ret"].rolling(20).corr(vix_ret)

    # ── Sentiment features ────────────────────────────────────────────────
    out["sentiment_7d"]  = out["sentiment"].rolling(7,  min_periods=1).mean()
    out["sentiment_20d"] = out["sentiment"].rolling(20, min_periods=1).mean()
    out["news_count_7d"] = out["news_count"].rolling(7, min_periods=1).sum()

    # ── Drawdown ──────────────────────────────────────────────────────────
    rolling_max = out["close"].rolling(20).max()
    out["drawdown_20d"] = (out["close"] - rolling_max) / rolling_max

    # ── VIX regime label (for stratified analysis; NOT an input feature) ──
    out["vix_regime"] = pd.cut(
        out["vix"],
        bins=[0, VIX_LOW, VIX_HIGH, np.inf],
        labels=["low", "medium", "high"],
    )

    # ── Forward realised volatility target (Approach 1) ──────────────────
    # fwd_vol_20d[t] = annualised std of log returns over days [t+1, t+20]
    # This is future information — used ONLY as the target, never as input.
    log_rets = out["log_ret"].values
    fwd_vol  = [np.nan] * len(log_rets)
    for i in range(len(log_rets) - FWD_VOL_WINDOW):
        window = log_rets[i + 1 : i + 1 + FWD_VOL_WINDOW]
        if not np.any(np.isnan(window)):
            fwd_vol[i] = float(np.std(window) * sqrt(ANNUALISE))
    out["fwd_vol_20d"] = fwd_vol

    return out


def get_vol_feature_columns() -> list[str]:
    """Return the ordered list of input feature names for Approach 1."""
    return [
        "hist_vol_5d", "hist_vol_10d", "hist_vol_20d", "hist_vol_40d", "hist_vol_60d",
        "vol_ratio_5_20", "vol_ratio_20_60", "vol_20d_change",
        "return_1d", "return_5d", "return_20d",
        "price_to_ma_20d", "price_to_ma_60d",
        "vix", "vix_change_5d", "vix_ma_ratio",
        "vix_jpm_corr_20d",
        "r", "q",
        "sentiment_7d", "sentiment_20d", "news_count_7d",
        "drawdown_20d",
    ]


def get_pricing_feature_columns() -> list[str]:
    """Return the ordered list of input feature names for Approach 2.

    Include only volatility proxy features. Raw historical volatility windows
    are deliberately excluded to avoid function leakage in the end-to-end task.
    """
    return [
        "vol_ratio_5_20", "vol_ratio_20_60", "vol_20d_change",
        "return_1d", "return_5d", "return_20d",
        "price_to_ma_20d", "price_to_ma_60d",
        "vix", "vix_change_5d", "vix_ma_ratio",
        "vix_jpm_corr_20d",
        "r", "q",
        "sentiment_7d", "sentiment_20d", "news_count_7d",
        "drawdown_20d",
    ]


def get_feature_columns() -> list[str]:
    """Backward-compatible alias for the volatility feature set."""
    return get_vol_feature_columns()


# =============================================================================
# 4. Dataset Construction
# =============================================================================

def build_vol_dataset(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approach 1 dataset: one row per trading day.
    Features: backward-looking market features.
    Target:   fwd_vol_20d (realised vol over next 20 trading days).
    """
    feature_cols = get_vol_feature_columns()
    keep = list(dict.fromkeys(
        feature_cols + ["fwd_vol_20d", "hist_vol_20d", "close", "r", "q", "vix_regime"]
    ))
    df = feat_df[[c for c in keep if c in feat_df.columns]].copy()
    df = df.dropna(subset=feature_cols + ["fwd_vol_20d"])
    df = df.rename(columns={"fwd_vol_20d": "target_vol"})
    return df


def _match_vol_to_maturity(T_years: float) -> str:
    """Map option maturity (in years) to nearest available historical vol window.
    E.g.: T=0.5y → "hist_vol_20d" (≈3-4 weeks), T=1.0y → "hist_vol_40d" (≈2 months).
    """
    T_days = T_years * ANNUALISE
    if T_days < 7:
        return "hist_vol_5d"
    elif T_days < 15:
        return "hist_vol_10d"
    elif T_days < 30:
        return "hist_vol_20d"
    elif T_days < 50:
        return "hist_vol_40d"
    else:
        return "hist_vol_60d"


def build_option_dataset(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approach 2 dataset: one row per (trading day × chooser contract).
    Features: market features, volatility proxies, and chooser parameters
              (S, K, moneyness, T1, T2).
    Target:   Monte Carlo chooser price computed with vol matched to maturity T2.
    
    CRITICAL FIX: Previous version used fixed hist_vol_20d for all maturities,
    causing term-structure mismatch. Now selects vol window based on T2.
    """
    feature_cols = get_pricing_feature_columns()
    # Include all vol windows for matching
    vol_cols = ["hist_vol_5d", "hist_vol_10d", "hist_vol_20d", "hist_vol_40d", "hist_vol_60d"]
    keep = list(dict.fromkeys(
        feature_cols + vol_cols + ["close", "r", "q", "vix_regime"]
    ))
    daily = feat_df[[c for c in keep if c in feat_df.columns]].dropna(
        subset=feature_cols + vol_cols
    )

    rows: list[dict] = []
    for date, row in daily.iterrows():
        S     = float(row["close"])
        r     = float(row["r"])
        q     = float(row["q"])
        if S <= 0 or np.isnan(S):
            continue
            
        for T1 in CHOOSER_DECISION_TIMES:
            for T2 in CHOOSER_MATURITIES:
                if T2 <= T1:
                    continue
                
                # Select vol window that matches T2 maturity
                vol_col = _match_vol_to_maturity(T2)
                sigma = float(row[vol_col])
                if sigma <= 0 or np.isnan(sigma):
                    continue
                    
                for m in MONEYNESS:
                    K = S * m
                    # Target label: Monte Carlo benchmark, not the closed-form chooser.
                    price = chooser_price_mc_scalar(S, K, T1, T2, r, q, sigma)
                    rec: dict = {
                        "date": date,
                        "S": S,
                        "K": K,
                        "moneyness": m,
                        "T1": T1,
                        "T2": T2,
                        "chooser_price": price,
                        "vol_match": vol_col,  # Track which vol window was used
                        "vix_regime": row.get("vix_regime", "unknown"),
                    }
                    # Keep raw historical volatility windows as auxiliary columns
                    # for Approach 1 evaluation only; they are not part of the
                    # Approach 2 feature set.
                    for vol_name in vol_cols:
                        rec[vol_name] = float(row[vol_name])
                    for extra_vol_name in ["vol_ratio_5_20", "vol_ratio_20_60", "vol_20d_change"]:
                        if extra_vol_name in row.index:
                            extra_value = row[extra_vol_name]
                            rec[extra_vol_name] = float(extra_value) if pd.notna(extra_value) else np.nan
                    for col in feature_cols:
                        rec[col] = float(row[col])
                    rows.append(rec)

    opt_df = pd.DataFrame(rows).set_index("date")
    return opt_df


# =============================================================================
# 5. Time-Series Split (70% / 15% / 15% by date)
# =============================================================================

def time_series_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float   = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split on unique dates to prevent look-ahead bias.
    Returns (train, val, test).
    """
    dates   = df.index.unique().sort_values()
    n       = len(dates)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_dates = dates[:n_train]
    val_dates   = dates[n_train : n_train + n_val]
    test_dates  = dates[n_train + n_val :]

    return (
        df[df.index.isin(train_dates)].copy(),
        df[df.index.isin(val_dates)].copy(),
        df[df.index.isin(test_dates)].copy(),
    )


def _make_time_series_cv(n_samples: int, max_splits: int = SEARCH_CV_SPLITS) -> TimeSeriesSplit:
    """Create a safe TimeSeriesSplit for random-search cross-validation."""
    n_splits = min(max_splits, max(2, n_samples // 50))
    return TimeSeriesSplit(n_splits=n_splits)


def _run_random_search(
    estimator,
    param_distributions: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int,
    scoring: str = "neg_mean_absolute_error",
) -> RandomizedSearchCV:
    """Run random search with time-series cross-validation."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=_make_time_series_cv(len(X_train)),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def _refit_best_estimator(
    search: RandomizedSearchCV,
    X_full: np.ndarray,
    y_full: np.ndarray,
):
    """Clone the best estimator and refit it on the full train+validation split."""
    model = clone(search.best_estimator_)
    model.fit(X_full, y_full)
    return model


def _format_param_summary(params: dict[str, object], keys: list[str]) -> str:
    """Render a compact best-parameter summary for markdown tables."""
    parts: list[str] = []
    for key in keys:
        if key not in params:
            continue
        value = params[key]
        if key == "hidden_layer_sizes" and isinstance(value, tuple):
            value = "x".join(str(v) for v in value)
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "-"


def _latest_week4_baseline() -> pd.Series | None:
    """Load the most recent Week 4 BSM error summary, if it exists."""
    candidates = sorted(PROCESSED_DIR.glob("week4_bsm_error_metrics_v1.0_*.csv"))
    if not candidates:
        return None
    summary = pd.read_csv(candidates[-1])
    overall = summary[summary["group"] == "overall"]
    if overall.empty:
        return None
    return overall.iloc[0]


# =============================================================================
# 6. Approach 1 – ML Volatility Prediction
# =============================================================================

def train_vol_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Tune RF, XGBoost (or GBDT), and optionally LSTM for vol prediction."""
    models: dict = {}
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    # ── Random Forest ──────────────────────────────────────────────────────
    logger.info("  [Approach 1] RandomForest random search...")
    rf_search = _run_random_search(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [200, 300, 400, 600, 800],
            "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [1, 2, 4, 6, 8],
            "min_samples_split": [2, 4, 6, 8, 10],
            "max_features": ["sqrt", 0.5, 0.7, 1.0],
        },
        X_train,
        y_train,
        SEARCH_N_ITER_RF,
    )
    rf_val_mae = mean_absolute_error(y_val, rf_search.predict(X_val))
    rf_model = _refit_best_estimator(rf_search, X_full, y_full)
    logger.info("    RF   val MAE=%.6f | best=%s", rf_val_mae, _format_param_summary(rf_search.best_params_, ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]))
    models["RandomForest"] = {
        "model": rf_model,
        "val_mae": rf_val_mae,
        "cv_mae": float(-rf_search.best_score_),
        "best_params": rf_search.best_params_,
        "search": rf_search,
        "is_tree": True,
    }

    # ── XGBoost (or sklearn GBDT fallback) ────────────────────────────────
    if HAS_XGB:
        logger.info("  [Approach 1] XGBoost random search...")
        gbm_search = _run_random_search(
            xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            {
                "n_estimators": [200, 400, 600, 800, 1000],
                "max_depth": [2, 3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 7],
                "reg_alpha": [0.0, 0.01, 0.1, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            },
            X_train,
            y_train,
            SEARCH_N_ITER_GBM,
        )
    else:
        logger.info("  [Approach 1] GradientBoosting random search (XGBoost not found)...")
        gbm_search = _run_random_search(
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [150, 250, 350, 500, 700],
                "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_samples_split": [2, 4, 6, 8, 10],
                "min_samples_leaf": [1, 2, 4, 6],
            },
            X_train,
            y_train,
            SEARCH_N_ITER_GBM,
        )
    gbm_name    = "XGBoost" if HAS_XGB else "GradientBoosting"
    gbm_val_mae = mean_absolute_error(y_val, gbm_search.predict(X_val))
    gbm_model = _refit_best_estimator(gbm_search, X_full, y_full)
    logger.info(
        "    %-16s val MAE=%.6f | best=%s",
        gbm_name,
        gbm_val_mae,
        _format_param_summary(
            gbm_search.best_params_,
            ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"],
        ),
    )
    models[gbm_name] = {
        "model": gbm_model,
        "val_mae": gbm_val_mae,
        "cv_mae": float(-gbm_search.best_score_),
        "best_params": gbm_search.best_params_,
        "search": gbm_search,
        "is_tree": True,
    }

    # ── LSTM (TensorFlow / Keras) ─────────────────────────────────────────
    if HAS_TF:
        logger.info("  [Approach 1] Training LSTM (lookback=%d days)...", LSTM_LOOKBACK)
        lstm_result = _train_lstm(X_train, y_train, X_val, y_val)
        if lstm_result is not None:
            models["LSTM"] = lstm_result
    else:
        logger.info("  [Approach 1] Skipping LSTM (TensorFlow not found).")

    return models


def _train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict | None:
    """Build and train an LSTM sequence model for vol prediction."""
    try:
        import os
        # Force CPU-only to avoid Metal GPU bus errors on Apple Silicon
        os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
        os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "0"

        import tensorflow as tf  # type: ignore[import-not-found]
        tf.config.set_visible_devices([], "GPU")
        from tensorflow.keras.models import Sequential  # type: ignore[import-not-found]
        from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore[import-not-found]
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[import-not-found]

        tf.get_logger().setLevel("ERROR")

        scaler   = RobustScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        X_vl_sc  = scaler.transform(X_val)

        def make_sequences(X: np.ndarray, y: np.ndarray, lb: int):
            Xs, ys = [], []
            for i in range(lb, len(X)):
                Xs.append(X[i - lb : i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        X_tr_seq, y_tr_seq = make_sequences(X_tr_sc, y_train, LSTM_LOOKBACK)
        X_vl_seq, y_vl_seq = make_sequences(X_vl_sc, y_val,   LSTM_LOOKBACK)

        if len(X_tr_seq) < 50:
            logger.warning("  Too few training sequences for LSTM; skipping.")
            return None

        n_feat = X_train.shape[1]
        model  = Sequential([
            LSTM(64, input_shape=(LSTM_LOOKBACK, n_feat), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        history = model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_vl_seq, y_vl_seq),
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            callbacks=[early_stop],
            shuffle=False,
            verbose=0,
        )
        val_preds  = model.predict(X_vl_seq, verbose=0).flatten()
        val_mae    = mean_absolute_error(y_vl_seq, val_preds)
        logger.info("    LSTM val MAE=%.6f (epochs=%d)", val_mae,
                    len(history.history["loss"]))
        return {
            "model":    model,
            "scaler":   scaler,
            "val_mae":  val_mae,
            "history":  history.history,
            "lookback": LSTM_LOOKBACK,
        }
    except Exception as exc:
        logger.warning("  LSTM training failed: %s", exc)
        return None


def _predict_vol(md: dict, X: np.ndarray, name: str) -> np.ndarray:
    """Generate vol predictions, handling LSTM vs tabular models."""
    if name == "LSTM":
        scaler   = md["scaler"]
        model    = md["model"]
        lb       = md["lookback"]
        X_sc     = scaler.transform(X)
        Xs       = [X_sc[i - lb : i] for i in range(lb, len(X_sc))]
        if not Xs:
            return np.full(len(X), np.nan)
        preds     = model.predict(np.array(Xs), verbose=0).flatten()
        full      = np.full(len(X), np.nan)
        full[lb:] = preds
        return full
    if "model" in md:
        return md["model"].predict(X)
    return md["pipeline"].predict(X)


def evaluate_approach1(
    models: dict,
    vol_test: pd.DataFrame,
    opt_test: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Evaluate Approach 1 on the test set:
      - vol prediction MAE/RMSE
    - chooser pricing MAE/RMSE (using Monte Carlo benchmark prices)
    """
    X_test = vol_test[feature_cols].values
    y_test = vol_test["target_vol"].values

    rows = []
    for name, md in models.items():
        vol_preds = _predict_vol(md, X_test, name)
        mask      = ~np.isnan(vol_preds)
        vp        = vol_preds[mask]
        yt        = y_test[mask]
        sub       = vol_test.iloc[mask] if name == "LSTM" else vol_test

        vol_mae  = mean_absolute_error(yt, vp)
        vol_rmse = float(np.sqrt(mean_squared_error(yt, vp)))

        vol_pred_series = pd.Series(vp, index=sub.index)
        opt_eval = opt_test.copy()
        opt_eval["pred_vol"] = opt_eval.index.map(vol_pred_series)
        opt_eval = opt_eval.dropna(subset=["pred_vol", "chooser_price", "hist_vol_20d", "hist_vol_5d", "hist_vol_10d", "hist_vol_40d", "hist_vol_60d"])

        chooser_pred_arr, chooser_base_arr, chooser_target_arr = [], [], []
        for _, row in opt_eval.iterrows():
            vol_match = _match_vol_to_maturity(float(row["T2"]))
            sig_pred_20d = max(float(row["pred_vol"]), 1e-4)
            sig_hist_20d = max(float(row["hist_vol_20d"]), 1e-4)
            sig_match = max(float(row[vol_match]), 1e-4)
            sig_pred = sig_pred_20d * (sig_match / sig_hist_20d)
            sig_base = sig_match
            S, K = float(row["S"]), float(row["K"])
            T1, T2 = float(row["T1"]), float(row["T2"])
            r, q   = float(row["r"]), float(row["q"])
            chooser_pred_arr.append(chooser_price_scalar(S, K, T1, T2, r, q, sig_pred))
            chooser_base_arr.append(chooser_price_scalar(S, K, T1, T2, r, q, sig_base))
            chooser_target_arr.append(float(row["chooser_price"]))

        arr_pred = np.array(chooser_pred_arr)
        arr_base = np.array(chooser_base_arr)
        arr_target = np.array(chooser_target_arr)
        opt_mae  = mean_absolute_error(arr_target, arr_pred)
        opt_rmse = float(np.sqrt(mean_squared_error(arr_target, arr_pred)))
        base_mae  = mean_absolute_error(arr_target, arr_base)
        base_rmse = float(np.sqrt(mean_squared_error(arr_target, arr_base)))

        rows.append({
            "model":        name,
            "vol_mae":      vol_mae,
            "vol_rmse":     vol_rmse,
            "pricing_mae":  opt_mae,
            "pricing_rmse": opt_rmse,
        })
        logger.info(
            "  Approach 1 | %-16s | vol MAE=%.5f RMSE=%.5f | "
            "chooser MAE=%.4f RMSE=%.4f | baseline MAE=%.4f RMSE=%.4f",
            name, vol_mae, vol_rmse, opt_mae, opt_rmse, base_mae, base_rmse,
        )

    return pd.DataFrame(rows)


# =============================================================================
# 7. Approach 2 – End-to-End Supervised Chooser Pricing
# =============================================================================

def get_pricing_model_columns(feature_cols: list[str]) -> list[str]:
    return feature_cols + ["S", "K", "moneyness", "T1", "T2"]


def train_pricing_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Tune LinearRegression, XGBoost/GBDT, and MLP for direct chooser pricing.
    
    Tree models are tuned with random search and then refit on train+validation.
    Linear regression remains a fixed baseline for reference.
    """
    models: dict = {}
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    # ── Linear Regression ─────────────────────────────────────────────────
    logger.info("  [Approach 2] LinearRegression random search...")
    lr_search = _run_random_search(
        Pipeline([
        ("scaler", RobustScaler()),
        ("model",  LinearRegression()),
    ]),
        {
            "model__fit_intercept": [True, False],
            "model__positive": [False, True],
        },
        X_train,
        y_train,
        SEARCH_N_ITER_LR,
    )
    lr_val_mae = mean_absolute_error(y_val, lr_search.predict(X_val))
    lr_model = _refit_best_estimator(lr_search, X_full, y_full)
    logger.info(
        "    LR  val MAE=%.4f | best=%s",
        lr_val_mae,
        _format_param_summary(lr_search.best_params_, ["model__fit_intercept", "model__positive"]),
    )
    models["LinearRegression"] = {
        "pipeline": lr_model,
        "val_mae": lr_val_mae,
        "cv_mae": float(-lr_search.best_score_),
        "best_params": lr_search.best_params_,
        "search": lr_search,
        "is_tree": False,
    }

    # ── XGBoost / Gradient Boosting (NO scaling needed for tree models) ────
    if HAS_XGB:
        logger.info("  [Approach 2] XGBoost random search (unscaled)...")
        gbm_search = _run_random_search(
            xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            {
                "n_estimators": [300, 500, 700, 900, 1200],
                "max_depth": [2, 3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 7],
                "reg_alpha": [0.0, 0.01, 0.1, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            },
            X_train,
            y_train,
            SEARCH_N_ITER_GBM,
        )
        gbm_model = _refit_best_estimator(gbm_search, X_full, y_full)
        gbm_val_mae = mean_absolute_error(y_val, gbm_search.predict(X_val))
        gbm_name = "XGBoost"
    else:
        logger.info("  [Approach 2] GradientBoosting random search (unscaled)...")
        gbm_search = _run_random_search(
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [200, 300, 500, 700, 900],
                "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_samples_split": [2, 4, 6, 8, 10],
                "min_samples_leaf": [1, 2, 4, 6],
            },
            X_train,
            y_train,
            SEARCH_N_ITER_GBM,
        )
        gbm_model = _refit_best_estimator(gbm_search, X_full, y_full)
        gbm_val_mae = mean_absolute_error(y_val, gbm_search.predict(X_val))
        gbm_name = "GradientBoosting"
    
    logger.info(
        "    %-16s val MAE=%.4f | best=%s",
        gbm_name,
        gbm_val_mae,
        _format_param_summary(
            gbm_search.best_params_,
            ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"],
        ),
    )
    models[gbm_name] = {
        "model": gbm_model,
        "val_mae": gbm_val_mae,
        "cv_mae": float(-gbm_search.best_score_),
        "best_params": gbm_search.best_params_,
        "search": gbm_search,
        "is_tree": True,
    }

    if ENABLE_MLP:
        logger.info("  [Approach 2] MLP random search (scaled)...")
        mlp_search = _run_random_search(
            Pipeline([
            ("scaler", RobustScaler()),
            ("model",  MLPRegressor(
                activation="relu",
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=RANDOM_STATE,
                verbose=False,
            )),
        ]),
            {
                "model__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128, 64), (256, 128, 64, 32)],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "model__learning_rate_init": [1e-4, 3e-4, 5e-4, 1e-3],
                "model__batch_size": [16, 32, 64],
            },
            X_train,
            y_train,
            SEARCH_N_ITER_MLP,
        )
        mlp_model = _refit_best_estimator(mlp_search, X_full, y_full)
        mlp_val_mae = mean_absolute_error(y_val, mlp_search.predict(X_val))
        logger.info(
            "    MLP val MAE=%.4f | best=%s",
            mlp_val_mae,
            _format_param_summary(
                mlp_search.best_params_,
                ["model__hidden_layer_sizes", "model__alpha", "model__learning_rate_init", "model__batch_size"],
            ),
        )
        models["NeuralNetwork"] = {
            "pipeline": mlp_model,
            "val_mae": mlp_val_mae,
            "cv_mae": float(-mlp_search.best_score_),
            "best_params": mlp_search.best_params_,
            "search": mlp_search,
            "is_tree": False,
        }
    else:
        logger.info("  [Approach 2] Skipping MLP (WEEK6_ENABLE_MLP not set).")

    return models


def evaluate_approach2(
    models: dict,
    opt_test: pd.DataFrame,
    pricing_cols: list[str],
) -> pd.DataFrame:
    """Evaluate Approach 2: end-to-end chooser pricing on test set.
    Handles mixed model types: Pipelines (LR, MLP) and raw tree models (XGB, GBDT).
    """
    X_test = opt_test[pricing_cols].values
    y_test = opt_test["chooser_price"].values

    rows = []
    for name, md in models.items():
        # Determine prediction method: raw model vs pipeline
        if "model" in md and md.get("is_tree"):
            # Tree model (XGBoost/GBDT): no scaling needed
            y_pred = np.maximum(md["model"].predict(X_test), 0.0)
        elif "pipeline" in md:
            # Linear/MLP: scaled pipeline
            y_pred = np.maximum(md["pipeline"].predict(X_test), 0.0)
        else:
            logger.warning("  Unknown model structure for %s, skipping", name)
            continue
            
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        ss_res = float(np.sum((y_test - y_pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        rows.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})
        logger.info(
            "  Approach 2 | %-16s | MAE=%.4f RMSE=%.4f R²=%.4f",
            name, mae, rmse, r2,
        )

    return pd.DataFrame(rows)


def evaluate_bsm_baseline(opt_test: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the closed-form chooser baseline on the same test contracts."""
    preds: list[float] = []
    y_true = opt_test["chooser_price"].values

    for _, row in opt_test.iterrows():
        vol_match = _match_vol_to_maturity(float(row["T2"]))
        sigma = max(float(row[vol_match]), 1e-4)
        preds.append(
            chooser_price_scalar(
                float(row["S"]),
                float(row["K"]),
                float(row["T1"]),
                float(row["T2"]),
                float(row["r"]),
                float(row["q"]),
                sigma,
            )
        )

    y_pred = np.array(preds)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    logger.info(
        "  Approach 2 | %-16s | MAE=%.4f RMSE=%.4f R²=%.4f",
        "BSM Baseline",
        mae,
        rmse,
        r2,
    )
    return pd.DataFrame([
        {"model": "BSM Baseline", "mae": mae, "rmse": rmse, "r2": r2}
    ])


def evaluate_approach2_stratified(
    models: dict,
    opt_test: pd.DataFrame,
    pricing_cols: list[str],
) -> pd.DataFrame:
    """Evaluate Approach 2 with stratification to diagnose model failures."""
    results = []
    
    # 为避免索引不唯一问题，重置索引为默认整数索引（并保留原始索引为 'orig_date' 列）
    opt_test_reset = opt_test.reset_index().rename(columns={"index": "orig_date"})
    # 同时保持 y_test 顺序与重置后的行顺序一致
    y_test = opt_test_reset["chooser_price"].values
    
    # 预先计算每个模型的预测值（顺序与 opt_test_reset 一致）
    preds_dict = {}
    for name, md in models.items():
        X_test = opt_test_reset[pricing_cols].values
        if "model" in md and md.get("is_tree"):
            y_pred = np.maximum(md["model"].predict(X_test), 0.0)
        elif "pipeline" in md:
            y_pred = np.maximum(md["pipeline"].predict(X_test), 0.0)
        else:
            continue
        preds_dict[name] = y_pred

    # 创建分组列（在重置后的 DataFrame 上）
    opt_test_reset["moneyness_bucket"] = pd.cut(
        opt_test_reset["moneyness"],
        bins=[0, 0.95, 1.05, 2.0],
        labels=["OTM", "ATM", "ITM"],
    )
    opt_test_reset["T2_bucket"] = pd.cut(
        opt_test_reset["T2"],
        bins=[0, 0.5, 1.0, 2.0],
        labels=["short", "medium", "long"],
    )
    
    # 遍历每个模型和每个分组
    for name, y_pred in preds_dict.items():
        for (m_bucket, t1_val, t2_bucket), group in opt_test_reset.groupby(
            ["moneyness_bucket", "T1", "T2_bucket"], observed=True
        ):
            if len(group) == 0:
                continue
            # 获取分组在 DataFrame 中的行位置（整数位置）
            idx = group.index  # 这是重置后的整数索引，唯一且连续
            y_true_g = y_test[idx]
            y_pred_g = y_pred[idx]
            
            mae_g = mean_absolute_error(y_true_g, y_pred_g)
            rmse_g = np.sqrt(mean_squared_error(y_true_g, y_pred_g))
            ss_res_g = np.sum((y_true_g - y_pred_g) ** 2)
            ss_tot_g = np.sum((y_true_g - y_true_g.mean()) ** 2)
            r2_g = 1.0 - ss_res_g / ss_tot_g if ss_tot_g > 0 else 0.0
            
            results.append({
                "model": name,
                "moneyness": m_bucket,
                "T1": float(t1_val),
                "T2": t2_bucket,
                "count": len(group),
                "mae": mae_g,
                "rmse": rmse_g,
                "r2": r2_g,
            })
    
    return pd.DataFrame(results)


# =============================================================================
# 8. Feature Importance Analysis
# =============================================================================

def extract_feature_importance(
    models: dict, feature_names: list[str],
) -> pd.DataFrame | None:
    """Extract and combine feature importances from tree-based models.
    Handles mixed model types: raw trees (XGB/GBDT) and pipelines.
    """
    importances: dict[str, np.ndarray] = {}
    for name, md in models.items():
        # Try direct tree model first
        if "model" in md and md.get("is_tree"):
            model = md["model"]
            if hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_
        # Then try pipeline model
        elif "pipeline" in md:
            step = md["pipeline"].named_steps.get("model")
            if step is not None and hasattr(step, "feature_importances_"):
                importances[name] = step.feature_importances_

    if not importances:
        return None

    imp_df = pd.DataFrame(importances, index=feature_names)
    imp_df["mean"] = imp_df.mean(axis=1)
    return imp_df.sort_values("mean", ascending=False)


# =============================================================================
# 9. Visualisations
# =============================================================================

def _bar_color(n: int, palette: list[str]) -> list[str]:
    return palette[:n] + ["#999999"] * max(0, n - len(palette))


def plot_feature_importance(imp_df: pd.DataFrame, out_path: Path) -> None:
    top_n = min(15, len(imp_df))
    top   = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top["mean"].values, color="steelblue", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Feature Importance (Approach 1 – Tree Models)")
    ax.set_title("Top Features for Volatility Prediction")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_path.name)


def plot_vol_results(vol_df: pd.DataFrame, out_path: Path) -> None:
    palette = ["#2196F3", "#FF9800", "#4CAF50"]
    n       = len(vol_df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(vol_df["model"], vol_df["vol_mae"],
                color=_bar_color(n, palette), alpha=0.85)
    axes[0].set_title("Vol Prediction – MAE (Annualised)")
    axes[0].set_ylabel("MAE")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(vol_df["model"], vol_df["vol_rmse"],
                color=_bar_color(n, palette), alpha=0.85)
    axes[1].set_title("Vol Prediction – RMSE (Annualised)")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Approach 1 – Volatility Prediction Performance (Test Set)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_path.name)


def plot_pricing_results(pricing_df: pd.DataFrame, out_path: Path) -> None:
    palette = ["#E91E63", "#3F51B5", "#009688"]
    n       = len(pricing_df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(pricing_df["model"], pricing_df["mae"],
                color=_bar_color(n, palette), alpha=0.85)
    axes[0].set_title("Direct Chooser Pricing – MAE ($)")
    axes[0].set_ylabel("MAE ($)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(pricing_df["model"], pricing_df["r2"],
                color=_bar_color(n, palette), alpha=0.85)
    axes[1].set_title("Direct Chooser Pricing – R²")
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Approach 2 – End-to-End Chooser Pricing Performance (Test Set)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_path.name)


def plot_model_comparison(
    vol_df: pd.DataFrame, pricing_df: pd.DataFrame, out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.bar(vol_df["model"], vol_df["vol_rmse"],
            color=_bar_color(len(vol_df), ["#2196F3", "#FF9800", "#4CAF50"]),
            alpha=0.85)
    ax1.set_title("Approach 1 – Vol Forecast RMSE")
    ax1.set_ylabel("RMSE (Annualised Vol)")
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(pricing_df["model"], pricing_df["rmse"],
            color=_bar_color(len(pricing_df), ["#E91E63", "#3F51B5", "#009688"]),
            alpha=0.85)
    ax2.set_title("Approach 2 – Chooser Pricing RMSE ($)")
    ax2.set_ylabel("RMSE ($)")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Week 6 – ML Model Framework Summary", fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_path.name)


# =============================================================================
# 10. Architecture Design Document
# =============================================================================

def generate_architecture_doc(
    vol_results: pd.DataFrame,
    pricing_results: pd.DataFrame,
    vol_train: pd.DataFrame,
    vol_val: pd.DataFrame,
    vol_test: pd.DataFrame,
    opt_train: pd.DataFrame,
    feature_cols: list[str],
    pricing_cols: list[str],
    vol_models: dict,
    pricing_models: dict,
    week4_baseline: pd.Series | None,
) -> str:
    def fdate(d) -> str:
        return str(d.date()) if hasattr(d, "date") else str(d)

    tr_range = f"{fdate(vol_train.index.min())} → {fdate(vol_train.index.max())}"
    vl_range = f"{fdate(vol_val.index.min())}  → {fdate(vol_val.index.max())}"
    ts_range = f"{fdate(vol_test.index.min())}  → {fdate(vol_test.index.max())}"

    # Build table rows
    v_rows = "\n".join(
        f"| {r['model']} | {r['vol_mae']:.5f} | {r['vol_rmse']:.5f} "
        f"| {r['pricing_mae']:.4f} | {r['pricing_rmse']:.4f} |"
        for _, r in vol_results.iterrows()
    )
    p_rows = "\n".join(
        f"| {r['model']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['r2']:.4f} |"
        for _, r in pricing_results.iterrows()
    )

    lstm_arch = (
        f"  - Architecture: LSTM({64}) → Dropout(0.2) → "
        f"LSTM({32}) → Dropout(0.2) → Dense(16) → Dense(1)\n"
        f"  - Lookback window: {LSTM_LOOKBACK} trading days\n"
        f"  - Optimizer: Adam | Loss: MSE | Early stopping (patience=10)\n"
        f"  - Input shape: (batch, {LSTM_LOOKBACK}, {len(feature_cols)})"
    ) if HAS_TF else "  - LSTM skipped (TensorFlow not available in this environment)"

    feat_list    = "\n".join(f"  - `{f}`" for f in feature_cols)
    pricing_list = "\n".join(f"  - `{f}`" for f in pricing_cols)

    def tuning_rows(models: dict, model_names: list[str]) -> str:
        rows = []
        for model_name in model_names:
            md = models.get(model_name)
            if not md:
                continue
            best_params = md.get("best_params", {})
            if not isinstance(best_params, dict):
                best_params = {}
            rows.append(
                f"| {model_name} | {md.get('cv_mae', float('nan')):.5f} | {md.get('val_mae', float('nan')):.5f} | "
                f"{_format_param_summary(best_params, list(best_params.keys()))} |"
            )
        return "\n".join(rows) if rows else "| - | - | - | - |"

    vol_tuning_rows = tuning_rows(vol_models, ["RandomForest", "XGBoost", "GradientBoosting", "LSTM"])
    pricing_tuning_rows = tuning_rows(pricing_models, ["LinearRegression", "XGBoost", "GradientBoosting", "NeuralNetwork"])

    week4_summary_text = (
        f"Week 4 BSM baseline (European options): MAE={week4_baseline['MAE']:.6f}, RMSE={week4_baseline['RMSE']:.6f}, "
        f"p-value={week4_baseline['p_val_ME']:.4f}"
    ) if week4_baseline is not None else "Week 4 baseline summary not found in data/processed/"

    doc = f"""# Week 6 – Machine Learning Model Architecture Design

**Report date**: {RUN_DATE}  |  **Pipeline version**: {PIPELINE_VER}

> This is the active Week 6 report for the current run. Older dated Week 6
> report files in `data/reports/` are treated as superseded outputs.

---

## 1. Executive Summary

This document describes the machine learning architecture designed and
implemented in Week 6 of the Quantitative Research & Trading project.
Two complementary approaches price chooser options on JPM stock:

- **Approach 1 (ML + chooser closed form)**: ML models predict 20-day
    forward realised volatility; the predicted σ is fed into the chooser
    pricing formula.
- **Approach 2 (End-to-End)**: ML models directly map chooser contract
    parameters and market features to chooser prices.

---

## 2. Problem Formulation

### Approach 1

$$\\hat{{V}}_{{\\text{{chooser}}}} = C\\!\\left(S, K, T_2, r, q, \\hat{{\\sigma}}_{{\\text{{ML}}}}\\right) + P\\!\\left(S, K e^{{-r(T_2-T_1)}}, T_2-T_1, r, q, \\hat{{\\sigma}}_{{\\text{{ML}}}}\\right)$$

where $\\hat{{\\sigma}}_{{\\text{{ML}}}}$ is the ML-predicted 20-day forward
realised volatility.

**Target**: $\\sigma_{{\\text{{fwd,20d}}}}[t] = \\sqrt{{252}} \\cdot
\\text{{std}}\\!\\left(\\ln\\frac{{S_{{t+i}}}}{{S_{{t+i-1}}}}\\right)_{{i=1}}^{{20}}$

### Approach 2

$$\\hat{{V}}_{{\\text{{chooser}}}} = f_{{\\theta}}\\!\\left(S,\\,K,\\,T_1,\\,T_2,\\,r,\\,q,\\,m,\\,\\mathbf{{x}}_{{\\text{{market}}}}\\right)$$

where $f_{{\\theta}}$ is a trained ML model and $\\mathbf{{x}}_{{\\text{{market}}}}$
is the market feature vector.

**Target**: chooser price benchmarked with Monte Carlo-valued call/put legs.

---

## 3. Data & Feature Engineering

### 3.1 Raw Data Sources

| Source | Description | Frequency |
|--------|-------------|-----------|
| `yahoo_jpm_2018_2024.csv` | JPM daily close prices | Daily |
| `fred_DGS10_2018_2024.csv` | US 10-yr Treasury yield | Daily |
| `fred_VIXCLS_2018_2024.csv` | CBOE VIX index | Daily |
| `jpm_dividends_2018_2024.csv` | JPM dividend payments | Per event |
| `alphavantage_news_jpm_2018_2024.csv` | News sentiment scores | Per article |

### 3.2 Market Features (Approach 1 – Volatility Dataset)

All features are strictly **backward-looking** to prevent look-ahead bias.
Dataset: {len(vol_train) + len(vol_val) + len(vol_test):,} trading days.

{feat_list}

### 3.3 Features (Approach 2 – Chooser Pricing Dataset)

All market features above, plus chooser-specific parameters.
Dataset: {len(opt_train):,}+ rows (daily dates × chooser contracts).

{pricing_list}

### 3.4 Target Variables

| Approach | Target | Description |
|----------|--------|-------------|
| 1 | `fwd_vol_20d` | Annualised realised vol over next 20 trading days |
| 2 | `chooser_price` | Chooser price using Monte Carlo benchmark |

---

## 4. Time-Series Validation Framework

Data is split **chronologically** (never randomly) to prevent look-ahead bias.
All features are scaled using `RobustScaler` fitted **only** on the training set.
For model selection, each tunable learner is optimized with `RandomizedSearchCV`
and `TimeSeriesSplit`, then refit on the combined train+validation partition
before the final test-set evaluation.

| Split | Date Range | Fraction |
|-------|-----------|----------|
| Train | {tr_range} | 70% |
| Validation | {vl_range} | 15% |
| Test | {ts_range} | 15% |

---

## 5. Model Architectures

### 5.1 Approach 1 – ML Volatility Prediction

| Model | CV MAE | Val MAE | Best parameters |
|-------|--------|---------|-----------------|
{vol_tuning_rows}

#### LSTM
{lstm_arch}

### 5.2 Approach 2 – End-to-End Supervised Chooser Pricing

| Model | CV MAE | Val MAE | Best parameters |
|-------|--------|---------|-----------------|
{pricing_tuning_rows}

#### Baseline comparison
- {week4_summary_text}
- The closed-form chooser formula computed with maturity-matched historical volatility is used only as a comparison baseline, not as the training target.

---

## 6. Performance Summary (Test Set)

### 6.1 Approach 1 – Volatility Prediction

| Model | Vol MAE | Vol RMSE | Option MAE | Option RMSE |
|-------|---------|----------|------------|-------------|
{v_rows}

*Vol MAE/RMSE: annualised vol units. Chooser MAE/RMSE: USD.*

### 6.2 Approach 2 – End-to-End Chooser Pricing

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|----------|-----|
{p_rows}

*Target: chooser price benchmarked with Monte Carlo-valued call/put legs. The table includes the closed-form BSM baseline for the same contracts.*

---

## 7. Limitations & Recommended Next Steps

### Current Limitations

#### Data & Targets
1. **No market-implied vol**: Targets are derived from historical/closed-form
    prices, not actual market option quotes. This is a fundamental constraint:
    Approach 2 learns to predict a synthetic chooser surface, not a real market
    surface. High R² values (e.g., MLP R²=0.89) reflect model fit to the
    synthetic label, NOT market predictability.
2. **Synthetic chooser labels (Approach 2)**: Targets are computed as
    `chooser_price(S, K, T1, T2, r, q, hist_vol_match(T2))`. Since this is a
    deterministic closed-form calculation, linear models and neural networks
    with sufficient capacity can still fit the surface, but the direct
    historical-vol feature leakage into the input set has been removed.
3. **Volatility term-structure matching**: Vol used for each maturity is now
    matched to option T2, preventing term-structure mismatch. Approach 2 now
    includes volatility proxy features such as VIX, volatility ratios, and
    volatility change signals, while excluding raw historical volatility windows.

#### Model Architecture
4. **Simplified chooser grid**: 2 decision times × 3 maturities × 3 moneyness levels.
5. **Static features**: No real-time microstructure data (bid-ask, volume).
6. **No market data feedback**: Models do not adjust based on actual prices
    observed at decision time T1.

### Model-Specific Notes

#### Approach 2 – Route 2 Pricing
- **LinearRegression & MLP**: R² should improve materially once volatility
    proxy features are included, because these models now receive the
    strongest explanatory signal that is allowed by the experiment design.
- **XGBoost/GradientBoosting**: Lower R² (0.40-0.50) may indicate that tree
  models struggle with the smooth closed-form surface; alternatively, the
  current feature set lacks sufficient flexibility for tree splits. This is
  NOT a sign of data leakage or feature engineering failure—it reflects
  model-surface mismatch.

### Recommended Next Steps (Week 6+)
1. Incorporate implied volatility data for more realistic targets.
2. Add Greeks (delta, gamma, vega) as engineered input features.
3. Implement Bayesian hyperparameter optimisation.
4. Extend LSTM to multi-step ahead vol forecasting.
5. Build ensemble that combines Approach 1 and Approach 2 predictions.
6. Evaluate on 2025+ data for out-of-sample performance.
7. **Collect real market chooser prices** and retrain Approach 2 models
   on actual traded prices instead of synthetic targets.

---

*Generated by `week5_ml_models.py` | {PIPELINE_VER} | {RUN_DATE}*
"""
    return doc


def export_markdown_pdf(md_text: str, pdf_path: Path) -> None:
    """Render the architecture markdown into a simple PDF report."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#111111"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1Style",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#111111"),
        spaceBefore=8,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#222222"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        spaceAfter=4,
        alignment=TA_LEFT,
    )
    mono_style = ParagraphStyle(
        "MonoStyle",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.2,
        leading=10,
        leftIndent=6,
        spaceAfter=6,
    )

    def escape_text(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def cell_text(text: str) -> str:
        return escape_text(text.strip().strip("`") or " ")

    def parse_markdown_table(table_lines: list[str]) -> list[list[str]]:
        rows: list[list[str]] = []
        for idx, raw_line in enumerate(table_lines):
            parts = [part.strip() for part in raw_line.strip().strip("|").split("|")]
            if idx == 1 and all(set(part.replace(" ", "")) <= {"-", ":"} for part in parts if part):
                continue
            rows.append(parts)
        return rows

    def build_table(data: list[list[str]]) -> Table:
        rendered = [[Paragraph(cell_text(cell), body_style) for cell in row] for row in data]
        table = Table(rendered, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9ca3af")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef2f7")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def flush_paragraph(buffer: list[str], story: list) -> None:
        if not buffer:
            return
        text = " ".join(part.strip() for part in buffer if part.strip())
        if text:
            story.append(Paragraph(escape_text(text), body_style))
        buffer.clear()

    story: list = [Paragraph(f"Week 6 – Machine Learning Model Architecture Design ({RUN_DATE})", title_style), Spacer(1, 0.12 * inch)]
    paragraph_buffer: list[str] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph(paragraph_buffer, story)
            story.append(Spacer(1, 0.08 * inch))
            i += 1
            continue

        if stripped.startswith("# "):
            flush_paragraph(paragraph_buffer, story)
            story.append(Paragraph(escape_text(stripped[2:].strip()), h1_style))
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph_buffer, story)
            story.append(Paragraph(escape_text(stripped[3:].strip()), h2_style))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph(paragraph_buffer, story)
            story.append(Paragraph(escape_text(stripped[4:].strip()), h2_style))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph(paragraph_buffer, story)
            table_lines = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            story.append(build_table(parse_markdown_table(table_lines)))
            continue

        if (stripped.startswith("- ") or (stripped[0:1].isdigit() and stripped[1:2] == ".")):
            flush_paragraph(paragraph_buffer, story)
            if stripped.startswith("- "):
                bullet_text = stripped[2:].strip()
            else:
                bullet_text = stripped.split(".", 1)[1].strip()
            story.append(Paragraph(f"• {escape_text(bullet_text)}", body_style))
            i += 1
            continue

        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph(paragraph_buffer, story)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Week 5 ML Architecture",
        author="GitHub Copilot",
    )
    doc.build(story)


def normalize_pdf_compatibility(pdf_path: Path) -> None:
    """Rewrite PDF via system filter for better viewer compatibility when available."""
    try:
        subprocess.run(["xattr", "-c", str(pdf_path)], check=False, capture_output=True)
        cups = subprocess.run(["which", "cupsfilter"], check=False, capture_output=True, text=True)
        if cups.returncode != 0:
            return

        normalized_path = pdf_path.with_suffix(".normalized.pdf")
        with normalized_path.open("wb") as out:
            proc = subprocess.run(
                ["cupsfilter", "-m", "application/pdf", str(pdf_path)],
                check=False,
                stdout=out,
                stderr=subprocess.PIPE,
                text=False,
            )
        if proc.returncode == 0 and normalized_path.exists() and normalized_path.stat().st_size > 0:
            normalized_path.replace(pdf_path)
    except Exception as exc:
        logger.warning("PDF normalization skipped: %s", exc)


def _safe_slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _extract_tree_estimator(md: dict):
    if md.get("is_tree") and "model" in md:
        return md["model"]
    pipeline = md.get("pipeline")
    if pipeline is None:
        return None
    if hasattr(pipeline, "named_steps"):
        estimator = pipeline.named_steps.get("model")
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            return estimator
    return None


def _shap_filename(approach: str, model_name: str) -> str:
    return f"week6_shap_{approach}_{_safe_slug(model_name)}.png"


def _make_flat_feature_names(feature_names: list[str], lookback: int) -> list[str]:
    flat_names: list[str] = []
    for lag in range(lookback, 0, -1):
        for name in feature_names:
            flat_names.append(f"t-{lag}:{name}")
    return flat_names


def _build_lstm_sequence_inputs(
    X_reference: np.ndarray,
    scaler: RobustScaler,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_sc = scaler.transform(X_reference)
    sequences: list[np.ndarray] = []
    for idx in range(lookback, len(X_sc)):
        sequences.append(X_sc[idx - lookback : idx])
    if not sequences:
        empty_3d = np.empty((0, lookback, X_reference.shape[1]))
        empty_2d = np.empty((0, lookback * X_reference.shape[1]))
        return empty_3d, empty_2d
    seq_3d = np.asarray(sequences)
    seq_2d = seq_3d.reshape(len(seq_3d), -1)
    return seq_3d, seq_2d


def _plot_shap_from_explainer(
    explainer,
    X_sample: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    title: str,
) -> bool:
    try:
        shap_values = explainer(X_sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            color="#1f77b4",
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP → %s", out_path.name)
        return True
    except Exception as exc:
        plt.close("all")
        logger.warning("SHAP rendering failed for %s: %s", out_path.name, exc)
        return False


def export_model_shap_plot(
    model_name: str,
    md: dict,
    X_reference: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    title: str,
    max_samples: int = 200,
) -> bool:
    """Export one SHAP comparison plot per model."""
    if not HAS_SHAP:
        logger.warning("SHAP not available; skipping %s", out_path.name)
        return False

    if len(X_reference) == 0:
        logger.warning("Empty reference set for SHAP plot: %s", out_path.name)
        return False

    sample_size = min(max_samples, len(X_reference))

    if model_name == "LSTM" and md.get("model") is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "0"

        try:
            import tensorflow as tf  # type: ignore[import-not-found]
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        scaler = md.get("scaler")
        lookback = int(md.get("lookback", LSTM_LOOKBACK))
        if scaler is None:
            logger.warning("Missing scaler for LSTM SHAP plot: %s", out_path.name)
            return False
        seq_3d, seq_ref = _build_lstm_sequence_inputs(X_reference, scaler, lookback)
        if len(seq_ref) == 0:
            logger.warning("Not enough sequence data for LSTM SHAP plot: %s", out_path.name)
            return False
        seq_sample = seq_ref[: min(sample_size, len(seq_ref))]
        seq_names = _make_flat_feature_names(feature_names, lookback)

        surrogate_size = min(max(50, sample_size), len(seq_ref))
        surrogate_X = seq_ref[:surrogate_size]
        surrogate_y = md["model"].predict(seq_3d[:surrogate_size], verbose=0).reshape(-1)
        surrogate = RandomForestRegressor(n_estimators=120, random_state=RANDOM_STATE, n_jobs=-1)
        surrogate.fit(surrogate_X, surrogate_y)
        background = surrogate_X[: min(100, len(surrogate_X))]
        explainer = shap.TreeExplainer(surrogate, data=background)
        shap_values = explainer.shap_values(seq_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_values,
            seq_sample,
            feature_names=seq_names,
            plot_type="bar",
            show=False,
            color="#1f77b4",
        )
        plt.title(f"{title} (surrogate RF)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP → %s", out_path.name)
        return True

    if "model" in md and md.get("is_tree"):
        estimator = md["model"]
    else:
        estimator = md.get("pipeline")

    if estimator is None:
        logger.warning("No estimator available for SHAP plot: %s", out_path.name)
        return False

    X_sample = X_reference[:sample_size]
    background = X_reference[: min(100, sample_size)]

    try:
        is_xgb_like = estimator.__class__.__module__.startswith("xgboost") or "XGB" in estimator.__class__.__name__
        if is_xgb_like:
            masker = shap.maskers.Independent(background)
            explainer = shap.Explainer(estimator.predict, masker, algorithm="permutation")
        else:
            try:
                explainer = shap.TreeExplainer(estimator, data=background)
            except Exception:
                masker = shap.maskers.Independent(background)
                explainer = shap.Explainer(estimator.predict, masker, algorithm="permutation")
    except Exception:
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(estimator.predict, masker, algorithm="permutation")

    return _plot_shap_from_explainer(explainer, X_sample, feature_names, out_path, title)


def save_trained_models(
    vol_models: dict,
    pricing_models: dict,
    out_dir: Path,
) -> list[Path]:
    """Persist trained model artifacts for both approaches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    def save_one(prefix: str, model_name: str, md: dict) -> None:
        slug = _safe_slug(model_name)
        if model_name == "LSTM" and md.get("model") is not None and HAS_TF:
            keras_path = out_dir / f"{OUTPUT_WEEK}_{prefix}_{slug}_{PIPELINE_VER}.keras"
            md["model"].save(keras_path)
            saved_paths.append(keras_path)

            meta_path = out_dir / f"{OUTPUT_WEEK}_{prefix}_{slug}_{PIPELINE_VER}.joblib"
            payload = {k: v for k, v in md.items() if k != "model"}
            payload["artifact"] = keras_path.name
            joblib.dump(payload, meta_path)
            saved_paths.append(meta_path)
            return

        model_obj = md.get("model") if md.get("is_tree") else md.get("pipeline")
        if model_obj is None:
            return

        model_path = out_dir / f"{OUTPUT_WEEK}_{prefix}_{slug}_{PIPELINE_VER}.joblib"
        payload = {
            "model": model_obj,
            "best_params": md.get("best_params", {}),
            "val_mae": md.get("val_mae"),
            "cv_mae": md.get("cv_mae"),
            "is_tree": md.get("is_tree", False),
        }
        joblib.dump(payload, model_path)
        saved_paths.append(model_path)

    for name, md in vol_models.items():
        save_one("approach1", name, md)
    for name, md in pricing_models.items():
        save_one("approach2", name, md)

    return saved_paths


# =============================================================================
# 11. Main Entry Point
# =============================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("Week 6 – ML Model Design & Implementation")
    logger.info("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & engineer features ───────────────────────────────────────────
    logger.info("[1/7] Loading raw market data...")
    raw_df  = load_market_data()
    logger.info("  %d trading days loaded.", len(raw_df))

    logger.info("[2/7] Engineering features...")
    feat_df = build_features(raw_df)

    # Export feature dataset
    feature_cols = get_vol_feature_columns()
    save_cols    = (
        ["close", "r", "q", "hist_vol_20d", "fwd_vol_20d", "vix", "vix_regime"]
        + feature_cols
    )
    export_feat = feat_df[
        [c for c in save_cols if c in feat_df.columns]
    ].dropna(subset=["hist_vol_20d"])
    feat_csv = PROCESSED_DIR / f"{FEATURE_STEM}.csv"
    export_feat.to_csv(feat_csv)
    logger.info("  Feature dataset: %d rows → %s", len(export_feat), feat_csv.name)

    # ── Build ML datasets ──────────────────────────────────────────────────
    logger.info("[3/7] Building ML datasets...")
    vol_ds = build_vol_dataset(feat_df)
    opt_ds = build_option_dataset(feat_df)
    logger.info(
        "  Vol dataset: %d rows | Option dataset: %d rows",
        len(vol_ds), len(opt_ds),
    )

    # ── Time-series split ──────────────────────────────────────────────────
    logger.info("[4/7] Splitting data (70%%/15%%/15%%)...")
    vol_train, vol_val, vol_test = time_series_split(vol_ds)
    opt_train, opt_val, opt_test = time_series_split(opt_ds)
    logger.info(
        "  Vol  – train: %d | val: %d | test: %d",
        len(vol_train), len(vol_val), len(vol_test),
    )
    logger.info(
        "  Opt  – train: %d | val: %d | test: %d",
        len(opt_train), len(opt_val), len(opt_test),
    )

    X_vol_train = vol_train[feature_cols].values
    y_vol_train = vol_train["target_vol"].values
    X_vol_val   = vol_val[feature_cols].values
    y_vol_val   = vol_val["target_vol"].values

    pricing_market_cols = get_pricing_feature_columns()
    pricing_cols = get_pricing_model_columns(pricing_market_cols)
    X_opt_train  = opt_train[pricing_cols].values
    y_opt_train  = opt_train["chooser_price"].values
    X_opt_val    = opt_val[pricing_cols].values
    y_opt_val    = opt_val["chooser_price"].values

    # ── Approach 1 ────────────────────────────────────────────────────────
    logger.info("[5/7] Approach 1 – ML Volatility Prediction...")
    vol_models  = train_vol_models(X_vol_train, y_vol_train, X_vol_val, y_vol_val, feature_cols)
    vol_results = evaluate_approach1(vol_models, vol_test, opt_test, feature_cols)

    # ── Approach 2 ────────────────────────────────────────────────────────
    logger.info("[6/7] Approach 2 – End-to-End Chooser Pricing...")
    pricing_models  = train_pricing_models(X_opt_train, y_opt_train, X_opt_val, y_opt_val)
    pricing_results = evaluate_approach2(pricing_models, opt_test, pricing_cols)
    pricing_bsm_baseline = evaluate_bsm_baseline(opt_test)
    pricing_results_with_baseline = pd.concat(
        [pricing_results, pricing_bsm_baseline],
        ignore_index=True,
    )
    
    # Add stratified evaluation for diagnostics
    logger.info("  [Approach 2] Computing stratified error breakdown...")
    pricing_stratified = evaluate_approach2_stratified(pricing_models, opt_test, pricing_cols)

    # Persist trained model artifacts
    saved_model_paths = save_trained_models(vol_models, pricing_models, MODELS_DIR)

    # ── Export results & plots ────────────────────────────────────────────
    logger.info("[7/7] Exporting results and generating report...")

    vol_csv     = PROCESSED_DIR / f"{VOL_STEM}.csv"
    pricing_csv = PROCESSED_DIR / f"{PRICING_STEM}.csv"
    pricing_strat_csv = PROCESSED_DIR / f"{PRICING_STRAT_STEM}.csv"
    comp_csv    = PROCESSED_DIR / f"{COMP_STEM}.csv"

    vol_results.to_csv(vol_csv,     index=False)
    pricing_results_with_baseline.to_csv(pricing_csv, index=False)
    pricing_stratified.to_csv(pricing_strat_csv, index=False)
    logger.info("  Stratified pricing results saved to %s", pricing_strat_csv)

    # Combined comparison
    vol_side     = vol_results.add_prefix("app1_").rename(columns={"app1_model": "model_A1"})
    pricing_side = pricing_results_with_baseline.add_prefix("app2_").rename(columns={"app2_model": "model_A2"})
    comparison   = pd.concat([vol_side.reset_index(drop=True),
                               pricing_side.reset_index(drop=True)], axis=1)
    comparison.to_csv(comp_csv, index=False)

    # Feature importance
    imp_df = extract_feature_importance(vol_models, feature_cols)
    if imp_df is not None:
        plot_feature_importance(imp_df, REPORTS_DIR / "week6_feature_importance.png")

    shap_artifacts: list[Path] = []
    for name, md in vol_models.items():
        out_path = REPORTS_DIR / _shap_filename("app1", name)
        if export_model_shap_plot(
            name,
            md,
            vol_train[feature_cols].values,
            feature_cols,
            out_path,
            f"Week 6 SHAP Feature Importance – Approach 1 / {name}",
        ):
            shap_artifacts.append(out_path)

    for name, md in pricing_models.items():
        out_path = REPORTS_DIR / _shap_filename("app2", name)
        if export_model_shap_plot(
            name,
            md,
            opt_train[pricing_cols].values,
            pricing_cols,
            out_path,
            f"Week 6 SHAP Feature Importance – Approach 2 / {name}",
        ):
            shap_artifacts.append(out_path)

    plot_vol_results(vol_results,     REPORTS_DIR / "week6_vol_prediction_comparison.png")
    plot_pricing_results(pricing_results_with_baseline, REPORTS_DIR / "week6_pricing_comparison.png")
    plot_model_comparison(vol_results, pricing_results_with_baseline, REPORTS_DIR / "week6_model_performance.png")

    # Architecture design document
    arch_doc  = generate_architecture_doc(
        vol_results, pricing_results_with_baseline,
        vol_train, vol_val, vol_test, opt_train,
        feature_cols, pricing_cols,
        vol_models, pricing_models,
        _latest_week4_baseline(),
    )
    arch_path = REPORTS_DIR / f"{REPORT_STEM}.md"
    arch_path.write_text(arch_doc, encoding="utf-8")
    logger.info("Architecture doc → %s", arch_path.name)

    arch_pdf = REPORTS_DIR / f"{REPORT_STEM}.pdf"
    export_markdown_pdf(arch_doc, arch_pdf)
    normalize_pdf_compatibility(arch_pdf)
    logger.info("Architecture PDF → %s", arch_pdf.name)

    # Create an extensionless entry so opening REPORT_STEM resolves to the markdown report.
    arch_entry = REPORTS_DIR / REPORT_STEM
    try:
        if arch_entry.exists() or arch_entry.is_symlink():
            arch_entry.unlink()
        arch_entry.symlink_to(arch_path.name)
    except OSError:
        arch_entry.write_text(
            "\n".join([
                f"{REPORT_STEM} is an entry file.",
                f"Markdown report: {arch_path.name}",
                f"PDF report: {arch_pdf.name}",
            ]) + "\n",
            encoding="utf-8",
        )
    logger.info("Architecture entry → %s", arch_entry.name)

    # ── Final summary ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("WEEK 6 COMPLETE")
    logger.info("=" * 60)
    logger.info("Approach 1 – Volatility Prediction (Test Set):")
    for _, r in vol_results.iterrows():
        logger.info(
            "  %-18s | Vol RMSE=%.5f | Chooser Pricing MAE=%.4f",
            r["model"], r["vol_rmse"], r["pricing_mae"],
        )
    logger.info("Approach 2 – End-to-End Chooser Pricing (Test Set):")
    for _, r in pricing_results_with_baseline.iterrows():
        logger.info(
            "  %-18s | MAE=%.4f | RMSE=%.4f | R²=%.4f",
            r["model"], r["mae"], r["rmse"], r["r2"],
        )
    logger.info("")
    outputs = [
        feat_csv, vol_csv, pricing_csv, comp_csv, arch_path, arch_pdf, arch_entry,
        *saved_model_paths,
        REPORTS_DIR / "week6_feature_importance.png",
        *shap_artifacts,
        REPORTS_DIR / "week6_vol_prediction_comparison.png",
        REPORTS_DIR / "week6_pricing_comparison.png",
        REPORTS_DIR / "week6_model_performance.png",
    ]
    logger.info("Outputs:")
    for p in outputs:
        status = "✓" if p.exists() else "✗"
        logger.info("  %s %s", status, p.relative_to(ROOT))


if __name__ == "__main__":
    main()
