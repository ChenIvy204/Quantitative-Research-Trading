"""
Week 5 – Machine Learning Model Design & Implementation
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
  data/processed/week5_feature_dataset_v1.0_<date>.csv
  data/processed/week5_vol_results_v1.0_<date>.csv
  data/processed/week5_pricing_results_v1.0_<date>.csv
  data/processed/week5_model_comparison_v1.0_<date>.csv
  data/reports/week5_ml_architecture_v1.0_<date>.md
  data/reports/week5_feature_importance.png
  data/reports/week5_vol_prediction_comparison.png
  data/reports/week5_pricing_comparison.png
  data/reports/week5_model_performance.png
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    HAS_TF = True
except ImportError:
    HAS_TF = False

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR   = ROOT / "data" / "reports"
PIPELINE_VER  = "v1.0"
RUN_DATE      = datetime.now().strftime("%Y%m%d")

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
VIX_LOW         = 20.0
VIX_HIGH        = 30.0

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
    df["dgs10"] = df["dgs10"].ffill().bfill()
    df["vix"]   = df["vix"].ffill().bfill()

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

    # ── Backward volatility features ──────────────────────────────────────
    out["hist_vol_5d"]  = out["log_ret"].rolling(5).std()  * sqrt(ANNUALISE)
    out["hist_vol_20d"] = out["log_ret"].rolling(20).std() * sqrt(ANNUALISE)
    out["hist_vol_60d"] = out["log_ret"].rolling(60).std() * sqrt(ANNUALISE)

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


def get_feature_columns() -> list[str]:
    """Return the ordered list of input feature names (all backward-looking)."""
    return [
        "hist_vol_5d", "hist_vol_20d", "hist_vol_60d",
        "vol_ratio_5_20", "vol_ratio_20_60", "vol_20d_change",
        "return_1d", "return_5d", "return_20d",
        "price_to_ma_20d", "price_to_ma_60d",
        "vix", "vix_change_5d", "vix_ma_ratio",
        "vix_jpm_corr_20d",
        "r", "q",
        "sentiment_7d", "sentiment_20d", "news_count_7d",
        "drawdown_20d",
    ]


# =============================================================================
# 4. Dataset Construction
# =============================================================================

def build_vol_dataset(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approach 1 dataset: one row per trading day.
    Features: backward-looking market features.
    Target:   fwd_vol_20d (realised vol over next 20 trading days).
    """
    feature_cols = get_feature_columns()
    keep = list(dict.fromkeys(
        feature_cols + ["fwd_vol_20d", "hist_vol_20d", "close", "r", "q", "vix_regime"]
    ))
    df = feat_df[[c for c in keep if c in feat_df.columns]].copy()
    df = df.dropna(subset=feature_cols + ["fwd_vol_20d"])
    df = df.rename(columns={"fwd_vol_20d": "target_vol"})
    return df


def build_option_dataset(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approach 2 dataset: one row per (trading day × chooser contract).
    Features: market features + chooser parameters (S, K, moneyness, T1, T2).
    Target:   chooser price computed with hist_vol_20d as σ.
    """
    feature_cols = get_feature_columns()
    keep = list(dict.fromkeys(
        feature_cols + ["hist_vol_20d", "close", "r", "q", "vix_regime"]
    ))
    daily = feat_df[[c for c in keep if c in feat_df.columns]].dropna(
        subset=feature_cols + ["hist_vol_20d"]
    )

    rows: list[dict] = []
    for date, row in daily.iterrows():
        S     = float(row["close"])
        r     = float(row["r"])
        q     = float(row["q"])
        sigma = float(row["hist_vol_20d"])
        if sigma <= 0 or np.isnan(sigma) or S <= 0:
            continue
        for T1 in CHOOSER_DECISION_TIMES:
            for T2 in CHOOSER_MATURITIES:
                if T2 <= T1:
                    continue
                for m in MONEYNESS:
                    K = S * m
                    price = chooser_price_scalar(S, K, T1, T2, r, q, sigma)
                    rec: dict = {
                        "date": date,
                        "S": S,
                        "K": K,
                        "moneyness": m,
                        "T1": T1,
                        "T2": T2,
                        "chooser_price": price,
                        "vix_regime": row.get("vix_regime", "unknown"),
                    }
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
    """Train RF, XGBoost (or GBDT), and optionally LSTM for vol prediction."""
    models: dict = {}

    # ── Random Forest ──────────────────────────────────────────────────────
    logger.info("  [Approach 1] Training RandomForest...")
    rf = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    rf_val_mae = mean_absolute_error(y_val, rf.predict(X_val))
    logger.info("    RF   val MAE=%.6f", rf_val_mae)
    models["RandomForest"] = {"pipeline": rf, "val_mae": rf_val_mae}

    # ── XGBoost (or sklearn GBDT fallback) ────────────────────────────────
    if HAS_XGB:
        logger.info("  [Approach 1] Training XGBoost...")
        gbm = Pipeline([
            ("scaler", RobustScaler()),
            ("model",  xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbosity=0,
            )),
        ])
    else:
        logger.info("  [Approach 1] Training GradientBoosting (XGBoost not found)...")
        gbm = Pipeline([
            ("scaler", RobustScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_STATE,
            )),
        ])
    gbm.fit(X_train, y_train)
    gbm_name    = "XGBoost" if HAS_XGB else "GradientBoosting"
    gbm_val_mae = mean_absolute_error(y_val, gbm.predict(X_val))
    logger.info("    %-16s val MAE=%.6f", gbm_name, gbm_val_mae)
    models[gbm_name] = {"pipeline": gbm, "val_mae": gbm_val_mae}

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
      - chooser pricing MAE/RMSE (using predicted σ vs historical σ)
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
        opt_eval = opt_eval.dropna(subset=["pred_vol", "chooser_price", "hist_vol_20d"])

        chooser_pred_arr, chooser_base_arr = [], []
        for _, row in opt_eval.iterrows():
            sig_pred = max(float(row["pred_vol"]), 1e-4)
            sig_base = max(float(row["hist_vol_20d"]), 1e-4)
            S, K = float(row["S"]), float(row["K"])
            T1, T2 = float(row["T1"]), float(row["T2"])
            r, q   = float(row["r"]), float(row["q"])
            chooser_pred_arr.append(chooser_price_scalar(S, K, T1, T2, r, q, sig_pred))
            chooser_base_arr.append(chooser_price_scalar(S, K, T1, T2, r, q, sig_base))

        arr_pred = np.array(chooser_pred_arr)
        arr_base = np.array(chooser_base_arr)
        opt_mae  = mean_absolute_error(arr_base, arr_pred)
        opt_rmse = float(np.sqrt(mean_squared_error(arr_base, arr_pred)))

        rows.append({
            "model":        name,
            "vol_mae":      vol_mae,
            "vol_rmse":     vol_rmse,
            "pricing_mae":  opt_mae,
            "pricing_rmse": opt_rmse,
        })
        logger.info(
            "  Approach 1 | %-16s | vol MAE=%.5f RMSE=%.5f | "
            "chooser MAE=%.4f RMSE=%.4f",
            name, vol_mae, vol_rmse, opt_mae, opt_rmse,
        )

    return pd.DataFrame(rows)


# =============================================================================
# 7. Approach 2 – End-to-End Supervised Chooser Pricing
# =============================================================================

def get_pricing_feature_columns(feature_cols: list[str]) -> list[str]:
    return feature_cols + ["S", "K", "moneyness", "T1", "T2"]


def train_pricing_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train LinearRegression, XGBoost/GBDT, and MLP for direct chooser pricing."""
    models: dict = {}

    # ── Linear Regression ─────────────────────────────────────────────────
    logger.info("  [Approach 2] Training LinearRegression...")
    lr = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  LinearRegression()),
    ])
    lr.fit(X_train, y_train)
    lr_val_mae = mean_absolute_error(y_val, lr.predict(X_val))
    logger.info("    LR  val MAE=%.4f", lr_val_mae)
    models["LinearRegression"] = {"pipeline": lr, "val_mae": lr_val_mae}

    # ── XGBoost / Gradient Boosting ───────────────────────────────────────
    if HAS_XGB:
        logger.info("  [Approach 2] Training XGBoost...")
        gbm = Pipeline([
            ("scaler", RobustScaler()),
            ("model",  xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbosity=0,
            )),
        ])
        gbm_name = "XGBoost"
    else:
        logger.info("  [Approach 2] Training GradientBoosting...")
        gbm = Pipeline([
            ("scaler", RobustScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_STATE,
            )),
        ])
        gbm_name = "GradientBoosting"

    gbm.fit(X_train, y_train)
    gbm_val_mae = mean_absolute_error(y_val, gbm.predict(X_val))
    logger.info("    %-16s val MAE=%.4f", gbm_name, gbm_val_mae)
    models[gbm_name] = {"pipeline": gbm, "val_mae": gbm_val_mae}

    # ── Neural Network (MLP) ──────────────────────────────────────────────
    logger.info("  [Approach 2] Training MLP (128-64-32)...")
    mlp = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=False,
        )),
    ])
    mlp.fit(X_train, y_train)
    mlp_val_mae = mean_absolute_error(y_val, mlp.predict(X_val))
    logger.info("    MLP val MAE=%.4f", mlp_val_mae)
    models["NeuralNetwork"] = {"pipeline": mlp, "val_mae": mlp_val_mae}

    return models


def evaluate_approach2(
    models: dict,
    opt_test: pd.DataFrame,
    pricing_cols: list[str],
) -> pd.DataFrame:
    """Evaluate Approach 2: end-to-end chooser pricing on the test set."""
    X_test = opt_test[pricing_cols].values
    y_test = opt_test["chooser_price"].values

    rows = []
    for name, md in models.items():
        y_pred = np.maximum(md["pipeline"].predict(X_test), 0.0)
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


# =============================================================================
# 8. Feature Importance Analysis
# =============================================================================

def extract_feature_importance(
    models: dict, feature_names: list[str],
) -> pd.DataFrame | None:
    """Extract and combine feature importances from tree-based models."""
    importances: dict[str, np.ndarray] = {}
    for name, md in models.items():
        if "pipeline" not in md:
            continue
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

    fig.suptitle("Week 5 – ML Model Framework Summary", fontsize=13, fontweight="bold")
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

    gbm_label = "XGBoost" if HAS_XGB else "GradientBoosting"

    doc = f"""# Week 5 – Machine Learning Model Architecture Design

**Run date**: {RUN_DATE}  |  **Pipeline version**: {PIPELINE_VER}

---

## 1. Executive Summary

This document describes the machine learning architecture designed and
implemented in Week 5 of the Quantitative Research & Trading project.
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

**Target**: chooser price computed with 20-day historical σ.

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
| 2 | `chooser_price` | Chooser price using hist_vol_20d as σ |

---

## 4. Time-Series Validation Framework

Data is split **chronologically** (never randomly) to prevent look-ahead bias.
All features are scaled using `RobustScaler` fitted **only** on the training set.

| Split | Date Range | Fraction |
|-------|-----------|----------|
| Train | {tr_range} | 70% |
| Validation | {vl_range} | 15% |
| Test | {ts_range} | 15% |

---

## 5. Model Architectures

### 5.1 Approach 1 – ML Volatility Prediction

#### Random Forest
- `n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`
- Preprocessing: RobustScaler
- Input: {len(feature_cols)} market features (1 row per trading day)

#### {gbm_label}
- `n_estimators=500`, `max_depth=5`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- Preprocessing: RobustScaler

#### LSTM
{lstm_arch}

### 5.2 Approach 2 – End-to-End Supervised Chooser Pricing

#### Linear Regression
- Standard OLS with intercept, Preprocessing: RobustScaler
- Input: {len(pricing_cols)} features (market + chooser parameters)

#### {gbm_label}
- `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`

#### Neural Network (MLP)
- Architecture: Dense(128) → ReLU → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1)
- Optimizer: Adam (`lr=1e-3`), early stopping (`patience` via `validation_fraction=0.1`)
- `max_iter=500`

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

*Target: chooser price (hist_vol_20d as σ). Evaluation grid: 2T1 × 3T2 × 3K.*

---

## 7. Limitations & Recommended Next Steps

### Current Limitations
1. **No market-implied vol**: Targets are derived from historical/closed-form
    prices, not actual market option quotes.
2. **Simplified chooser grid**: 2 decision times × 3 maturities × 3 moneyness levels.
3. **Static features**: No real-time microstructure data (bid-ask, volume).
4. **Synthetic chooser surface learning**: Approach 2 learns a synthetic
    chooser surface from historical-vol inputs; real market quotes may differ.

### Recommended Next Steps (Week 6+)
1. Incorporate implied volatility data for more realistic targets.
2. Add Greeks (delta, gamma, vega) as engineered input features.
3. Implement Bayesian hyperparameter optimisation.
4. Extend LSTM to multi-step ahead vol forecasting.
5. Build ensemble that combines Approach 1 and Approach 2 predictions.
6. Evaluate on 2025+ data for out-of-sample performance.

---

*Generated by `week5_ml_models.py` | {PIPELINE_VER} | {RUN_DATE}*
"""
    return doc


# =============================================================================
# 11. Main Entry Point
# =============================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("Week 5 – ML Model Design & Implementation")
    logger.info("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & engineer features ───────────────────────────────────────────
    logger.info("[1/7] Loading raw market data...")
    raw_df  = load_market_data()
    logger.info("  %d trading days loaded.", len(raw_df))

    logger.info("[2/7] Engineering features...")
    feat_df = build_features(raw_df)

    # Export feature dataset
    feature_cols = get_feature_columns()
    save_cols    = (
        ["close", "r", "q", "hist_vol_20d", "fwd_vol_20d", "vix", "vix_regime"]
        + feature_cols
    )
    export_feat = feat_df[
        [c for c in save_cols if c in feat_df.columns]
    ].dropna(subset=["hist_vol_20d"])
    feat_csv = PROCESSED_DIR / f"week5_feature_dataset_{PIPELINE_VER}_{RUN_DATE}.csv"
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

    pricing_cols = get_pricing_feature_columns(feature_cols)
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

    # ── Export results & plots ────────────────────────────────────────────
    logger.info("[7/7] Exporting results and generating report...")

    vol_csv     = PROCESSED_DIR / f"week5_vol_results_{PIPELINE_VER}_{RUN_DATE}.csv"
    pricing_csv = PROCESSED_DIR / f"week5_pricing_results_{PIPELINE_VER}_{RUN_DATE}.csv"
    comp_csv    = PROCESSED_DIR / f"week5_model_comparison_{PIPELINE_VER}_{RUN_DATE}.csv"

    vol_results.to_csv(vol_csv,     index=False)
    pricing_results.to_csv(pricing_csv, index=False)

    # Combined comparison
    vol_side     = vol_results.add_prefix("app1_").rename(columns={"app1_model": "model_A1"})
    pricing_side = pricing_results.add_prefix("app2_").rename(columns={"app2_model": "model_A2"})
    comparison   = pd.concat([vol_side.reset_index(drop=True),
                               pricing_side.reset_index(drop=True)], axis=1)
    comparison.to_csv(comp_csv, index=False)

    # Feature importance
    imp_df = extract_feature_importance(vol_models, feature_cols)
    if imp_df is not None:
        plot_feature_importance(imp_df, REPORTS_DIR / "week5_feature_importance.png")

    plot_vol_results(vol_results,     REPORTS_DIR / "week5_vol_prediction_comparison.png")
    plot_pricing_results(pricing_results, REPORTS_DIR / "week5_pricing_comparison.png")
    plot_model_comparison(vol_results, pricing_results, REPORTS_DIR / "week5_model_performance.png")

    # Architecture design document
    arch_doc  = generate_architecture_doc(
        vol_results, pricing_results,
        vol_train, vol_val, vol_test, opt_train,
        feature_cols, pricing_cols,
    )
    arch_path = REPORTS_DIR / f"week5_ml_architecture_{PIPELINE_VER}_{RUN_DATE}.md"
    arch_path.write_text(arch_doc, encoding="utf-8")
    logger.info("Architecture doc → %s", arch_path.name)

    # ── Final summary ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("WEEK 5 COMPLETE")
    logger.info("=" * 60)
    logger.info("Approach 1 – Volatility Prediction (Test Set):")
    for _, r in vol_results.iterrows():
        logger.info(
            "  %-18s | Vol RMSE=%.5f | Chooser Pricing MAE=%.4f",
            r["model"], r["vol_rmse"], r["pricing_mae"],
        )
    logger.info("Approach 2 – End-to-End Chooser Pricing (Test Set):")
    for _, r in pricing_results.iterrows():
        logger.info(
            "  %-18s | MAE=%.4f | RMSE=%.4f | R²=%.4f",
            r["model"], r["mae"], r["rmse"], r["r2"],
        )
    logger.info("")
    outputs = [
        feat_csv, vol_csv, pricing_csv, comp_csv, arch_path,
        REPORTS_DIR / "week5_feature_importance.png",
        REPORTS_DIR / "week5_vol_prediction_comparison.png",
        REPORTS_DIR / "week5_pricing_comparison.png",
        REPORTS_DIR / "week5_model_performance.png",
    ]
    logger.info("Outputs:")
    for p in outputs:
        status = "✓" if p.exists() else "✗"
        logger.info("  %s %s", status, p.relative_to(ROOT))


if __name__ == "__main__":
    main()
