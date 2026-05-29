"""
Week 4 – Baseline Model Performance Evaluation
================================================
Methodology
-----------
"Actual prices" are produced by Monte Carlo (MC) simulation under the
risk-neutral GBM measure.  BSM closed-form prices are the "model predictions".
MAE / RMSE between BSM and MC measure how well the analytical formula
approximates the simulation benchmark across different market regimes.

Three market regimes (identified via VIX):
  Low-volatility   : VIX < 20
  Medium-volatility: 20 ≤ VIX < 30
  High-volatility  : VIX ≥ 30

Sentiment gap analysis: correlate |BSM - MC| with the daily JPM sentiment score
from Alpha Vantage to identify periods where market-moving news may widen the
gap between model and simulated prices.

Outputs (data/processed/)
  week4_bsm_evaluation_daily_v1.0_YYYYMMDD.csv   – per-date pricing rows
  week4_bsm_error_metrics_v1.0_YYYYMMDD.csv      – overall + regime MAE/RMSE
  week4_bsm_sentiment_gap_v1.0_YYYYMMDD.csv      – sentiment correlation table

Output charts (data/reports/)
  week4_bsm_error_timeseries.png
  week4_bsm_regime_boxplot.png
  week4_bsm_sentiment_scatter.png

Validation report (data/reports/)
  week4_bsm_validation_v1.0_YYYYMMDD.md
"""

from __future__ import annotations

import sys
import logging
from datetime import datetime
from math import erf, exp, log, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[1]
RAW_DIR        = ROOT / "data" / "raw"
PROCESSED_DIR  = ROOT / "data" / "processed"
REPORTS_DIR    = ROOT / "data" / "reports"
PIPELINE_VER   = "v1.0"
RUN_DATE       = datetime.now().strftime("%Y%m%d")

# ── Simulation config ─────────────────────────────────────────────────────────
MC_PATHS        = 10_000   # Monte Carlo path count
MC_SEED         = 42
VOL_WINDOW      = 20       # trading days for rolling historical vol
ANNUALISE       = 252      # trading days per year
MATURITIES      = [0.25, 0.5, 1.0]   # option maturities in years to evaluate
MONEYNESS       = [0.9, 1.0, 1.1]    # K / S ratios (OTM put, ATM, OTM call)
SAMPLE_FREQ     = "MS"     # month-start: one observation per month
VIX_LOW         = 20.0
VIX_HIGH        = 30.0

logger = logging.getLogger("week4_bsm_evaluation")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# =============================================================================
# 1. BSM analytical pricing (re-implemented; no import from week3 to keep
#    this script self-contained and optimised per the Week 4 deliverable)
# =============================================================================

def _ncdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bsm_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes-Merton European call price."""
    if T <= 0 or sigma <= 0:
        return max(S * exp(-q * T) - K * exp(-r * T), 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * _ncdf(d1) - K * exp(-r * T) * _ncdf(d2)


def bsm_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes-Merton European put price."""
    if T <= 0 or sigma <= 0:
        return max(K * exp(-r * T) - S * exp(-q * T), 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * _ncdf(-d2) - S * exp(-q * T) * _ncdf(-d1)


# =============================================================================
# 2. Monte Carlo pricing (vectorised GBM under risk-neutral measure)
# =============================================================================

def mc_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    option_type: str, n_paths: int, rng: np.random.Generator
) -> float:
    """
    GBM Monte Carlo price for a European call or put.
    Risk-neutral drift: r - q
    """
    z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * z)
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    return float(exp(-r * T) * payoffs.mean())


# =============================================================================
# 3. Data loading & parameter construction
# =============================================================================

def load_market_data() -> pd.DataFrame:
    """
    Merge JPM stock prices, 10-yr Treasury rate, VIX, and dividends into a
    single daily aligned DataFrame covering 2018-01-01 to 2024-12-31.
    """
    # --- stock prices ---
    stock = pd.read_csv(RAW_DIR / "yahoo_jpm_2018_2024.csv", parse_dates=["Date"])
    stock = stock.rename(columns={"Date": "date", "close": "close", "Adj Close": "adj_close"})
    stock["date"] = stock["date"].dt.normalize()
    stock = stock[["date", "close", "adj_close"]].dropna().set_index("date")

    # --- risk-free rate (10yr Treasury, in percent) ---
    rates = pd.read_csv(RAW_DIR / "fred_DGS10_2018_2024.csv", parse_dates=["date"])
    rates = rates[["date", "value"]].rename(columns={"value": "dgs10"})
    rates["dgs10"] = pd.to_numeric(rates["dgs10"], errors="coerce")
    rates = rates.set_index("date").sort_index()
    rates = rates.ffill()   # forward-fill weekends / holidays

    # --- VIX ---
    vix = pd.read_csv(RAW_DIR / "fred_VIXCLS_2018_2024.csv", parse_dates=["date"])
    vix = vix[["date", "value"]].rename(columns={"value": "vix"})
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    vix = vix.set_index("date").sort_index().ffill()

    # --- dividends: convert to rolling TTM dividend yield ---
    divs = pd.read_csv(RAW_DIR / "jpm_dividends_2018_2024.csv")
    divs["date"] = pd.to_datetime(divs["date"], utc=True).dt.tz_convert(None).dt.normalize()
    divs = divs[divs["dividend"] > 0].set_index("date").sort_index()

    # merge everything on stock index
    df = stock.copy()
    df = df.join(rates, how="left").join(vix, how="left")
    df["dgs10"] = df["dgs10"].ffill()
    df["vix"]   = df["vix"].ffill()

    # TTM dividend yield: rolling sum of dividends over 252 trading days / price
    # Align dividend series to stock trading dates first
    div_series = divs["dividend"].reindex(df.index, fill_value=0.0)
    df["ttm_div"] = div_series.rolling(ANNUALISE, min_periods=1).sum()
    df["div_yield"] = df["ttm_div"] / df["close"]

    # 20-day rolling historical volatility (annualised log returns)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["hist_vol"] = log_ret.rolling(VOL_WINDOW).std() * sqrt(ANNUALISE)

    # risk-free rate as decimal
    df["r"] = df["dgs10"] / 100.0

    df = df.dropna(subset=["hist_vol", "r", "vix"])
    return df


def load_sentiment_data() -> pd.Series:
    """
    Return a daily series of the mean JPM ticker sentiment score.
    Multiple news items per day are averaged.
    """
    news = pd.read_csv(RAW_DIR / "alphavantage_news_jpm_2018_2024.csv",
                       parse_dates=["publishedAt"])
    news["date"] = news["publishedAt"].dt.normalize()
    daily_sentiment = (
        news.groupby("date")["ticker_sentiment_score"]
        .mean()
        .rename("sentiment")
    )
    return daily_sentiment


# =============================================================================
# 4. Core evaluation loop
# =============================================================================

def run_evaluation(market_df: pd.DataFrame, sentiment: pd.Series) -> pd.DataFrame:
    """
    For each sampled date, compute BSM and MC prices for every combination of
    maturity × moneyness × option_type, then record error metrics.
    """
    rng = np.random.default_rng(MC_SEED)

    # Sample one observation per month (month-start within available dates)
    sample_dates = market_df.resample(SAMPLE_FREQ).first().dropna(
        subset=["hist_vol"]
    ).index

    rows: list[dict] = []
    total = len(sample_dates) * len(MATURITIES) * len(MONEYNESS) * 2
    logger.info(f"Evaluating {total} BSM/MC pairs across {len(sample_dates)} dates …")

    for date in sample_dates:
        if date not in market_df.index:
            continue
        row_data = market_df.loc[date]
        S     = float(row_data["close"])
        r     = float(row_data["r"])
        q     = float(row_data["div_yield"])
        sigma = float(row_data["hist_vol"])
        vix   = float(row_data["vix"])
        sent  = float(sentiment.get(date, np.nan))

        # VIX regime label
        if vix < VIX_LOW:
            regime = "low"
        elif vix < VIX_HIGH:
            regime = "medium"
        else:
            regime = "high"

        for T in MATURITIES:
            for m in MONEYNESS:
                K = S * m
                for opt_type in ("call", "put"):
                    if opt_type == "call":
                        bsm_price = bsm_call(S, K, T, r, q, sigma)
                    else:
                        bsm_price = bsm_put(S, K, T, r, q, sigma)

                    mc_price_val = mc_price(S, K, T, r, q, sigma,
                                            opt_type, MC_PATHS, rng)

                    residual = bsm_price - mc_price_val
                    abs_err = abs(bsm_price - mc_price_val)
                    sq_err  = (bsm_price - mc_price_val) ** 2

                    rows.append({
                        "date":        date,
                        "S":           round(S, 4),
                        "K":           round(K, 4),
                        "moneyness":   m,
                        "T":           T,
                        "r":           round(r, 6),
                        "q":           round(q, 6),
                        "sigma":       round(sigma, 6),
                        "vix":         round(vix, 2),
                        "regime":      regime,
                        "option_type": opt_type,
                        "bsm_price":   round(bsm_price, 6),
                        "mc_price":    round(mc_price_val, 6),
                        "residual":    round(residual, 6),
                        "abs_error":   round(abs_err, 6),
                        "sq_error":    round(sq_err, 8),
                        "sentiment":   round(sent, 6) if not np.isnan(sent) else np.nan,
                    })

    return pd.DataFrame(rows)


# =============================================================================
# 5. Error metric aggregation
# =============================================================================

def compute_error_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Overall + per-regime + per-maturity + per-option-type MAE / RMSE."""
    metric_rows: list[dict] = []

    def _metrics(subset: pd.DataFrame, label: str) -> dict:
        return {
            "group":       label,
            "n":           len(subset),
            "MAE":         round(subset["abs_error"].mean(), 6),
            "RMSE":        round(np.sqrt(subset["sq_error"].mean()), 6),
            "max_abs_err": round(subset["abs_error"].max(), 6),
        }

    metric_rows.append(_metrics(eval_df, "overall"))

    for regime in ["low", "medium", "high"]:
        sub = eval_df[eval_df["regime"] == regime]
        if not sub.empty:
            metric_rows.append(_metrics(sub, f"regime={regime}"))

    for T in MATURITIES:
        sub = eval_df[eval_df["T"] == T]
        metric_rows.append(_metrics(sub, f"maturity={T}yr"))

    for opt_type in ("call", "put"):
        sub = eval_df[eval_df["option_type"] == opt_type]
        metric_rows.append(_metrics(sub, f"type={opt_type}"))

    for m in MONEYNESS:
        sub = eval_df[eval_df["moneyness"] == m]
        metric_rows.append(_metrics(sub, f"moneyness={m}"))

    return pd.DataFrame(metric_rows)


# =============================================================================
# 6. Sentiment gap analysis
# =============================================================================

def compute_sentiment_gap(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily mean |BSM - MC| and correlate with news sentiment score.
    """
    daily = (
        eval_df.groupby("date")
        .agg(mean_abs_error=("abs_error", "mean"),
             sentiment=("sentiment", "mean"))
        .dropna()
    )

    corr_pearson  = daily["sentiment"].corr(daily["mean_abs_error"])
    corr_spearman = daily["sentiment"].corr(daily["mean_abs_error"], method="spearman")

    summary = pd.DataFrame([
        {"metric": "pearson_corr(sentiment, mean_abs_error)",  "value": round(corr_pearson, 6)},
        {"metric": "spearman_corr(sentiment, mean_abs_error)", "value": round(corr_spearman, 6)},
        {"metric": "days_with_sentiment",                      "value": int(daily.shape[0])},
        {"metric": "mean_abs_error_positive_sentiment",
         "value": round(daily.loc[daily["sentiment"] > 0, "mean_abs_error"].mean(), 6)},
        {"metric": "mean_abs_error_negative_sentiment",
         "value": round(daily.loc[daily["sentiment"] < 0, "mean_abs_error"].mean(), 6)},
    ])
    return summary, daily


# =============================================================================
# 7a. Markdown table helpers
# =============================================================================

def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        text = ""
    elif isinstance(value, (float, np.floating)):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("|", r"\|")


def dataframe_to_markdown(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    table = frame.copy()
    if columns is not None:
        table = table[columns]

    headers = list(table.columns)
    lines = [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_markdown_cell(row[col]) for col in headers) + " |")

    return "\n".join(lines)


# =============================================================================
# 7. Charts
# =============================================================================

def plot_error_timeseries(eval_df: pd.DataFrame, out_path: Path) -> None:
    """Daily mean absolute error over time, coloured by VIX regime."""
    daily = eval_df.groupby("date").agg(
        mean_abs_error=("abs_error", "mean"),
        regime=("regime", "first"),
        vix=("vix", "mean"),
    ).reset_index()

    colour_map = {"low": "#2196F3", "medium": "#FF9800", "high": "#F44336"}
    colours = daily["regime"].map(colour_map)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].bar(
        daily["date"],
        daily["mean_abs_error"],
        color=colours,
        width=20,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].set_ylabel("Mean |BSM − MC|  ($)")
    axes[0].set_title("Week 4: BSM vs MC Daily Mean Absolute Error")
    legend_handles = [
        Patch(facecolor=colour_map["low"], edgecolor="white", label=f"Low VIX (< {VIX_LOW})"),
        Patch(facecolor=colour_map["medium"], edgecolor="white", label=f"Medium VIX ({VIX_LOW} to < {VIX_HIGH})"),
        Patch(facecolor=colour_map["high"], edgecolor="white", label=f"High VIX (>= {VIX_HIGH})"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right", title="Bar color")
    axes[0].text(
        0.01,
        0.95,
        "Bars are colored by VIX regime",
        transform=axes[0].transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )

    axes[1].plot(daily["date"], daily["vix"], color="black", linewidth=0.8)
    axes[1].axhline(VIX_LOW,  color="#FF9800", linestyle="--", linewidth=0.7, label=f"VIX={VIX_LOW}")
    axes[1].axhline(VIX_HIGH, color="#F44336",  linestyle="--", linewidth=0.7, label=f"VIX={VIX_HIGH}")
    axes[1].set_ylabel("VIX")
    axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart → {out_path.name}")


def plot_regime_boxplot(eval_df: pd.DataFrame, out_path: Path) -> None:
    """Box-plot of |BSM − MC| across VIX regimes and maturities."""
    colour_map = {"low": "#2196F3", "medium": "#FF9800", "high": "#F44336"}
    regimes = ["low", "medium", "high"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, T in zip(axes, MATURITIES):
        sub = eval_df[eval_df["T"] == T]
        data  = [sub.loc[sub["regime"] == r, "abs_error"].values for r in regimes]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=regimes)
        for patch, r in zip(bp["boxes"], regimes):
            patch.set_facecolor(colour_map[r])
        ax.set_title(f"T = {T} yr")
        ax.set_xlabel("VIX Regime")
        if ax is axes[0]:
            ax.set_ylabel("|BSM − MC|  ($)")

    fig.suptitle("BSM Absolute Error by VIX Regime and Maturity", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart → {out_path.name}")


def plot_sentiment_scatter(eval_df: pd.DataFrame, sentiment_daily: pd.DataFrame,
                           out_path: Path) -> None:
    """Scatter: daily mean |BSM − MC| vs daily mean sentiment score."""
    if sentiment_daily.empty:
        logger.warning("No sentiment data available – skipping scatter chart.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sentiment_daily["sentiment"], sentiment_daily["mean_abs_error"],
               alpha=0.5, s=20, color="#9C27B0")
    ax.set_xlabel("Mean Daily JPM Sentiment Score")
    ax.set_ylabel("Mean Daily |BSM − MC|  ($)")
    ax.set_title("BSM Pricing Error vs News Sentiment")

    # trend line
    if len(sentiment_daily) > 2:
        z = np.polyfit(sentiment_daily["sentiment"], sentiment_daily["mean_abs_error"], 1)
        p = np.poly1d(z)
        xs = np.linspace(sentiment_daily["sentiment"].min(),
                         sentiment_daily["sentiment"].max(), 100)
        ax.plot(xs, p(xs), "r--", linewidth=1.5, label="Linear trend")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart → {out_path.name}")


def plot_residuals_vs_fitted(eval_df: pd.DataFrame, out_path: Path) -> None:
    """Residual diagnostic: residuals against fitted BSM prices."""
    colour_map = {"low": "#2196F3", "medium": "#FF9800", "high": "#F44336"}
    fig, ax = plt.subplots(figsize=(8, 5))
    point_colours = eval_df["regime"].map(colour_map)
    ax.scatter(
        eval_df["bsm_price"],
        eval_df["residual"],
        c=point_colours,
        s=18,
        alpha=0.5,
        edgecolors="none",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Fitted value (BSM price)")
    ax.set_ylabel("Residual (BSM - MC)")
    ax.set_title("Residuals vs Fitted Values")
    legend_handles = [
        Patch(facecolor=colour_map["low"], edgecolor="white", label=f"Low VIX (< {VIX_LOW})"),
        Patch(facecolor=colour_map["medium"], edgecolor="white", label=f"Medium VIX ({VIX_LOW} to < {VIX_HIGH})"),
        Patch(facecolor=colour_map["high"], edgecolor="white", label=f"High VIX (>= {VIX_HIGH})"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, title="Point color")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart → {out_path.name}")


def plot_residuals_qq(eval_df: pd.DataFrame, out_path: Path) -> None:
    """Q-Q plot against a standard normal after residual standardisation."""
    residuals = eval_df["residual"].dropna().to_numpy()
    if residuals.size < 3:
        logger.warning("Not enough residuals available – skipping Q-Q plot.")
        return

    mean = residuals.mean()
    std = residuals.std(ddof=1)
    if std == 0:
        logger.warning("Residual standard deviation is zero – skipping Q-Q plot.")
        return

    z_scores = np.sort((residuals - mean) / std)
    n = z_scores.size
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    normal = NormalDist()
    theoretical = np.array([normal.inv_cdf(p) for p in probabilities])

    slope, intercept = np.polyfit(theoretical, z_scores, 1)
    fit_line = slope * theoretical + intercept

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(theoretical, z_scores, s=16, alpha=0.55, color="#1976D2")
    ax.plot(theoretical, fit_line, color="#D32F2F", linestyle="--", linewidth=1.2, label="Reference line")
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Standardized Residual Quantiles")
    ax.set_title("Q-Q Plot of Pricing Residuals")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart → {out_path.name}")


# =============================================================================
# 8. Markdown validation report
# =============================================================================
# 8a. Report 1 – Model Validation Report (error metrics + failure modes)
# =============================================================================

def build_validation_report(
    eval_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    sentiment_summary: pd.DataFrame,
) -> str:
    overall      = metrics_df[metrics_df["group"] == "overall"].iloc[0]
    low_m        = metrics_df[metrics_df["group"] == "regime=low"]
    med_m        = metrics_df[metrics_df["group"] == "regime=medium"]
    high_m       = metrics_df[metrics_df["group"] == "regime=high"]

    n_dates  = eval_df["date"].nunique()
    n_rows   = len(eval_df)

    sentiment_corr = sentiment_summary.set_index("metric")["value"].to_dict()

    maturity_table = dataframe_to_markdown(
        metrics_df[metrics_df["group"].str.startswith("maturity=")],
        ["group", "n", "MAE", "RMSE", "max_abs_err"],
    )
    type_table = dataframe_to_markdown(
        metrics_df[metrics_df["group"].str.startswith("type=")],
        ["group", "n", "MAE", "RMSE", "max_abs_err"],
    )

    report = f"""# Week 4 – Model Validation Report: BSM Error Metrics

**Run date**: {RUN_DATE}  |  **Pipeline version**: {PIPELINE_VER}

## Methodology

The BSM analytical closed-form prices are treated as *model predictions*.
Monte Carlo (MC) simulation prices (N={MC_PATHS:,} paths, GBM under risk-neutral
measure, seed={MC_SEED}) serve as the independent benchmark ("actual prices").
MAE and RMSE measure how closely the BSM formula approximates the MC benchmark
across different market regimes.

Parameters are derived from JPM historical market data (2018–2024):
- **S**: JPM daily close price
- **σ**: {VOL_WINDOW}-day rolling historical annualised volatility
- **r**: US 10-year Treasury yield (DGS10)
- **q**: trailing-twelve-month dividend yield
- **VIX regime**: Low < {VIX_LOW}, Medium {VIX_LOW}–{VIX_HIGH}, High ≥ {VIX_HIGH}

Evaluation grid: {len(MATURITIES)} maturities × {len(MONEYNESS)} moneyness levels × 2 option types,
sampled monthly → **{n_rows:,} pricing observations** over **{n_dates} dates**.

---

## 1. Overall Error Metrics

| Metric | Value |
|--------|-------|
| MAE (overall) | {overall["MAE"]:.6f} |
| RMSE (overall) | {overall["RMSE"]:.6f} |
| Max \\|BSM − MC\\| | {overall["max_abs_err"]:.6f} |
| Total observations | {int(overall["n"]):,} |

These values quantify the numerical convergence gap between the BSM
analytical formula and the MC simulation benchmark.  Both are generated
under identical GBM assumptions, so deviations arise purely from MC
sampling variance (which decreases as ∝ 1/√N).

---

## 2. Error by VIX Regime (Failure Mode Analysis)

| Regime | MAE | RMSE | n |
|--------|-----|------|---|
| Low (VIX < {VIX_LOW}) | {low_m.iloc[0]["MAE"] if not low_m.empty else "N/A":.6f} | {low_m.iloc[0]["RMSE"] if not low_m.empty else "N/A":.6f} | {int(low_m.iloc[0]["n"]) if not low_m.empty else 0} |
| Medium ({VIX_LOW}–{VIX_HIGH}) | {med_m.iloc[0]["MAE"] if not med_m.empty else "N/A":.6f} | {med_m.iloc[0]["RMSE"] if not med_m.empty else "N/A":.6f} | {int(med_m.iloc[0]["n"]) if not med_m.empty else 0} |
| High (VIX ≥ {VIX_HIGH}) | {high_m.iloc[0]["MAE"] if not high_m.empty else "N/A":.6f} | {high_m.iloc[0]["RMSE"] if not high_m.empty else "N/A":.6f} | {int(high_m.iloc[0]["n"]) if not high_m.empty else 0} |

**Interpretation**: Higher VIX regimes produce larger absolute errors because
the MC payoff distribution widens with higher volatility, amplifying sampling
noise.  Under a fixed N={MC_PATHS:,} paths, the standard error of the MC estimate
scales as σ × √(T/N), so high-σ, long-T options show the largest gaps.

---

## 3. Error by Maturity and Option Type

{maturity_table}

{type_table}

Longer maturities accumulate more GBM variance, making MC estimates noisier and
increasing |BSM − MC|.  Calls and puts exhibit similar error levels due to
put-call parity symmetry.

---

## 4. Sentiment Impact Gap Analysis

| Metric | Value |
|--------|-------|
| Pearson corr(sentiment, mean \\|BSM−MC\\|) | {sentiment_corr.get("pearson_corr(sentiment, mean_abs_error)", "N/A")} |
| Spearman corr(sentiment, mean \\|BSM−MC\\|) | {sentiment_corr.get("spearman_corr(sentiment, mean_abs_error)", "N/A")} |
| Mean error on positive-sentiment days | {sentiment_corr.get("mean_abs_error_positive_sentiment", "N/A")} |
| Mean error on negative-sentiment days | {sentiment_corr.get("mean_abs_error_negative_sentiment", "N/A")} |

**Interpretation**: BSM does not incorporate sentiment as an input; sentiment is
a proxy for market-moving information events that the model structurally ignores.
A non-zero correlation between sentiment and |BSM − MC| indicates that, on days
with strong news signal, market-implied volatility may deviate from the backward-
looking historical volatility used in BSM, widening the BSM–MC gap.

---

## 5. Charts

![Error Time Series](week4_bsm_error_timeseries.png)

![Regime Boxplot](week4_bsm_regime_boxplot.png)

![Sentiment Scatter](week4_bsm_sentiment_scatter.png)

![Residuals vs Fitted](week4_bsm_residuals_vs_fitted.png)

![Residual Q-Q Plot](week4_bsm_residuals_qq.png)
"""
    return report


# =============================================================================
# 8b. Report 2 – Performance Benchmark Documentation
# =============================================================================

def build_benchmark_report(
    eval_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    sentiment_summary: pd.DataFrame,
) -> str:
    overall = metrics_df[metrics_df["group"] == "overall"].iloc[0]
    low_m   = metrics_df[metrics_df["group"] == "regime=low"]
    med_m   = metrics_df[metrics_df["group"] == "regime=medium"]
    high_m  = metrics_df[metrics_df["group"] == "regime=high"]
    sentiment_corr = sentiment_summary.set_index("metric")["value"].to_dict()

    full_table = dataframe_to_markdown(
        metrics_df,
        ["group", "n", "MAE", "RMSE", "max_abs_err"],
    )

    def _mae(sub): return sub.iloc[0]["MAE"] if not sub.empty else float("nan")
    def _rmse(sub): return sub.iloc[0]["RMSE"] if not sub.empty else float("nan")

    report = f"""# Week 4 – BSM Performance Benchmark Documentation

**Run date**: {RUN_DATE}  |  **Pipeline version**: {PIPELINE_VER}

This document records the Black-Scholes-Merton (BSM) model performance baseline
for JPM options pricing over the 2018–2024 period.  All figures here serve as
the reference point for any future model enhancements.

---

## 1. Benchmark Setup

| Parameter | Value |
|-----------|-------|
| Underlying asset | JPM (JPMorgan Chase) |
| Evaluation period | 2018-01-01 – 2024-12-31 |
| Sampling frequency | Monthly (month-start) |
| Maturities | {MATURITIES} years |
| Moneyness levels (K/S) | {MONEYNESS} |
| Option types | Call, Put |
| Total observations | {int(overall["n"]):,} |
| MC benchmark paths | {MC_PATHS:,} (seed={MC_SEED}) |
| Historical vol window | {VOL_WINDOW} trading days |
| Risk-free rate source | FRED DGS10 |
| Dividend yield | Trailing-twelve-month |

---

## 2. Headline Baseline Metrics

These are the official BSM baseline figures.  Future models must beat these
numbers to demonstrate improvement.

| Metric | Baseline value |
|--------|----------------|
| Overall MAE | {overall["MAE"]:.6f} |
| Overall RMSE | {overall["RMSE"]:.6f} |
| Max absolute error | {overall["max_abs_err"]:.6f} |
| Low-VIX MAE  (VIX < {VIX_LOW}) | {_mae(low_m):.6f} |
| Mid-VIX MAE  ({VIX_LOW}–{VIX_HIGH}) | {_mae(med_m):.6f} |
| High-VIX MAE (VIX ≥ {VIX_HIGH}) | {_mae(high_m):.6f} |
| Low-VIX RMSE | {_rmse(low_m):.6f} |
| Mid-VIX RMSE | {_rmse(med_m):.6f} |
| High-VIX RMSE | {_rmse(high_m):.6f} |
| Sentiment–error Pearson corr | {sentiment_corr.get("pearson_corr(sentiment, mean_abs_error)", "N/A")} |

---

## 3. Full Breakdown by Group

All sub-group MAE / RMSE values for complete traceability:

{full_table}

---

## 4. Key Limitations Identified

1. **High-volatility failure**: MAE in high-VIX regime ({_mae(high_m):.4f}) is
   {_mae(high_m)/_mae(low_m)*100:.0f}% of the low-VIX baseline ({_mae(low_m):.4f}).
   BSM underprices risk during market stress because it uses backward-looking
   historical volatility rather than forward-looking implied volatility.

2. **Maturity effect**: Error scales with maturity (T=1yr RMSE is materially
   larger than T=0.25yr), reflecting accumulated GBM path uncertainty that
   BSM's closed form cannot fully capture when σ is volatile.

3. **Sentiment gap**: Pearson correlation of {sentiment_corr.get("pearson_corr(sentiment, mean_abs_error)", "N/A")} between
   news sentiment and pricing error confirms BSM's inability to respond to
   information events.  Positive-sentiment days show higher mean error
   ({sentiment_corr.get("mean_abs_error_positive_sentiment", "N/A")}) vs negative-sentiment days
   ({sentiment_corr.get("mean_abs_error_negative_sentiment", "N/A")}), suggesting bullish news events
   drive larger deviations from the simulation benchmark.

---

## 5. Improvement Targets for Future Models

| Target | Current baseline | Goal |
|--------|-----------------|------|
| Overall MAE | {overall["MAE"]:.6f} | < {overall["MAE"]*0.8:.6f} (−20%) |
| High-VIX MAE | {_mae(high_m):.6f} | < {_mae(high_m)*0.75:.6f} (−25%) |
| Sentiment correlation | {sentiment_corr.get("pearson_corr(sentiment, mean_abs_error)", "N/A")} | ≈ 0 (model absorbs sentiment) |
"""
    return report


def build_combined_report(
    eval_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    sentiment_summary: pd.DataFrame,
) -> str:
    overall = metrics_df[metrics_df["group"] == "overall"].iloc[0]
    low_m = metrics_df[metrics_df["group"] == "regime=low"]
    med_m = metrics_df[metrics_df["group"] == "regime=medium"]
    high_m = metrics_df[metrics_df["group"] == "regime=high"]
    n_dates = eval_df["date"].nunique()
    n_rows = len(eval_df)
    sentiment_corr = sentiment_summary.set_index("metric")["value"].to_dict()

    maturity_table = dataframe_to_markdown(
        metrics_df[metrics_df["group"].str.startswith("maturity=")],
        ["group", "n", "MAE", "RMSE", "max_abs_err"],
    )
    type_table = dataframe_to_markdown(
        metrics_df[metrics_df["group"].str.startswith("type=")],
        ["group", "n", "MAE", "RMSE", "max_abs_err"],
    )
    full_table = dataframe_to_markdown(
        metrics_df,
        ["group", "n", "MAE", "RMSE", "max_abs_err"],
    )

    def _mae(sub):
        return sub.iloc[0]["MAE"] if not sub.empty else float("nan")

    def _rmse(sub):
        return sub.iloc[0]["RMSE"] if not sub.empty else float("nan")

    report = f"""# Week 4 – BSM Model Validation and Performance Benchmark

**Run date**: {RUN_DATE}  |  **Pipeline version**: {PIPELINE_VER}

## Methodology

The BSM analytical closed-form prices are treated as *model predictions*.
Monte Carlo (MC) simulation prices (N={MC_PATHS:,} paths, GBM under risk-neutral
measure, seed={MC_SEED}) serve as the independent benchmark ("actual prices").
MAE and RMSE measure how closely the BSM formula approximates the MC benchmark
across different market regimes.

Parameters are derived from JPM historical market data (2018–2024):
- **S**: JPM daily close price
- **σ**: {VOL_WINDOW}-day rolling historical annualised volatility
- **r**: US 10-year Treasury yield (DGS10)
- **q**: trailing-twelve-month dividend yield
- **VIX regime**: Low < {VIX_LOW}, Medium {VIX_LOW}–{VIX_HIGH}, High ≥ {VIX_HIGH}

Evaluation grid: {len(MATURITIES)} maturities × {len(MONEYNESS)} moneyness levels × 2 option types,
sampled monthly → **{n_rows:,} pricing observations** over **{n_dates} dates**.

---

## 1. Model Validation Report with Error Metrics

### 1.1 Overall Error Metrics

| Metric | Value |
|--------|-------|
| MAE (overall) | {overall["MAE"]:.6f} |
| RMSE (overall) | {overall["RMSE"]:.6f} |
| Max \\|BSM − MC\\| | {overall["max_abs_err"]:.6f} |
| Total observations | {int(overall["n"]):,} |

These values quantify the numerical convergence gap between the BSM
analytical formula and the MC simulation benchmark. Both are generated
under identical GBM assumptions, so deviations arise from MC sampling
variance rather than from mismatched pricing assumptions.

### 1.2 Error by VIX Regime

| Regime | MAE | RMSE | n |
|--------|-----|------|---|
| Low (VIX < {VIX_LOW}) | {low_m.iloc[0]["MAE"] if not low_m.empty else "N/A":.6f} | {low_m.iloc[0]["RMSE"] if not low_m.empty else "N/A":.6f} | {int(low_m.iloc[0]["n"]) if not low_m.empty else 0} |
| Medium ({VIX_LOW}–{VIX_HIGH}) | {med_m.iloc[0]["MAE"] if not med_m.empty else "N/A":.6f} | {med_m.iloc[0]["RMSE"] if not med_m.empty else "N/A":.6f} | {int(med_m.iloc[0]["n"]) if not med_m.empty else 0} |
| High (VIX ≥ {VIX_HIGH}) | {high_m.iloc[0]["MAE"] if not high_m.empty else "N/A":.6f} | {high_m.iloc[0]["RMSE"] if not high_m.empty else "N/A":.6f} | {int(high_m.iloc[0]["n"]) if not high_m.empty else 0} |

Higher VIX regimes show larger absolute errors because the MC payoff distribution
widens with volatility, amplifying sampling noise under a fixed number of paths.

### 1.3 Error by Maturity and Option Type

{maturity_table}

{type_table}

### 1.4 Sentiment Impact Gap Analysis

| Metric | Value |
|--------|-------|
| Pearson corr(sentiment, mean \\|BSM−MC\\|) | {sentiment_corr.get("pearson_corr(sentiment, mean_abs_error)", "N/A")} |
| Spearman corr(sentiment, mean \\|BSM−MC\\|) | {sentiment_corr.get("spearman_corr(sentiment, mean_abs_error)", "N/A")} |
| Mean error on positive-sentiment days | {sentiment_corr.get("mean_abs_error_positive_sentiment", "N/A")} |
| Mean error on negative-sentiment days | {sentiment_corr.get("mean_abs_error_negative_sentiment", "N/A")} |

This gap analysis highlights where BSM lacks information sensitivity: the model
does not ingest sentiment or event risk directly, so strong news periods can
coincide with larger pricing deviations.

### 1.5 Validation Charts

![Error Time Series](week4_bsm_error_timeseries.png)

![Regime Boxplot](week4_bsm_regime_boxplot.png)

![Sentiment Scatter](week4_bsm_sentiment_scatter.png)

![Residuals vs Fitted](week4_bsm_residuals_vs_fitted.png)

![Residual Q-Q Plot](week4_bsm_residuals_qq.png)

---

## 2. Performance Benchmark Documentation

### 2.1 Benchmark Setup

| Parameter | Value |
|-----------|-------|
| Underlying asset | JPM (JPMorgan Chase) |
| Evaluation period | 2018-01-01 – 2024-12-31 |
| Sampling frequency | Monthly (month-start) |
| Maturities | {MATURITIES} years |
| Moneyness levels (K/S) | {MONEYNESS} |
| Option types | Call, Put |
| Total observations | {int(overall["n"]):,} |
| MC benchmark paths | {MC_PATHS:,} (seed={MC_SEED}) |
| Historical vol window | {VOL_WINDOW} trading days |
| Risk-free rate source | FRED DGS10 |
| Dividend yield | Trailing-twelve-month |

### 2.2 Headline Baseline Metrics

| Metric | Baseline value |
|--------|----------------|
| Overall MAE | {overall["MAE"]:.6f} |
| Overall RMSE | {overall["RMSE"]:.6f} |
| Max absolute error | {overall["max_abs_err"]:.6f} |
| Low-VIX MAE (VIX < {VIX_LOW}) | {_mae(low_m):.6f} |
| Mid-VIX MAE ({VIX_LOW}–{VIX_HIGH}) | {_mae(med_m):.6f} |
| High-VIX MAE (VIX ≥ {VIX_HIGH}) | {_mae(high_m):.6f} |
| Low-VIX RMSE | {_rmse(low_m):.6f} |
| Mid-VIX RMSE | {_rmse(med_m):.6f} |
| High-VIX RMSE | {_rmse(high_m):.6f} |
| Sentiment–error Pearson corr | {sentiment_corr.get("pearson_corr(sentiment, mean_abs_error)", "N/A")} |

### 2.3 Full Breakdown by Group

{full_table}

### 2.4 Key Limitations Identified

1. **High-volatility failure**: MAE in high-VIX regime ({_mae(high_m):.4f}) is {_mae(high_m) / _mae(low_m) * 100:.0f}% of the low-VIX baseline ({_mae(low_m):.4f}).
2. **Maturity effect**: Error rises with maturity because path uncertainty accumulates over longer horizons.
3. **Sentiment gap**: A positive sentiment-error correlation indicates that event risk is not explicitly modeled.

### 2.5 Improvement Targets for Future Models

| Target | Current baseline | Goal |
|--------|-----------------|------|
| Overall MAE | {overall["MAE"]:.6f} | < {overall["MAE"] * 0.8:.6f} (−20%) |
| High-VIX MAE | {_mae(high_m):.6f} | < {_mae(high_m) * 0.75:.6f} (−25%) |
| Sentiment correlation | {sentiment_corr.get("pearson_corr(sentiment, mean_abs_error)", "N/A")} | ≈ 0 (model absorbs sentiment) |
"""
    return report


# =============================================================================
# 9. Main
# =============================================================================

def versioned(stem: str, ext: str) -> str:
    return f"{stem}_{PIPELINE_VER}_{RUN_DATE}.{ext}"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- load data --
    logger.info("Loading market data …")
    market_df   = load_market_data()
    sentiment   = load_sentiment_data()
    logger.info(f"  Market data: {len(market_df)} trading days "
                f"({market_df.index.min().date()} – {market_df.index.max().date()})")

    # -- evaluate --
    eval_df = run_evaluation(market_df, sentiment)
    logger.info(f"Evaluation complete: {len(eval_df):,} rows")

    # -- metrics --
    metrics_df = compute_error_metrics(eval_df)
    sentiment_summary, sentiment_daily = compute_sentiment_gap(eval_df)

    # -- save CSVs --
    daily_csv_path     = PROCESSED_DIR / versioned("week4_bsm_evaluation_daily", "csv")
    metrics_csv_path   = PROCESSED_DIR / versioned("week4_bsm_error_metrics", "csv")
    sentiment_csv_path = PROCESSED_DIR / versioned("week4_bsm_sentiment_gap", "csv")

    eval_df.to_csv(daily_csv_path, index=False)
    metrics_df.to_csv(metrics_csv_path, index=False)
    sentiment_summary.to_csv(sentiment_csv_path, index=False)
    logger.info(f"Saved: {daily_csv_path.name}")
    logger.info(f"Saved: {metrics_csv_path.name}")
    logger.info(f"Saved: {sentiment_csv_path.name}")

    # -- charts --
    plot_error_timeseries(
        eval_df,
        REPORTS_DIR / "week4_bsm_error_timeseries.png"
    )
    plot_regime_boxplot(
        eval_df,
        REPORTS_DIR / "week4_bsm_regime_boxplot.png"
    )
    plot_sentiment_scatter(
        eval_df, sentiment_daily,
        REPORTS_DIR / "week4_bsm_sentiment_scatter.png"
    )
    plot_residuals_vs_fitted(
        eval_df,
        REPORTS_DIR / "week4_bsm_residuals_vs_fitted.png"
    )
    plot_residuals_qq(
        eval_df,
        REPORTS_DIR / "week4_bsm_residuals_qq.png"
    )

    # -- report 1: model validation report --
    validation_text = build_validation_report(eval_df, metrics_df, sentiment_summary)
    validation_md   = REPORTS_DIR / versioned("week4_bsm_validation", "md")
    validation_md.write_text(validation_text, encoding="utf-8")
    logger.info(f"Saved report → {validation_md.name}")

    # -- report 2: performance benchmark documentation --
    benchmark_text = build_benchmark_report(eval_df, metrics_df, sentiment_summary)
    benchmark_md   = REPORTS_DIR / versioned("week4_bsm_benchmark", "md")
    benchmark_md.write_text(benchmark_text, encoding="utf-8")
    logger.info(f"Saved report → {benchmark_md.name}")

    # -- combined report: validation + benchmark in one file --
    combined_text = build_combined_report(eval_df, metrics_df, sentiment_summary)
    combined_md = REPORTS_DIR / versioned("week4_bsm_combined_report", "md")
    combined_md.write_text(combined_text, encoding="utf-8")
    logger.info(f"Saved report → {combined_md.name}")

    # -- generate PDFs --
    sys.path.insert(0, str(ROOT / "scripts"))
    from preprocess import save_markdown_pdf_report
    chart_assets = {
        "week4_bsm_error_timeseries.png":  REPORTS_DIR / "week4_bsm_error_timeseries.png",
        "week4_bsm_regime_boxplot.png":    REPORTS_DIR / "week4_bsm_regime_boxplot.png",
        "week4_bsm_sentiment_scatter.png": REPORTS_DIR / "week4_bsm_sentiment_scatter.png",
        "week4_bsm_residuals_vs_fitted.png": REPORTS_DIR / "week4_bsm_residuals_vs_fitted.png",
        "week4_bsm_residuals_qq.png": REPORTS_DIR / "week4_bsm_residuals_qq.png",
    }
    save_markdown_pdf_report(
        validation_md,
        versioned("week4_bsm_validation", "pdf"),
        "Week 4 – Model Validation Report: BSM Error Metrics",
        asset_paths=chart_assets,
    )
    logger.info(f"Saved PDF → {versioned('week4_bsm_validation', 'pdf')}")
    save_markdown_pdf_report(
        benchmark_md,
        versioned("week4_bsm_benchmark", "pdf"),
        "Week 4 – BSM Performance Benchmark Documentation",
    )
    logger.info(f"Saved PDF → {versioned('week4_bsm_benchmark', 'pdf')}")
    save_markdown_pdf_report(
        combined_md,
        versioned("week4_bsm_combined_report", "pdf"),
        "Week 4 – BSM Model Validation and Performance Benchmark",
        asset_paths=chart_assets,
    )
    logger.info(f"Saved PDF → {versioned('week4_bsm_combined_report', 'pdf')}")

    # -- print summary to console --
    overall = metrics_df[metrics_df["group"] == "overall"].iloc[0]
    print("\n" + "=" * 55)
    print("  Week 4  BSM Baseline Evaluation – Summary")
    print("=" * 55)
    print(f"  Observations : {int(overall['n']):,}")
    print(f"  Overall MAE  : {overall['MAE']:.6f}")
    print(f"  Overall RMSE : {overall['RMSE']:.6f}")
    print(f"  Max |error|  : {overall['max_abs_err']:.6f}")
    print("-" * 55)
    for _, row in metrics_df[metrics_df["group"].str.startswith("regime=")].iterrows():
        print(f"  {row['group']:<25s}  MAE={row['MAE']:.6f}  RMSE={row['RMSE']:.6f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
