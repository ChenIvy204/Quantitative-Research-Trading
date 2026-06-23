from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from week5_ml_models import (  # noqa: E402
    HAS_SHAP,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    build_features,
    chooser_price_mc_scalar,
    chooser_price_scalar,
    export_markdown_pdf,
    get_pricing_feature_columns,
    get_pricing_model_columns,
    load_market_data,
    normalize_pdf_compatibility,
    shap,
)

PIPELINE_VER = "v1.0"
OUTPUT_WEEK = "week7"
TODAY_STAMP = datetime.now().strftime("%Y%m%d")

REPORT_STEM = f"{OUTPUT_WEEK}_analysis_{PIPELINE_VER}"
SENSITIVITY_STEM = f"{OUTPUT_WEEK}_sensitivity_{PIPELINE_VER}_{TODAY_STAMP}"
SCENARIO_STEM = f"{OUTPUT_WEEK}_scenario_tests_{PIPELINE_VER}_{TODAY_STAMP}"
SHAP_STEM = f"{OUTPUT_WEEK}_shap_summary_{PIPELINE_VER}_{TODAY_STAMP}"

DEFAULT_MODEL_PRIORITY = (
    "week6_approach2_neuralnetwork_v1.0.joblib",
    "week6_approach2_xgboost_v1.0.joblib",
    "week6_approach2_linearregression_v1.0.joblib",
    "week6_approach1_xgboost_v1.0.joblib",
)


def _latest_processed_file(pattern: str) -> Path | None:
    candidates = sorted(PROCESSED_DIR.glob(pattern))
    return candidates[-1] if candidates else None

RAW_DATA_FILES = (
    "yahoo_jpm_2018_2024.csv",
    "fred_DGS10_2018_2024.csv",
    "fred_VIXCLS_2018_2024.csv",
    "jpm_dividends_2018_2024.csv",
    "alphavantage_news_jpm_2018_2024.csv",
)
MARKET_DATA_MAX_AGE_HOURS = 24.0


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _render_markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows_"

    display = frame.copy()
    if max_rows is not None and len(display) > max_rows:
        display = display.head(max_rows).copy()

    columns = list(display.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")

    if max_rows is not None and len(frame) > max_rows:
        rows.append(f"| _{len(frame) - max_rows} more rows omitted_ |")

    return "\n".join([header, separator, *rows])


def available_model_artifacts(approach: str = "approach2") -> list[Path]:
    return sorted(MODELS_DIR.glob(f"week6_{approach}_*_v1.0.joblib"))


def resolve_model_artifact(model_name: str | None = None, approach: str = "approach2") -> Path:
    candidates = available_model_artifacts(approach)
    if not candidates:
        raise FileNotFoundError(f"No Week 6 model artifacts found in {MODELS_DIR}")

    if model_name:
        normalized = model_name.strip().lower()
        for candidate in candidates:
            if normalized in candidate.stem.lower():
                return candidate

    for preferred in DEFAULT_MODEL_PRIORITY:
        for candidate in candidates:
            if candidate.name == preferred or preferred in candidate.name:
                return candidate

    return candidates[0]


def load_model_bundle(model_name: str | None = None, approach: str = "approach2") -> tuple[Path, dict[str, object], object]:
    artifact_path = resolve_model_artifact(model_name=model_name, approach=approach)
    payload = joblib.load(artifact_path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
    else:
        model = payload
        payload = {"model": model}
    return artifact_path, payload, model


def load_pricing_performance_summary() -> pd.DataFrame:
    summary_path = _latest_processed_file("week6_pricing_results_v1.0*.csv")
    if summary_path is None:
        return pd.DataFrame(columns=["model", "mae", "rmse", "r2", "inference_time_ms"])

    frame = pd.read_csv(summary_path)
    for column in ("mae", "rmse", "r2", "inference_time_ms"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["mae", "rmse"], ascending=[True, True], na_position="last").reset_index(drop=True)


def load_bsm_error_summary() -> pd.DataFrame:
    summary_path = _latest_processed_file("week4_bsm_error_metrics_v1.0_*.csv")
    if summary_path is None:
        return pd.DataFrame(columns=["group", "n", "ME", "MAE", "RMSE", "t_stat", "p_val_ME", "max_abs_err"])

    frame = pd.read_csv(summary_path)
    for column in ("n", "ME", "MAE", "RMSE", "t_stat", "max_abs_err"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def best_pricing_model_name(performance_df: pd.DataFrame | None = None) -> str | None:
    frame = performance_df if performance_df is not None else load_pricing_performance_summary()
    if frame.empty or "model" not in frame.columns:
        return None
    non_baseline = frame[~frame["model"].astype(str).str.contains("bsm", case=False, na=False)].copy()
    if non_baseline.empty:
        non_baseline = frame.copy()
    if "mae" in non_baseline.columns:
        best_row = non_baseline.sort_values(["mae", "rmse"], ascending=[True, True], na_position="last").iloc[0]
    else:
        best_row = non_baseline.iloc[0]
    return str(best_row["model"])


def _match_vol_to_maturity(t_years: float) -> str:
    t_days = t_years * 252.0
    if t_days < 7:
        return "hist_vol_5d"
    if t_days < 15:
        return "hist_vol_10d"
    if t_days < 30:
        return "hist_vol_20d"
    if t_days < 50:
        return "hist_vol_40d"
    return "hist_vol_60d"


def load_feature_frame() -> pd.DataFrame:
    refresh_market_data_if_stale()
    raw_df = load_market_data()
    feat_df = build_features(raw_df)
    required = [col for col in get_pricing_feature_columns() if col in feat_df.columns]
    required.extend(["close", "r", "q", "vix", "hist_vol_20d"])
    required = list(dict.fromkeys(required))
    frame = feat_df.dropna(subset=required).sort_index()
    if frame.empty:
        raise ValueError("No usable feature rows were produced from the raw market data.")
    return frame


def select_reference_row(feature_frame: pd.DataFrame, reference_date: str | None = None) -> pd.Series:
    if reference_date is None:
        return feature_frame.iloc[-1]

    target = pd.Timestamp(reference_date).normalize()
    available_dates = feature_frame.index.sort_values().unique()
    if target in available_dates:
        row = feature_frame.loc[target]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

    earlier = available_dates[available_dates <= target]
    if len(earlier) > 0:
        row = feature_frame.loc[earlier[-1]]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

    return feature_frame.iloc[0]


def build_pricing_input(
    base_row: pd.Series,
    *,
    contract_overrides: dict[str, float] | None = None,
    market_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    contract_overrides = contract_overrides or {}
    market_overrides = market_overrides or {}

    record = base_row.to_dict()
    record.update(market_overrides)

    close_price = float(record.get("close", record.get("S", 0.0)))
    default_moneyness = float(contract_overrides.get("moneyness", record.get("moneyness", 1.0)))
    contract_s = float(contract_overrides.get("S", record.get("S", close_price)))
    contract_k = float(contract_overrides.get("K", contract_s * default_moneyness))
    contract_moneyness = float(contract_overrides.get("moneyness", contract_k / contract_s if contract_s else default_moneyness))
    contract_t1 = float(contract_overrides.get("T1", record.get("T1", 0.25)))
    contract_t2 = float(contract_overrides.get("T2", record.get("T2", 0.5)))

    record.update(
        {
            "S": contract_s,
            "K": contract_k,
            "moneyness": contract_moneyness,
            "T1": contract_t1,
            "T2": contract_t2,
        }
    )

    feature_cols = get_pricing_model_columns(get_pricing_feature_columns())
    missing = [column for column in feature_cols if column not in record]
    if missing:
        raise KeyError(f"Missing required model inputs: {missing}")

    data = [[float(record[column]) for column in feature_cols]]
    return pd.DataFrame(data, columns=feature_cols)


def predict_chooser_price(
    model: object,
    base_row: pd.Series,
    *,
    contract_overrides: dict[str, float] | None = None,
    market_overrides: dict[str, float] | None = None,
) -> float:
    inputs = build_pricing_input(
        base_row,
        contract_overrides=contract_overrides,
        market_overrides=market_overrides,
    )
    prediction = model.predict(inputs)
    return float(np.asarray(prediction).reshape(-1)[0])


def build_price_trend_frame(
    feature_frame: pd.DataFrame,
    model: object,
    *,
    contract_overrides: dict[str, float] | None = None,
    lookback_days: int = 30,
    anchor_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    contract_overrides = contract_overrides or {}
    if feature_frame.empty:
        return pd.DataFrame(columns=["date", "bsm_price", "model_price", "price_ci_lower", "price_ci_upper", "close"])

    ordered = feature_frame.sort_index()
    if anchor_date is not None:
        anchor_ts = pd.Timestamp(anchor_date).normalize()
        ordered = ordered.loc[:anchor_ts]
    trend_frame = ordered.tail(max(2, lookback_days)).copy()
    rows: list[dict[str, object]] = []
    for date, row in trend_frame.iterrows():
        row_contract = dict(contract_overrides)
        close_price = float(row.get("close", row.get("S", 0.0)))
        row_contract["S"] = close_price
        row_contract["K"] = close_price * float(row_contract.get("moneyness", 1.0))
        row_contract.setdefault("T1", float(contract_overrides.get("T1", 0.25)))
        row_contract.setdefault("T2", float(contract_overrides.get("T2", 0.5)))

        refs = reference_quotes(row, contract_overrides=row_contract)
        model_price = predict_chooser_price(model, row, contract_overrides=row_contract)
        interval = estimate_price_interval(row, contract_overrides=row_contract, sigma=refs["sigma_reference"])
        rows.append(
            {
                "date": date,
                "close": close_price,
                "bsm_price": refs["closed_form_quote"],
                "mc_price": refs["mc_quote"],
                "model_price": model_price,
                "price_ci_lower": interval["lower"],
                "price_ci_upper": interval["upper"],
                "price_gap": model_price - refs["closed_form_quote"],
            }
        )

    return pd.DataFrame(rows)


def _chooser_price_from_sigma(
    base_row: pd.Series,
    contract_overrides: dict[str, float],
    sigma: float,
) -> float:
    contract_s = float(contract_overrides.get("S", base_row.get("close", base_row.get("S", 0.0))))
    contract_moneyness = float(contract_overrides.get("moneyness", 1.0))
    contract_k = float(contract_overrides.get("K", contract_s * contract_moneyness))
    contract_t1 = float(contract_overrides.get("T1", 0.25))
    contract_t2 = float(contract_overrides.get("T2", 0.5))
    if contract_t2 <= contract_t1:
        contract_t2 = contract_t1 + 0.01
    contract_r = float(contract_overrides.get("r", base_row.get("r", 0.0)))
    contract_q = float(contract_overrides.get("q", base_row.get("q", 0.0)))
    return float(chooser_price_scalar(contract_s, contract_k, contract_t1, contract_t2, contract_r, contract_q, sigma))


def compute_greeks(
    base_row: pd.Series,
    *,
    contract_overrides: dict[str, float] | None = None,
    sigma: float | None = None,
    rel_step: float = 0.01,
) -> pd.DataFrame:
    contract_overrides = contract_overrides or {}
    refs = reference_quotes(base_row, contract_overrides=contract_overrides)
    sigma = float(sigma if sigma is not None else refs["sigma_reference"])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.25

    contract_s = float(contract_overrides.get("S", base_row.get("close", base_row.get("S", 0.0))))
    contract_moneyness = float(contract_overrides.get("moneyness", 1.0))
    contract_k = float(contract_overrides.get("K", contract_s * contract_moneyness))
    contract_t1 = float(contract_overrides.get("T1", 0.25))
    contract_t2 = float(contract_overrides.get("T2", 0.5))
    if contract_t2 <= contract_t1:
        contract_t2 = contract_t1 + 0.01
    contract_r = float(contract_overrides.get("r", base_row.get("r", 0.0)))
    contract_q = float(contract_overrides.get("q", base_row.get("q", 0.0)))

    step_s = max(0.01, abs(contract_s) * rel_step)
    step_sigma = max(0.0001, abs(sigma) * rel_step)

    base_price = chooser_price_scalar(contract_s, contract_k, contract_t1, contract_t2, contract_r, contract_q, sigma)
    price_up_s = chooser_price_scalar(contract_s + step_s, contract_k, contract_t1, contract_t2, contract_r, contract_q, sigma)
    price_dn_s = chooser_price_scalar(max(1e-8, contract_s - step_s), contract_k, contract_t1, contract_t2, contract_r, contract_q, sigma)
    price_up_sigma = chooser_price_scalar(contract_s, contract_k, contract_t1, contract_t2, contract_r, contract_q, sigma + step_sigma)
    price_dn_sigma = chooser_price_scalar(contract_s, contract_k, contract_t1, contract_t2, contract_r, contract_q, max(1e-8, sigma - step_sigma))

    delta = (price_up_s - price_dn_s) / (2.0 * step_s)
    gamma = (price_up_s - 2.0 * base_price + price_dn_s) / (step_s ** 2)
    vega = (price_up_sigma - price_dn_sigma) / (2.0 * step_sigma)

    result = pd.DataFrame(
        [
            {
                "metric": "price",
                "value": float(base_price),
            },
            {
                "metric": "delta",
                "value": float(delta),
            },
            {
                "metric": "gamma",
                "value": float(gamma),
            },
            {
                "metric": "vega",
                "value": float(vega),
            },
        ]
    )
    return result


def estimate_price_interval(
    base_row: pd.Series,
    *,
    contract_overrides: dict[str, float] | None = None,
    sigma: float | None = None,
    sigma_uncertainty: float | None = None,
) -> dict[str, float]:
    contract_overrides = contract_overrides or {}
    refs = reference_quotes(base_row, contract_overrides=contract_overrides)
    sigma = float(sigma if sigma is not None else refs["sigma_reference"])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.25

    if sigma_uncertainty is None:
        vol_series = pd.Series([base_row.get("hist_vol_5d"), base_row.get("hist_vol_20d"), base_row.get("hist_vol_60d")], dtype="float64")
        sigma_uncertainty = float(vol_series.dropna().std()) if vol_series.dropna().size > 1 else max(0.02, sigma * 0.1)
    sigma_uncertainty = max(0.0001, float(sigma_uncertainty))

    greeks = compute_greeks(base_row, contract_overrides=contract_overrides, sigma=sigma)
    base_price = float(greeks.loc[greeks["metric"] == "price", "value"].iloc[0])
    vega = float(greeks.loc[greeks["metric"] == "vega", "value"].iloc[0])
    half_width = 1.96 * abs(vega) * sigma_uncertainty
    return {
        "lower": max(0.0, base_price - half_width),
        "upper": max(0.0, base_price + half_width),
        "sigma_uncertainty": sigma_uncertainty,
    }


def solve_implied_vol(
    target_price: float,
    base_row: pd.Series,
    *,
    contract_overrides: dict[str, float] | None = None,
    sigma_bounds: tuple[float, float] = (0.01, 2.0),
    max_iter: int = 80,
    tol: float = 1e-6,
) -> float:
    contract_overrides = contract_overrides or {}
    lo, hi = sigma_bounds
    lo = max(1e-4, float(lo))
    hi = max(lo + 1e-4, float(hi))

    f_lo = _chooser_price_from_sigma(base_row, contract_overrides, lo) - target_price
    f_hi = _chooser_price_from_sigma(base_row, contract_overrides, hi) - target_price
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi

    if f_lo * f_hi > 0:
        return float(reference_quotes(base_row, contract_overrides=contract_overrides)["sigma_reference"])

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _chooser_price_from_sigma(base_row, contract_overrides, mid) - target_price
        if abs(f_mid) <= tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return float(0.5 * (lo + hi))


def build_iv_surface(
    base_row: pd.Series,
    *,
    model_price: float,
    moneyness_grid: list[float] | None = None,
    maturity_grid: list[float] | None = None,
    base_contract_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    base_contract_overrides = dict(base_contract_overrides or {})
    moneyness_grid = moneyness_grid or [0.8, 0.9, 1.0, 1.1, 1.2]
    maturity_grid = maturity_grid or [0.25, 0.5, 0.75, 1.0, 1.25]

    rows: list[dict[str, float]] = []
    for moneyness in moneyness_grid:
        for maturity in maturity_grid:
            contract_overrides = dict(base_contract_overrides)
            close_price = float(base_row.get("close", base_row.get("S", 0.0)))
            contract_overrides["moneyness"] = float(moneyness)
            contract_overrides["S"] = close_price
            contract_overrides["K"] = close_price * float(moneyness)
            contract_overrides["T1"] = float(min(contract_overrides.get("T1", 0.25), maturity - 0.05))
            contract_overrides["T2"] = float(max(maturity, contract_overrides["T1"] + 0.05))
            implied_vol = solve_implied_vol(model_price, base_row, contract_overrides=contract_overrides)
            rows.append(
                {
                    "moneyness": float(moneyness),
                    "T2": float(maturity),
                    "implied_vol": implied_vol,
                }
            )

    return pd.DataFrame(rows)


def batch_price_contracts(
    model: object,
    base_row: pd.Series,
    contract_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, contract_row in contract_frame.iterrows():
        contract_overrides: dict[str, float] = {}
        market_overrides: dict[str, float] = {}
        for key, value in contract_row.items():
            if pd.isna(value):
                continue
            if key in {"S", "K", "moneyness", "T1", "T2"}:
                contract_overrides[key] = float(value)
            else:
                market_overrides[key] = float(value)

        if "S" not in contract_overrides:
            contract_overrides["S"] = float(base_row.get("close", base_row.get("S", 0.0)))
        if "moneyness" not in contract_overrides and "K" in contract_overrides and "S" in contract_overrides and contract_overrides["S"]:
            contract_overrides["moneyness"] = contract_overrides["K"] / contract_overrides["S"]
        if "K" not in contract_overrides:
            contract_overrides["K"] = contract_overrides["S"] * float(contract_overrides.get("moneyness", 1.0))
        if "T1" not in contract_overrides:
            contract_overrides["T1"] = 0.25
        if "T2" not in contract_overrides:
            contract_overrides["T2"] = 0.5

        refs = reference_quotes(base_row, contract_overrides=contract_overrides)
        greeks = compute_greeks(base_row, contract_overrides=contract_overrides, sigma=refs["sigma_reference"])
        interval = estimate_price_interval(base_row, contract_overrides=contract_overrides, sigma=refs["sigma_reference"])
        model_price = predict_chooser_price(model, base_row, contract_overrides=contract_overrides, market_overrides=market_overrides)
        implied_vol = solve_implied_vol(model_price, base_row, contract_overrides=contract_overrides)

        rec = {
            "row_id": index,
            **contract_overrides,
            **market_overrides,
            "model_price": model_price,
            "closed_form_quote": refs["closed_form_quote"],
            "mc_quote": refs["mc_quote"],
            "sigma_reference": refs["sigma_reference"],
            "implied_vol": implied_vol,
            "price_ci_lower": interval["lower"],
            "price_ci_upper": interval["upper"],
        }
        for _, greek_row in greeks.iterrows():
            rec[greek_row["metric"]] = greek_row["value"]
        rows.append(rec)

    return pd.DataFrame(rows)


def reference_quotes(
    base_row: pd.Series,
    *,
    contract_overrides: dict[str, float] | None = None,
    market_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    contract_overrides = contract_overrides or {}
    market_overrides = market_overrides or {}

    scenario_row = base_row.to_dict()
    scenario_row.update(market_overrides)

    contract_s = float(contract_overrides.get("S", scenario_row.get("close", scenario_row.get("S", 0.0))))
    contract_moneyness = float(contract_overrides.get("moneyness", 1.0))
    contract_k = float(contract_overrides.get("K", contract_s * contract_moneyness))
    contract_t1 = float(contract_overrides.get("T1", 0.25))
    contract_t2 = float(contract_overrides.get("T2", 0.5))
    if contract_t2 <= contract_t1:
        contract_t2 = contract_t1 + 0.01
    contract_r = float(contract_overrides.get("r", scenario_row.get("r", 0.0)))
    contract_q = float(contract_overrides.get("q", scenario_row.get("q", 0.0)))

    sigma_col = _match_vol_to_maturity(contract_t2)
    sigma = float(scenario_row.get(sigma_col, scenario_row.get("hist_vol_20d", np.nan)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(scenario_row.get("hist_vol_20d", 0.25))

    base_vix = float(base_row.get("vix", np.nan))
    scenario_vix = float(scenario_row.get("vix", np.nan))
    if np.isfinite(base_vix) and np.isfinite(scenario_vix) and base_vix > 0 and scenario_vix > 0:
        sigma *= float(np.clip(scenario_vix / base_vix, 0.5, 2.5))

    closed_form = chooser_price_scalar(contract_s, contract_k, contract_t1, contract_t2, contract_r, contract_q, sigma)
    mc_quote = chooser_price_mc_scalar(contract_s, contract_k, contract_t1, contract_t2, contract_r, contract_q, sigma)
    return {
        "sigma_reference": sigma,
        "closed_form_quote": float(closed_form),
        "mc_quote": float(mc_quote),
    }


def build_sensitivity_grid(base_row: pd.Series) -> dict[str, list[float]]:
    base_vix = float(base_row.get("vix", 20.0))
    base_r = float(base_row.get("r", 0.0))
    base_q = float(base_row.get("q", 0.0))
    base_sentiment = float(base_row.get("sentiment_7d", 0.5))
    base_sentiment_20d = float(base_row.get("sentiment_20d", base_sentiment))
    base_t1 = float(base_row.get("T1", 0.25))
    base_t2 = float(base_row.get("T2", 0.5))
    close_price = float(base_row.get("close", base_row.get("S", 0.0)))

    return {
        "vix": [max(0.0, base_vix * factor) for factor in (0.7, 0.9, 1.0, 1.1, 1.5)],
        "sentiment_7d": [max(0.0, min(1.0, value)) for value in (base_sentiment - 0.25, base_sentiment - 0.1, base_sentiment, base_sentiment + 0.1, base_sentiment + 0.25)],
        "sentiment_20d": [max(0.0, min(1.0, value)) for value in (base_sentiment_20d - 0.2, base_sentiment_20d - 0.1, base_sentiment_20d, base_sentiment_20d + 0.1, base_sentiment_20d + 0.2)],
        "r": [max(0.0, base_r - 0.01), base_r, base_r + 0.01, base_r + 0.02],
        "q": [max(0.0, base_q - 0.005), base_q, base_q + 0.005],
        "T2": [max(base_t1 + 0.05, base_t2 - 0.25), max(base_t1 + 0.05, base_t2 - 0.1), max(base_t1 + 0.05, base_t2), base_t2 + 0.25, base_t2 + 0.5],
        "S": [close_price * factor for factor in (0.9, 0.95, 1.0, 1.05, 1.1)],
    }


def run_sensitivity_analysis(
    model: object,
    base_row: pd.Series,
    *,
    grid: dict[str, list[float]] | None = None,
    contract_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    grid = grid or build_sensitivity_grid(base_row)
    contract_overrides = contract_overrides or {}

    baseline_prediction = predict_chooser_price(model, base_row, contract_overrides=contract_overrides)
    rows: list[dict[str, object]] = []

    for feature_name, values in grid.items():
        for value in values:
            market_overrides: dict[str, float] = {}
            contract_mutation = dict(contract_overrides)
            if feature_name in {"S", "K", "moneyness", "T1", "T2"}:
                contract_mutation[feature_name] = float(value)
            else:
                market_overrides[feature_name] = float(value)

            model_price = predict_chooser_price(
                model,
                base_row,
                contract_overrides=contract_mutation,
                market_overrides=market_overrides,
            )
            refs = reference_quotes(
                base_row,
                contract_overrides=contract_mutation,
                market_overrides=market_overrides,
            )
            interval = estimate_price_interval(
                base_row,
                contract_overrides=contract_mutation,
                market_overrides=market_overrides,
                sigma=refs["sigma_reference"],
            )
            rows.append(
                {
                    "feature": feature_name,
                    "value": float(value),
                    "model_price": model_price,
                    "baseline_price": baseline_prediction,
                    "delta": model_price - baseline_prediction,
                    "delta_pct": (model_price - baseline_prediction) / baseline_prediction if baseline_prediction else np.nan,
                    "closed_form_quote": refs["closed_form_quote"],
                    "mc_quote": refs["mc_quote"],
                    "sigma_reference": refs["sigma_reference"],
                    "price_ci_lower": interval["lower"],
                    "price_ci_upper": interval["upper"],
                }
            )

    return pd.DataFrame(rows)


def run_scenario_stress_tests(
    model: object,
    base_row: pd.Series,
    *,
    contract_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    contract_overrides = contract_overrides or {}

    base_vix = float(base_row.get("vix", 20.0))
    base_r = float(base_row.get("r", 0.0))
    base_sentiment = float(base_row.get("sentiment_7d", 0.5))
    base_sentiment_20d = float(base_row.get("sentiment_20d", base_sentiment))

    scenarios: list[tuple[str, dict[str, float], dict[str, float]]] = [
        ("baseline", {}, {}),
        (
            "volatility_spike_50pct",
            {
                "vix": base_vix * 1.5,
                "vix_ma_ratio": float(base_row.get("vix_ma_ratio", 1.0)) * 1.5,
                "vix_change_5d": float(base_row.get("vix_change_5d", 0.0)) * 1.5,
                "vol_ratio_5_20": float(base_row.get("vol_ratio_5_20", 1.0)) * 1.5,
                "vol_ratio_20_60": float(base_row.get("vol_ratio_20_60", 1.0)) * 1.5,
                "vol_20d_change": float(base_row.get("vol_20d_change", 0.0)) * 1.5,
            },
            {},
        ),
        ("rate_hike_2pct", {"r": base_r + 0.02}, {}),
        (
            "sentiment_shock",
            {
                "sentiment_7d": max(0.0, base_sentiment - 0.3),
                "sentiment_20d": max(0.0, base_sentiment_20d - 0.25),
                "news_count_7d": float(base_row.get("news_count_7d", 0.0)) * 1.2,
            },
            {},
        ),
        (
            "combined_stress",
            {
                "vix": base_vix * 1.5,
                "vix_ma_ratio": float(base_row.get("vix_ma_ratio", 1.0)) * 1.5,
                "vix_change_5d": float(base_row.get("vix_change_5d", 0.0)) * 1.5,
                "sentiment_7d": max(0.0, base_sentiment - 0.3),
                "sentiment_20d": max(0.0, base_sentiment_20d - 0.25),
                "r": base_r + 0.02,
            },
            {},
        ),
    ]

    rows: list[dict[str, object]] = []
    baseline_price = predict_chooser_price(model, base_row, contract_overrides=contract_overrides)
    for scenario_name, market_overrides, contract_overrides_extra in scenarios:
        merged_contract_overrides = dict(contract_overrides)
        merged_contract_overrides.update(contract_overrides_extra)
        price = predict_chooser_price(
            model,
            base_row,
            contract_overrides=merged_contract_overrides,
            market_overrides=market_overrides,
        )
        refs = reference_quotes(
            base_row,
            contract_overrides=merged_contract_overrides,
            market_overrides=market_overrides,
        )
        rows.append(
            {
                "scenario": scenario_name,
                "model_price": price,
                "baseline_price": baseline_price,
                "delta": price - baseline_price,
                "delta_pct": (price - baseline_price) / baseline_price if baseline_price else np.nan,
                "closed_form_quote": refs["closed_form_quote"],
                "mc_quote": refs["mc_quote"],
                "sigma_reference": refs["sigma_reference"],
            }
        )

    return pd.DataFrame(rows)


def compute_shap_summary(
    model: object,
    feature_frame: pd.DataFrame,
    *,
    sample_size: int = 64,
) -> pd.DataFrame:
    if not HAS_SHAP or shap is None:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    feature_cols = get_pricing_model_columns(get_pricing_feature_columns())
    usable = feature_frame.copy()
    usable["S"] = usable.get("S", usable.get("close"))
    usable["K"] = usable.get("K", usable.get("close"))
    usable["moneyness"] = usable.get("moneyness", 1.0)
    usable["T1"] = usable.get("T1", 0.25)
    usable["T2"] = usable.get("T2", 0.5)
    usable = usable.dropna(subset=feature_cols)
    if usable.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    background = usable[feature_cols].head(min(sample_size, len(usable))).copy()
    sample = background.head(min(24, len(background))).copy()
    try:
        explainer = shap.Explainer(model.predict, background, algorithm="permutation")
        shap_values = explainer(sample)
        values = np.asarray(shap_values.values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        summary = pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        return summary.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])


def render_week7_report(
    *,
    artifact_path: Path,
    base_row: pd.Series,
    baseline_price: float,
    references: dict[str, float],
    sensitivity_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    shap_df: pd.DataFrame,
) -> str:
    feature_cols = get_pricing_model_columns(get_pricing_feature_columns())
    lines: list[str] = [
        "# Week 7 - Sensitivity analysis report",
        "",
        f"**Report date**: {TODAY_STAMP}  |  **Pipeline version**: {PIPELINE_VER}",
        "",
        "## 1. Tool Configuration",
        "",
        f"- Model artifact: {artifact_path.name}",
        f"- Reference date: {base_row.name.date() if isinstance(base_row.name, pd.Timestamp) else base_row.name}",
        f"- Input feature count: {len(feature_cols)}",
        f"- Baseline model price: {baseline_price:.6f}",
        f"- Closed-form reference: {references['closed_form_quote']:.6f}",
        f"- Monte Carlo reference: {references['mc_quote']:.6f}",
        "",
        "## 2. Extreme Scenario Tests",
        "",
        _render_markdown_table(scenario_df[["scenario", "model_price", "delta", "delta_pct", "closed_form_quote", "mc_quote"]]),
        "",
        "## 3. Sensitivity Grid",
        "",
        _render_markdown_table(sensitivity_df[["feature", "value", "model_price", "delta", "delta_pct"]], max_rows=25),
        "",
        "## 4. SHAP Impact Summary",
        "",
        _render_markdown_table(shap_df.head(10), max_rows=10) if not shap_df.empty else "_SHAP not available in the current environment._",
        "",
        "## 5. Observations",
        "",
        "- The pricing tool now exposes a direct quote, a closed-form reference, and a Monte Carlo reference for the same contract.",
        "- Scenario tests include a 50% volatility spike, a 2% rate hike, a sentiment shock, and a combined stress case.",
        "- Sensitivity grids are centered on the latest usable market row so the tool can be refreshed with new daily data.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _sanitize_report_for_pdf(md_text: str) -> str:
    cleaned_lines: list[str] = []
    in_code_block = False
    for raw_line in md_text.splitlines():
        line = raw_line.replace("`", "")
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            cleaned_lines.append(f"    {stripped}")
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).rstrip() + "\n"


def refresh_market_data() -> None:
    from apis import main as download_latest_data

    download_latest_data(force_refresh=True)


def _market_data_paths() -> list[Path]:
    from apis import DATA_DIR

    return [DATA_DIR / filename for filename in RAW_DATA_FILES]


def market_data_last_updated() -> datetime | None:
    timestamps: list[datetime] = []
    for path in _market_data_paths():
        if path.exists():
            timestamps.append(datetime.fromtimestamp(path.stat().st_mtime))
    return max(timestamps) if timestamps else None


def market_data_is_stale(*, max_age_hours: float = MARKET_DATA_MAX_AGE_HOURS) -> bool:
    latest_update = market_data_last_updated()
    if latest_update is None:
        return True
    return datetime.now() - latest_update >= timedelta(hours=max_age_hours)


def refresh_market_data_if_stale(*, max_age_hours: float = MARKET_DATA_MAX_AGE_HOURS) -> bool:
    if market_data_is_stale(max_age_hours=max_age_hours):
        refresh_market_data()
        return True
    return False


def run_week7_workflow(
    *,
    model_name: str | None = None,
    reference_date: str | None = None,
    refresh_raw_data: bool = False,
) -> dict[str, Path | pd.DataFrame | str]:
    if refresh_raw_data:
        refresh_market_data()

    artifact_path, _, model = load_model_bundle(model_name=model_name)
    feature_frame = load_feature_frame()
    base_row = select_reference_row(feature_frame, reference_date=reference_date)

    contract_overrides = {
        "S": float(base_row.get("close", base_row.get("S", 0.0))),
        "K": float(base_row.get("close", base_row.get("S", 0.0))),
        "moneyness": 1.0,
        "T1": 0.25,
        "T2": 0.5,
    }

    baseline_price = predict_chooser_price(model, base_row, contract_overrides=contract_overrides)
    refs = reference_quotes(base_row, contract_overrides=contract_overrides)
    sensitivity_df = run_sensitivity_analysis(model, base_row, contract_overrides=contract_overrides)
    scenario_df = run_scenario_stress_tests(model, base_row, contract_overrides=contract_overrides)
    shap_df = compute_shap_summary(model, feature_frame)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sensitivity_path = PROCESSED_DIR / f"{SENSITIVITY_STEM}.csv"
    scenario_path = PROCESSED_DIR / f"{SCENARIO_STEM}.csv"
    shap_path = PROCESSED_DIR / f"{SHAP_STEM}.csv"
    report_path = REPORTS_DIR / f"{REPORT_STEM}.md"

    sensitivity_df.to_csv(sensitivity_path, index=False)
    scenario_df.to_csv(scenario_path, index=False)
    shap_df.to_csv(shap_path, index=False)

    report_text = render_week7_report(
        artifact_path=artifact_path,
        base_row=base_row,
        baseline_price=baseline_price,
        references=refs,
        sensitivity_df=sensitivity_df,
        scenario_df=scenario_df,
        shap_df=shap_df,
    )
    report_path.write_text(report_text, encoding="utf-8")

    pdf_path = REPORTS_DIR / f"{REPORT_STEM}.pdf"
    export_markdown_pdf(_sanitize_report_for_pdf(report_text), pdf_path)
    normalize_pdf_compatibility(pdf_path)

    return {
        "artifact_path": artifact_path,
        "report_path": report_path,
        "pdf_path": pdf_path,
        "sensitivity_path": sensitivity_path,
        "scenario_path": scenario_path,
        "shap_path": shap_path,
        "sensitivity_df": sensitivity_df,
        "scenario_df": scenario_df,
        "shap_df": shap_df,
        "baseline_price": baseline_price,
        "references": pd.Series(refs),
        "base_row": base_row,
        "feature_frame": feature_frame,
        "report_text": report_text,
    }


def main() -> None:
    outputs = run_week7_workflow()
    print(f"[OK] Week 7 analysis saved to {outputs['report_path']}")
    print(f"[OK] Week 7 PDF saved to {outputs['pdf_path']}")
    print(f"[OK] Sensitivity table saved to {outputs['sensitivity_path']}")
    print(f"[OK] Scenario table saved to {outputs['scenario_path']}")
    print(f"[OK] SHAP summary saved to {outputs['shap_path']}")


if __name__ == "__main__":
    main()