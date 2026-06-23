from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MODELS_DIR = ROOT / "data" / "models"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import streamlit as st  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - only raised when running the app
    raise RuntimeError("Install streamlit to run the Week 7 pricing tool prototype.") from exc

DEFAULT_MODEL_PRIORITY = (
    "approach2_neuralnetwork_v1.0",
    "approach2_xgboost_v1.0",
    "approach2_linearregression_v1.0",
)


def _available_model_names() -> list[str]:
    model_paths = sorted(MODELS_DIR.glob("week6_approach2_*_v1.0.joblib"))
    model_names = [path.stem.replace("week6_", "") for path in model_paths]
    priority_map = {name: index for index, name in enumerate(DEFAULT_MODEL_PRIORITY)}
    return sorted(model_names, key=lambda name: (priority_map.get(name, len(DEFAULT_MODEL_PRIORITY)), name))


def _toolkit():
    from week7_toolkit import (  # noqa: E402
        best_pricing_model_name,
        batch_price_contracts,
        build_iv_surface,
        build_price_trend_frame,
        compute_greeks,
        estimate_price_interval,
        load_bsm_error_summary,
        load_pricing_performance_summary,
        market_data_is_stale,
        load_feature_frame,
        load_model_bundle,
        predict_chooser_price,
        reference_quotes,
        refresh_market_data,
        refresh_market_data_if_stale,
        run_scenario_stress_tests,
        run_sensitivity_analysis,
        select_reference_row,
    )

    return {
        "best_pricing_model_name": best_pricing_model_name,
        "batch_price_contracts": batch_price_contracts,
        "build_iv_surface": build_iv_surface,
        "build_price_trend_frame": build_price_trend_frame,
        "compute_greeks": compute_greeks,
        "estimate_price_interval": estimate_price_interval,
        "load_bsm_error_summary": load_bsm_error_summary,
        "load_pricing_performance_summary": load_pricing_performance_summary,
        "market_data_is_stale": market_data_is_stale,
        "refresh_market_data_if_stale": refresh_market_data_if_stale,
        "load_feature_frame": load_feature_frame,
        "load_model_bundle": load_model_bundle,
        "predict_chooser_price": predict_chooser_price,
        "reference_quotes": reference_quotes,
        "refresh_market_data": refresh_market_data,
        "run_scenario_stress_tests": run_scenario_stress_tests,
        "run_sensitivity_analysis": run_sensitivity_analysis,
        "select_reference_row": select_reference_row,
    }


@st.cache_data(show_spinner=False)
def _cached_feature_frame():
    return _toolkit()["load_feature_frame"]()


@st.cache_resource(show_spinner=False)
def _cached_model_bundle(model_name: str):
    return _toolkit()["load_model_bundle"](model_name=model_name)


@st.cache_data(show_spinner=False)
def _cached_pricing_context(reference_date: str | None, model_name: str):
    feature_frame = _cached_feature_frame()
    base_row = _toolkit()["select_reference_row"](feature_frame, reference_date=reference_date)
    artifact_path, payload, model = _cached_model_bundle(model_name)
    return feature_frame, base_row, artifact_path, payload, model


def _contract_defaults(base_row, *, strike_multiplier: float, time_to_choice: float, maturity: float) -> dict[str, float]:
    close_price = float(base_row.get("close", 0.0))
    return {
        "S": close_price,
        "K": close_price * strike_multiplier,
        "moneyness": strike_multiplier,
        "T1": time_to_choice,
        "T2": maturity,
    }


def _preset_contract(base_row, preset_name: str, time_to_choice: float, maturity: float) -> dict[str, float]:
    close_price = float(base_row.get("close", 0.0))
    presets = {
        "ATM": 1.00,
        "ITM 10%": 0.90,
        "OTM 10%": 1.10,
    }
    moneyness = presets.get(preset_name, 1.00)
    return {
        "S": close_price,
        "K": close_price * moneyness,
        "moneyness": moneyness,
        "T1": min(time_to_choice, max(0.05, maturity - 0.05)),
        "T2": max(maturity, time_to_choice + 0.05),
    }


def _render_iv_heatmap(iv_surface: pd.DataFrame) -> None:
    pivot = iv_surface.pivot_table(index="T2", columns="moneyness", values="implied_vol", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}x" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in pivot.index])
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("T2 (years)")
    ax.set_title("Implied Volatility Surface")
    fig.colorbar(image, ax=ax, label="Implied Vol")
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _render_dual_price_chart(bsm_price: float, ml_price: float, lower: float, upper: float, *, best_model_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.scatter([0], [bsm_price], s=120, color="#1565C0", label="BSM")
    ax.scatter([1], [ml_price], s=120, color="#E53935", label=best_model_label)
    ax.errorbar(
        [1],
        [ml_price],
        yerr=[[max(0.0, ml_price - lower)], [max(0.0, upper - ml_price)]],
        fmt="none",
        ecolor="#E53935",
        elinewidth=2,
        capsize=6,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BSM", best_model_label], rotation=0)
    ax.set_ylabel("Price")
    ax.set_title("Dual Pricing with Model Uncertainty")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _render_trend_chart(trend_df: pd.DataFrame, *, best_model_label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(trend_df["date"], trend_df["bsm_price"], color="#1565C0", linewidth=2.2, label="BSM")
    ax.plot(trend_df["date"], trend_df["model_price"], color="#E53935", linewidth=2.2, label=best_model_label)
    ax.fill_between(
        trend_df["date"],
        trend_df["price_ci_lower"],
        trend_df["price_ci_upper"],
        color="#E53935",
        alpha=0.12,
        label="ML uncertainty band",
    )
    ax.set_title("Price Trend Over Recent Market Dates")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _render_sensitivity_curve(sensitivity_df: pd.DataFrame, feature_name: str, *, best_model_label: str) -> None:
    subset = sensitivity_df[sensitivity_df["feature"] == feature_name].copy()
    if subset.empty:
        st.info("No sensitivity rows were produced for the selected feature.")
        return

    subset = subset.sort_values("value")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(subset["value"], subset["closed_form_quote"], color="#1565C0", linewidth=2.0, label="BSM")
    ax.plot(subset["value"], subset["model_price"], color="#E53935", linewidth=2.0, label=best_model_label)
    ax.errorbar(
        subset["value"],
        subset["model_price"],
        yerr=[
            (subset["model_price"] - subset["price_ci_lower"]).clip(lower=0.0),
            (subset["price_ci_upper"] - subset["model_price"]).clip(lower=0.0),
        ],
        fmt="none",
        ecolor="#E53935",
        alpha=0.55,
        capsize=4,
    )
    ax.set_title(f"Sensitivity Curve: {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def main(*, set_page_config: bool = True, show_landing_page: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="Pricing Dashboard", layout="wide")
    st.title("Pricing Dashboard")
    st.caption("Sensitivity analysis, stress testing, SHAP summary, and a live pricing prototype built on the Week 6 chooser model.")
    if _toolkit()["refresh_market_data_if_stale"]():
        st.info("Market data was refreshed automatically because the local cache was stale.")

    if show_landing_page:
        if "week7_dashboard_open" not in st.session_state:
            st.session_state["week7_dashboard_open"] = False

        if not st.session_state["week7_dashboard_open"]:
            st.markdown(
                """
                ### Quick Start
                This page now loads in a lightweight mode first.

                Click below to open the pricing dashboard, load the model, and run the quote workflow.
                """
            )
            if st.button("Open pricing dashboard", type="primary"):
                st.session_state["week7_dashboard_open"] = True
                st.rerun()
            st.stop()

    model_names = _available_model_names()
    if not model_names:
        st.error("No Week 6 pricing model artifacts were found.")
        return

    pricing_summary = _toolkit()["load_pricing_performance_summary"]()
    bsm_summary = _toolkit()["load_bsm_error_summary"]()
    best_model_label = _toolkit()["best_pricing_model_name"](pricing_summary) or model_names[0]
    best_model_index = next((idx for idx, name in enumerate(model_names) if best_model_label.lower() in name.lower()), 0)

    with st.sidebar:
        st.header("Quote Inputs")
        model_name = st.selectbox("Model", model_names, index=best_model_index)
        reference_date = st.date_input("Reference date")
        preset_name = st.selectbox("Quick preset", ["Custom", "ATM", "OTM 10%", "ITM 10%"], index=0)
        strike_multiplier = st.slider("Moneyness", 0.80, 1.20, 1.00, 0.01)
        time_to_choice = st.slider("T1 (years)", 0.05, 1.00, 0.25, 0.05)
        maturity = st.slider("T2 (years)", 0.10, 1.50, 0.50, 0.05)
        refresh_clicked = st.button("Refresh market data")

    if refresh_clicked:
        with st.spinner("Refreshing raw data feeds..."):
            _toolkit()["refresh_market_data"]()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Raw data refresh completed.")

    reference_key = str(reference_date) if reference_date else None
    if preset_name == "Custom":
        contract_overrides = None
    else:
        contract_overrides = None

    load_clicked = st.button("Load model and generate quote", type="primary")
    if not load_clicked:
        st.info("The dashboard is open. Click the button to load the model and generate the quote.")
        st.stop()

    toolkit = _toolkit()
    feature_frame, base_row, artifact_path, payload, model = _cached_pricing_context(reference_key, model_name)
    if preset_name == "Custom":
        contract_overrides = _contract_defaults(
            base_row,
            strike_multiplier=strike_multiplier,
            time_to_choice=time_to_choice,
            maturity=maturity,
        )
    else:
        contract_overrides = _preset_contract(base_row, preset_name, time_to_choice, maturity)
    run_heavy_analysis = True

    st.subheader("Contract Setup")
    contract_cols = st.columns(4)
    contract_cols[0].metric("S", f"{contract_overrides['S']:.2f}")
    contract_cols[1].metric("K", f"{contract_overrides['K']:.2f}")
    contract_cols[2].metric("T1", f"{contract_overrides['T1']:.2f}")
    contract_cols[3].metric("T2", f"{contract_overrides['T2']:.2f}")

    generate_clicked = st.button("Generate Quote", type="primary")
    if generate_clicked:
        st.toast("Quote refreshed", icon="✅")
    else:
        st.caption("Outputs below are shown from the latest selected inputs. Click Generate Quote to refresh them.")

    live_price = toolkit["predict_chooser_price"](model, base_row, contract_overrides=contract_overrides)
    refs = toolkit["reference_quotes"](base_row, contract_overrides=contract_overrides)
    greeks_df = toolkit["compute_greeks"](base_row, contract_overrides=contract_overrides, sigma=refs["sigma_reference"])
    interval = toolkit["estimate_price_interval"](base_row, contract_overrides=contract_overrides, sigma=refs["sigma_reference"])
    if run_heavy_analysis:
        sensitivity_df = toolkit["run_sensitivity_analysis"](model, base_row, contract_overrides=contract_overrides)
        scenario_df = toolkit["run_scenario_stress_tests"](model, base_row, contract_overrides=contract_overrides)
        iv_surface = toolkit["build_iv_surface"](base_row, model_price=live_price, base_contract_overrides=contract_overrides)
        trend_df = toolkit["build_price_trend_frame"](
            feature_frame,
            model,
            contract_overrides=contract_overrides,
            lookback_days=30,
            anchor_date=base_row.name if isinstance(base_row.name, pd.Timestamp) else None,
        )
    else:
        sensitivity_df = pd.DataFrame()
        scenario_df = pd.DataFrame()
        iv_surface = pd.DataFrame()
        trend_df = pd.DataFrame()

    st.subheader("Dual Pricing Overview")
    quote_cols = st.columns(4)
    quote_cols[0].metric("BSM price", f"{refs['closed_form_quote']:.4f}")
    quote_cols[1].metric(f"Best ML price", f"{live_price:.4f}")
    quote_cols[2].metric("ML-BSM gap", f"{live_price - refs['closed_form_quote']:.4f}")
    quote_cols[3].metric("95% CI width", f"{interval['upper'] - interval['lower']:.4f}")

    _render_dual_price_chart(
        refs["closed_form_quote"],
        live_price,
        interval["lower"],
        interval["upper"],
        best_model_label=best_model_label,
    )

    if not trend_df.empty:
        st.subheader("Price Trend")
        _render_trend_chart(trend_df, best_model_label=best_model_label)
    else:
        st.info("No trend data was produced for the selected date range.")

    st.subheader("Risk Metrics")
    greek_cols = st.columns(4)
    greek_map = dict(zip(greeks_df["metric"], greeks_df["value"], strict=False))
    greek_cols[0].metric("Delta", f"{greek_map.get('delta', 0.0):.4f}")
    greek_cols[1].metric("Gamma", f"{greek_map.get('gamma', 0.0):.4f}")
    greek_cols[2].metric("Vega", f"{greek_map.get('vega', 0.0):.4f}")
    greek_cols[3].metric("Price CI", f"{interval['lower']:.4f} - {interval['upper']:.4f}")

    st.dataframe(greeks_df, width="stretch", hide_index=True)

    st.subheader("Base Market Snapshot")
    snapshot_fields = [
        "close",
        "vix",
        "r",
        "q",
        "sentiment_7d",
        "sentiment_20d",
        "news_count_7d",
        "hist_vol_20d",
        "vol_20d_change",
    ]
    base_snapshot = {field: base_row.get(field) for field in snapshot_fields if field in base_row.index}
    st.dataframe(pd.DataFrame([base_snapshot]), width="stretch")

    st.subheader("Performance Summary")
    perf_cols = st.columns(3)
    ml_pricing_summary = pricing_summary[~pricing_summary["model"].astype(str).str.contains("bsm", case=False, na=False)].copy()
    best_perf_row = ml_pricing_summary.iloc[0] if not ml_pricing_summary.empty else None
    if best_perf_row is not None:
        perf_cols[0].metric("Best ML model", str(best_perf_row.get("model", best_model_label)))
        perf_cols[1].metric("Test MAE", f"{float(best_perf_row.get('mae', 0.0)):.4f}")
        perf_cols[2].metric("Test RMSE", f"{float(best_perf_row.get('rmse', 0.0)):.4f}")
    else:
        perf_cols[0].metric("Best ML model", best_model_label)
        perf_cols[1].metric("Test MAE", "N/A")
        perf_cols[2].metric("Test RMSE", "N/A")

    perf_left, perf_right = st.columns(2)
    with perf_left:
        st.caption("Week 6 ML pricing leaderboard")
        st.dataframe(ml_pricing_summary.head(5), width="stretch", hide_index=True)
    with perf_right:
        st.caption("Latest Week 4 BSM benchmark")
        if not bsm_summary.empty and "group" in bsm_summary.columns:
            overall_bsm = bsm_summary[bsm_summary["group"].astype(str) == "overall"].head(1)
            st.dataframe(overall_bsm if not overall_bsm.empty else bsm_summary.head(5), width="stretch", hide_index=True)
        else:
            st.info("No BSM benchmark summary CSV was found in data/processed.")

    left, right = st.columns(2)
    with left:
        st.subheader("Sensitivity Grid")
        if not sensitivity_df.empty:
            sensitivity_feature = st.selectbox("Sensitivity feature", sorted(sensitivity_df["feature"].unique().tolist()))
            st.dataframe(sensitivity_df[sensitivity_df["feature"] == sensitivity_feature], width="stretch")
            _render_sensitivity_curve(sensitivity_df, sensitivity_feature, best_model_label=best_model_label)
        else:
            st.info("No sensitivity data was produced for the selected inputs.")

    with right:
        st.subheader("Stress Scenarios")
        if not scenario_df.empty:
            st.dataframe(scenario_df, width="stretch")
            st.bar_chart(scenario_df.set_index("scenario")["delta"])
        else:
            st.info("No stress scenarios were produced for the selected inputs.")

    st.subheader("Implied Volatility Surface")
    if not iv_surface.empty:
        st.dataframe(iv_surface, width="stretch", hide_index=True)
        _render_iv_heatmap(iv_surface)
        st.caption("The heatmap shows the model-implied volatility surface across moneyness and maturity.")
    else:
        st.info("No IV surface data was produced for the selected inputs.")

    st.subheader("Batch Pricing")
    batch_file = st.file_uploader("Upload CSV with contract rows", type=["csv"])
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        batch_results = toolkit["batch_price_contracts"](model, base_row, batch_df)
        st.dataframe(batch_results, width="stretch")
        st.download_button(
            "Download priced CSV",
            batch_results.to_csv(index=False).encode("utf-8"),
            file_name="week7_batch_pricing_results.csv",
            mime="text/csv",
        )

    st.subheader("Model Metadata")
    st.write(
        {
            "artifact": artifact_path.name,
            "saved_keys": sorted(payload.keys()),
            "reference_date": str(base_row.name.date() if hasattr(base_row.name, "date") else base_row.name),
        }
    )


if __name__ == "__main__":
    main()