from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - only raised when running the app
    raise RuntimeError("Install streamlit to run the Week 7 pricing tool prototype.") from exc

from week7_toolkit import (  # noqa: E402
    available_model_artifacts,
    batch_price_contracts,
    build_iv_surface,
    load_feature_frame,
    load_model_bundle,
    compute_greeks,
    estimate_price_interval,
    predict_chooser_price,
    reference_quotes,
    refresh_market_data,
    run_scenario_stress_tests,
    run_sensitivity_analysis,
    select_reference_row,
)


@st.cache_data(show_spinner=False)
def _cached_feature_frame():
    return load_feature_frame()


@st.cache_resource(show_spinner=False)
def _cached_model_bundle(model_name: str):
    return load_model_bundle(model_name=model_name)


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


def main() -> None:
    st.set_page_config(page_title="Week 7 Pricing Tool", layout="wide")
    st.title("Week 7 Advanced Analysis & Pricing Tool")
    st.caption("Sensitivity analysis, stress testing, SHAP summary, and a live pricing prototype built on the Week 6 chooser model.")

    model_paths = available_model_artifacts("approach2")
    model_names = [path.stem.replace("week6_", "") for path in model_paths]
    if not model_names:
        st.error("No Week 6 pricing model artifacts were found.")
        return

    with st.sidebar:
        st.header("Quote Inputs")
        model_name = st.selectbox("Model", model_names, index=0)
        reference_date = st.date_input("Reference date")
        preset_name = st.selectbox("Quick preset", ["Custom", "ATM", "OTM 10%", "ITM 10%"], index=0)
        strike_multiplier = st.slider("Moneyness", 0.80, 1.20, 1.00, 0.01)
        time_to_choice = st.slider("T1 (years)", 0.05, 1.00, 0.25, 0.05)
        maturity = st.slider("T2 (years)", 0.10, 1.50, 0.50, 0.05)
        refresh_clicked = st.button("Refresh market data")

    if refresh_clicked:
        with st.spinner("Refreshing raw data feeds..."):
            refresh_market_data()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Raw data refresh completed.")

    artifact_path, payload, model = _cached_model_bundle(model_name)
    feature_frame = _cached_feature_frame()
    base_row = select_reference_row(feature_frame, reference_date=str(reference_date) if reference_date else None)
    if preset_name == "Custom":
        contract_overrides = _contract_defaults(
            base_row,
            strike_multiplier=strike_multiplier,
            time_to_choice=time_to_choice,
            maturity=maturity,
        )
    else:
        contract_overrides = _preset_contract(base_row, preset_name, time_to_choice, maturity)

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

    live_price = predict_chooser_price(model, base_row, contract_overrides=contract_overrides)
    refs = reference_quotes(base_row, contract_overrides=contract_overrides)
    greeks_df = compute_greeks(base_row, contract_overrides=contract_overrides, sigma=refs["sigma_reference"])
    interval = estimate_price_interval(base_row, contract_overrides=contract_overrides, sigma=refs["sigma_reference"])
    sensitivity_df = run_sensitivity_analysis(model, base_row, contract_overrides=contract_overrides)
    scenario_df = run_scenario_stress_tests(model, base_row, contract_overrides=contract_overrides)
    iv_surface = build_iv_surface(base_row, model_price=live_price, base_contract_overrides=contract_overrides)

    st.subheader("Live Quote")
    quote_cols = st.columns(4)
    quote_cols[0].metric("Model price", f"{live_price:.4f}")
    quote_cols[1].metric("Closed-form reference", f"{refs['closed_form_quote']:.4f}")
    quote_cols[2].metric("Monte Carlo reference", f"{refs['mc_quote']:.4f}")
    quote_cols[3].metric("Reference sigma", f"{refs['sigma_reference']:.4f}")

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

    left, right = st.columns(2)
    with left:
        st.subheader("Sensitivity Grid")
        st.dataframe(sensitivity_df, width="stretch")
        st.line_chart(sensitivity_df.pivot_table(index="value", columns="feature", values="model_price", aggfunc="mean"))

    with right:
        st.subheader("Stress Scenarios")
        st.dataframe(scenario_df, width="stretch")
        st.bar_chart(scenario_df.set_index("scenario")["delta"])

    st.subheader("Implied Volatility Surface")
    st.dataframe(iv_surface, width="stretch", hide_index=True)
    _render_iv_heatmap(iv_surface)
    st.caption("The heatmap shows the model-implied volatility surface across moneyness and maturity.")

    st.subheader("Batch Pricing")
    batch_file = st.file_uploader("Upload CSV with contract rows", type=["csv"])
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        batch_results = batch_price_contracts(model, base_row, batch_df)
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