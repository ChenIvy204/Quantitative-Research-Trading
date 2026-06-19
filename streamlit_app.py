from __future__ import annotations

import importlib

import streamlit as st


st.set_page_config(page_title="Week 7 - Sensitivity analysis report", layout="wide")
st.title("Week 7 - Sensitivity analysis report")
st.caption("Lightweight Streamlit Cloud entrypoint for the Week 7 pricing dashboard.")

st.markdown(
    """
    ### Quick Start
    This entry page is intentionally minimal so it can render immediately on Streamlit Cloud.

    Click below to open the full pricing dashboard after the page has loaded.
    """
)

if st.button("Open pricing dashboard", type="primary"):
    dashboard = importlib.import_module("scripts.week7_app")
    dashboard.main(set_page_config=False, show_landing_page=False)
else:
    st.info("The app is ready. Open the dashboard when you want to load the model and run analysis.")
