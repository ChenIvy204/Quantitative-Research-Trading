from __future__ import annotations

import streamlit as st

from scripts.week7_toolkit import run_week7_workflow


st.set_page_config(page_title="Week 7 Analysis Report", layout="wide")
st.title("Week 7 - Sensitivity analysis report")
st.caption("Standalone report view for the Week 7 workflow.")

run_clicked = st.button("Generate report", type="primary")
if run_clicked:
    outputs = run_week7_workflow()
    st.success("Report generated.")
    st.write(
        {
            "report": str(outputs["report_path"]),
            "pdf": str(outputs["pdf_path"]),
        }
    )
    st.code(outputs["report_text"], language="markdown")
else:
    st.info("Click the button to generate the report on demand.")