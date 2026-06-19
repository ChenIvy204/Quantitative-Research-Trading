from __future__ import annotations

import importlib

import streamlit as st


dashboard = importlib.import_module("scripts.week7_app")
dashboard.main(set_page_config=True, show_landing_page=False)
