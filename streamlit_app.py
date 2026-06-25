from __future__ import annotations

import importlib


dashboard = importlib.import_module("scripts.week7_app")
dashboard.main(set_page_config=True, show_landing_page=False)
