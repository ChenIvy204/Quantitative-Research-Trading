from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - only raised when running the API
    raise RuntimeError("Install fastapi and uvicorn to run the Week 7 API prototype.") from exc

from week7_toolkit import (  # noqa: E402
    available_model_artifacts,
    load_feature_frame,
    load_model_bundle,
    predict_chooser_price,
    reference_quotes,
    refresh_market_data,
    run_scenario_stress_tests,
    run_sensitivity_analysis,
    select_reference_row,
)

app = FastAPI(title="Week 7 Pricing Tool API", version="v1.0")


class QuoteRequest(BaseModel):
    model_name: str | None = None
    reference_date: str | None = None
    S: float | None = None
    K: float | None = None
    moneyness: float = Field(default=1.0, ge=0.5, le=2.0)
    T1: float = Field(default=0.25, ge=0.0)
    T2: float = Field(default=0.5, ge=0.01)


class SensitivityRequest(QuoteRequest):
    feature: str = "vix"
    values: list[float] | None = None


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "available_models": [path.name for path in available_model_artifacts("approach2")],
    }


@app.post("/quote")
def quote(request: QuoteRequest) -> dict[str, object]:
    try:
        artifact_path, _, model = load_model_bundle(model_name=request.model_name)
        feature_frame = load_feature_frame()
        base_row = select_reference_row(feature_frame, reference_date=request.reference_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    contract_overrides = {
        "S": request.S if request.S is not None else float(base_row.get("close", 0.0)),
        "K": request.K,
        "moneyness": request.moneyness,
        "T1": request.T1,
        "T2": request.T2,
    }
    live_price = predict_chooser_price(model, base_row, contract_overrides=contract_overrides)
    refs = reference_quotes(base_row, contract_overrides=contract_overrides)
    return {
        "artifact": artifact_path.name,
        "model_price": live_price,
        "closed_form_reference": refs["closed_form_quote"],
        "mc_reference": refs["mc_quote"],
        "sigma_reference": refs["sigma_reference"],
    }


@app.post("/sensitivity")
def sensitivity(request: SensitivityRequest) -> dict[str, object]:
    try:
        _, _, model = load_model_bundle(model_name=request.model_name)
        feature_frame = load_feature_frame()
        base_row = select_reference_row(feature_frame, reference_date=request.reference_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    contract_overrides = {
        "S": request.S if request.S is not None else float(base_row.get("close", 0.0)),
        "K": request.K,
        "moneyness": request.moneyness,
        "T1": request.T1,
        "T2": request.T2,
    }
    grid = {request.feature: request.values or []}
    sensitivity_df = run_sensitivity_analysis(model, base_row, grid=grid, contract_overrides=contract_overrides)
    return {"rows": sensitivity_df.to_dict(orient="records")}


@app.post("/scenario-tests")
def scenario_tests(request: QuoteRequest) -> dict[str, object]:
    try:
        _, _, model = load_model_bundle(model_name=request.model_name)
        feature_frame = load_feature_frame()
        base_row = select_reference_row(feature_frame, reference_date=request.reference_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    contract_overrides = {
        "S": request.S if request.S is not None else float(base_row.get("close", 0.0)),
        "K": request.K,
        "moneyness": request.moneyness,
        "T1": request.T1,
        "T2": request.T2,
    }
    scenario_df = run_scenario_stress_tests(model, base_row, contract_overrides=contract_overrides)
    return {"rows": scenario_df.to_dict(orient="records")}


@app.post("/refresh-data")
def refresh_data() -> dict[str, str]:
    try:
        refresh_market_data()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "refreshed"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("week7_api:app", host="127.0.0.1", port=8000, reload=False)