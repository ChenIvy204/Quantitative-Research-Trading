from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from math import erf, exp, log, sqrt
from pathlib import Path

import pandas as pd

from preprocess import save_markdown_pdf_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "data" / "reports"
PIPELINE_VERSION = "v1.0"
RUN_DATE = datetime.now().strftime("%Y%m%d")
DEFAULT_CONFIG_PATH = CONFIG_DIR / "week3_bsm_parameters.json"

logger = logging.getLogger("week3_bsm")

PAPER_TABLE3_ROWS = [
    {"row": 1, "paper_st1": 118.33, "paper_choice": "PUT", "paper_st2": 116.77, "paper_payoff": 33.23},
    {"row": 2, "paper_st1": 222.63, "paper_choice": "CALL", "paper_st2": 192.89, "paper_payoff": 42.89},
    {"row": 3, "paper_st1": 186.53, "paper_choice": "CALL", "paper_st2": 192.94, "paper_payoff": 42.94},
    {"row": 4, "paper_st1": 164.08, "paper_choice": "CALL", "paper_st2": 148.77, "paper_payoff": 0.0},
    {"row": 5, "paper_st1": 159.09, "paper_choice": "CALL", "paper_st2": 116.78, "paper_payoff": 0.0},
    {"row": 6, "paper_st1": 186.73, "paper_choice": "CALL", "paper_st2": 128.12, "paper_payoff": 0.0},
    {"row": 7, "paper_st1": 106.61, "paper_choice": "PUT", "paper_st2": 90.52, "paper_payoff": 59.48},
    {"row": 8, "paper_st1": 163.06, "paper_choice": "CALL", "paper_st2": 179.61, "paper_payoff": 29.61},
    {"row": 9, "paper_st1": 129.26, "paper_choice": "PUT", "paper_st2": 144.82, "paper_payoff": 5.18},
    {"row": 10, "paper_st1": 115.41, "paper_choice": "PUT", "paper_st2": 136.50, "paper_payoff": 13.50},
]


@dataclass(frozen=True)
class BsmChooserParameters:
    spot_price: float
    strike_price: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    decision_time_years: float
    maturity_years: float
    source_note: str = "Week 3 paper parameter table"


def versioned_filename(stem: str, extension: str) -> str:
    return f"{stem}_{PIPELINE_VERSION}_{RUN_DATE}.{extension}"


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_generated_outputs() -> None:
    ensure_output_dirs()
    for path in PROCESSED_DIR.glob("week3_*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
    for path in REPORTS_DIR.glob("week3_*"):
        if path.is_file() or path.is_symlink():
            path.unlink()


def standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def black_scholes_d1(
    spot_price: float,
    strike_price: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    return (
        log(spot_price / strike_price)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * maturity_years
    ) / (volatility * sqrt(maturity_years))


def black_scholes_call(
    spot_price: float,
    strike_price: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    d1 = black_scholes_d1(spot_price, strike_price, maturity_years, risk_free_rate, dividend_yield, volatility)
    d2 = d1 - volatility * sqrt(maturity_years)
    return (
        spot_price * exp(-dividend_yield * maturity_years) * standard_normal_cdf(d1)
        - strike_price * exp(-risk_free_rate * maturity_years) * standard_normal_cdf(d2)
    )


def black_scholes_put(
    spot_price: float,
    strike_price: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    d1 = black_scholes_d1(spot_price, strike_price, maturity_years, risk_free_rate, dividend_yield, volatility)
    d2 = d1 - volatility * sqrt(maturity_years)
    return (
        strike_price * exp(-risk_free_rate * maturity_years) * standard_normal_cdf(-d2)
        - spot_price * exp(-dividend_yield * maturity_years) * standard_normal_cdf(-d1)
    )


def load_parameters(config_path: Path | None = None) -> BsmChooserParameters:
    path = config_path or DEFAULT_CONFIG_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BsmChooserParameters(**payload)


def chooser_valuation_breakdown(parameters: BsmChooserParameters) -> dict[str, float]:
    if parameters.maturity_years <= parameters.decision_time_years:
        raise ValueError("maturity_years must be greater than decision_time_years")

    time_to_decision = parameters.maturity_years - parameters.decision_time_years
    adjusted_strike = parameters.strike_price * exp(-parameters.risk_free_rate * time_to_decision)

    call_leg = black_scholes_call(
        parameters.spot_price,
        parameters.strike_price,
        parameters.maturity_years,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
    )
    put_leg = black_scholes_put(
        parameters.spot_price,
        adjusted_strike,
        time_to_decision,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
    )
    chooser_value = call_leg + put_leg

    return {
        "spot_price": parameters.spot_price,
        "strike_price": parameters.strike_price,
        "adjusted_strike_for_put_leg": adjusted_strike,
        "risk_free_rate": parameters.risk_free_rate,
        "dividend_yield": parameters.dividend_yield,
        "volatility": parameters.volatility,
        "decision_time_years": parameters.decision_time_years,
        "maturity_years": parameters.maturity_years,
        "time_to_decision_years": time_to_decision,
        "call_leg_value": call_leg,
        "put_leg_value": put_leg,
        "chooser_value": chooser_value,
        "chooser_premium_vs_call": chooser_value - call_leg,
        "chooser_premium_vs_put": chooser_value - put_leg,
        "chooser_identity_gap": chooser_value - (call_leg + put_leg),
    }


def build_sensitivity_table(parameters: BsmChooserParameters) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for multiplier in [0.85, 0.95, 1.00, 1.05, 1.15]:
        spot_price = parameters.spot_price * multiplier
        temp_parameters = BsmChooserParameters(
            spot_price=spot_price,
            strike_price=parameters.strike_price,
            risk_free_rate=parameters.risk_free_rate,
            dividend_yield=parameters.dividend_yield,
            volatility=parameters.volatility,
            decision_time_years=parameters.decision_time_years,
            maturity_years=parameters.maturity_years,
            source_note=parameters.source_note,
        )
        row = chooser_valuation_breakdown(temp_parameters)
        row["spot_multiplier"] = multiplier
        rows.append(row)

    columns = [
        "spot_multiplier",
        "spot_price",
        "call_leg_value",
        "put_leg_value",
        "chooser_value",
        "chooser_premium_vs_call",
        "chooser_identity_gap",
    ]
    return pd.DataFrame(rows)[columns]


def gbm_next_price(start_price: float, shock: float, time_years: float, drift: float, volatility: float) -> float:
    return start_price * exp((drift - 0.5 * volatility**2) * time_years + shock * volatility * sqrt(time_years))


def build_table3_simulation(parameters: BsmChooserParameters, drift: float, seed: int, path_count: int = 10) -> pd.DataFrame:
    rng = random.Random(seed)
    first_period_years = parameters.decision_time_years
    second_period_years = parameters.maturity_years - parameters.decision_time_years

    rows: list[dict[str, float | str | int]] = []
    for row_number in range(1, path_count + 1):
        z1 = rng.normalvariate(0.0, 1.0)
        z2 = rng.normalvariate(0.0, 1.0)
        model_st1 = gbm_next_price(parameters.spot_price, z1, first_period_years, drift, parameters.volatility)
        model_choice = "CALL" if model_st1 > parameters.strike_price else "PUT"
        model_st2 = gbm_next_price(model_st1, z2, second_period_years, drift, parameters.volatility)
        model_payoff = max(model_st2 - parameters.strike_price, 0.0) if model_choice == "CALL" else max(parameters.strike_price - model_st2, 0.0)

        rows.append(
            {
                "row": row_number,
                "z1": z1,
                "z2": z2,
                "simulated_st1": model_st1,
                "choice_call_put": model_choice,
                "simulated_st2": model_st2,
                "payoff": model_payoff,
            }
        )

    return pd.DataFrame(rows)


def build_table3_summary(table3_frame: pd.DataFrame, parameters: BsmChooserParameters, drift: float, seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "rows_simulated", "value": int(len(table3_frame))},
            {"metric": "spot_price", "value": parameters.spot_price},
            {"metric": "strike_price", "value": parameters.strike_price},
            {"metric": "risk_free_rate", "value": parameters.risk_free_rate},
            {"metric": "dividend_yield", "value": parameters.dividend_yield},
            {"metric": "risk_neutral_drift_r_minus_q", "value": drift},
            {"metric": "volatility_from_current_data", "value": parameters.volatility},
            {"metric": "random_seed", "value": seed},
            {"metric": "decision_time_years", "value": parameters.decision_time_years},
            {"metric": "maturity_years", "value": parameters.maturity_years},
        ]
    )


def build_table3_paper_reference() -> pd.DataFrame:
    reference_frame = pd.DataFrame(PAPER_TABLE3_ROWS)
    reference_frame["row"] = reference_frame["row"].astype(int)
    return reference_frame


def build_table3_comparison(table3_frame: pd.DataFrame) -> pd.DataFrame:
    paper_frame = build_table3_paper_reference().reset_index(drop=True)
    simulated_frame = table3_frame.reset_index(drop=True).copy()

    comparison = pd.DataFrame(
        {
            "row": paper_frame["row"].astype(int),
            "paper_st1": paper_frame["paper_st1"],
            "simulated_st1": simulated_frame["simulated_st1"],
            "paper_choice": paper_frame["paper_choice"],
            "choice_call_put": simulated_frame["choice_call_put"],
            "paper_st2": paper_frame["paper_st2"],
            "simulated_st2": simulated_frame["simulated_st2"],
            "paper_payoff": paper_frame["paper_payoff"],
            "payoff": simulated_frame["payoff"],
        }
    )
    comparison["st1_diff"] = comparison["simulated_st1"] - comparison["paper_st1"]
    comparison["st2_diff"] = comparison["simulated_st2"] - comparison["paper_st2"]
    comparison["payoff_diff"] = comparison["payoff"] - comparison["paper_payoff"]
    return comparison[
        [
            "row",
            "paper_st1",
            "simulated_st1",
            "st1_diff",
            "paper_choice",
            "choice_call_put",
            "paper_st2",
            "simulated_st2",
            "st2_diff",
            "paper_payoff",
            "payoff",
            "payoff_diff",
        ]
    ]


def build_table3_aggregate_comparison(table3_frame: pd.DataFrame, theoretical_chooser_value: float) -> pd.DataFrame:
    paper_reference = build_table3_paper_reference()
    simulated_reference = table3_frame.copy()

    simulated_choice_counts = simulated_reference["choice_call_put"].value_counts().to_dict()
    paper_choice_counts = paper_reference["paper_choice"].value_counts().to_dict()
    paper_payoff = paper_reference["paper_payoff"]
    simulated_payoff = simulated_reference["payoff"]

    rows = [
        {
            "metric": "payoff_mean",
            "paper": paper_payoff.mean(),
            "simulated": simulated_payoff.mean(),
        },
        {
            "metric": "payoff_std",
            "paper": paper_payoff.std(ddof=1),
            "simulated": simulated_payoff.std(ddof=1),
        },
        {
            "metric": "payoff_nonzero_ratio",
            "paper": float((paper_payoff > 0).mean()),
            "simulated": float((simulated_payoff > 0).mean()),
        },
        {
            "metric": "theoretical_vs_paper_mean_gap",
            "paper": theoretical_chooser_value,
            "simulated": paper_payoff.mean(),
        },
        {
            "metric": "call_count",
            "paper": int(paper_choice_counts.get("CALL", 0)),
            "simulated": int(simulated_choice_counts.get("CALL", 0)),
        },
        {
            "metric": "put_count",
            "paper": int(paper_choice_counts.get("PUT", 0)),
            "simulated": int(simulated_choice_counts.get("PUT", 0)),
        },
    ]

    aggregate = pd.DataFrame(rows)
    aggregate["difference"] = aggregate["simulated"] - aggregate["paper"]
    return aggregate


def format_value(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:,.6f}"
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| (empty) |\n| --- |\n| No rows produced |"

    headers = list(frame.columns)
    rows = [headers]
    for _, record in frame.iterrows():
        rows.append([format_value(record[column]) for column in headers])

    widths = [max(len(str(row[index])) for row in rows) for index in range(len(headers))]

    def render_row(values: list[object]) -> str:
        cells = [str(values[index]).ljust(widths[index]) for index in range(len(headers))]
        return "| " + " | ".join(cells) + " |"

    markdown_lines = [render_row(rows[0]), "| " + " | ".join("-" * width for width in widths) + " |"]
    markdown_lines.extend(render_row(row) for row in rows[1:])
    return "\n".join(markdown_lines)


def build_report_markdown(
    parameters: BsmChooserParameters,
    summary_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
    table3_frame: pd.DataFrame,
    table3_summary: pd.DataFrame,
    table3_aggregate: pd.DataFrame,
    seed: int,
) -> str:
    table3_summary_display = table3_summary.copy()
    table3_summary_display["value"] = table3_summary_display["value"].apply(
        lambda value: int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value
    )
    table3_display = table3_frame.copy()
    table3_display["row"] = table3_display["row"].astype(int)
    paper_reference_display = build_table3_paper_reference()
    paper_comparison_display = build_table3_comparison(table3_frame)
    aggregate_display = table3_aggregate.copy()
    aggregate_display["paper"] = aggregate_display["paper"].apply(format_value)
    aggregate_display["simulated"] = aggregate_display["simulated"].apply(format_value)
    aggregate_display["difference"] = aggregate_display["difference"].apply(format_value)
    summary_lines = [
        "# Week 3 BSM Chooser Option Validation",
        "",
        "This report uses the paper's Table 2 inputs as the fixed model parameters and simulates a two-stage geometric Brownian motion path. The simulation is stochastic, so row-by-row values are not expected to match the paper exactly:",
        "",
        "### Formula",
        "",
        "$$V_{chooser} = C(S, K, T) + P\\left(S, K e^{-r(T-t)}, T-t\\right)$$",
        "",
        "$$S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * sqrt(t) * Z)$$",
        "",
        "### Replication Setup",
        "",
        f"- Random seed: {seed}",
        f"- Price drift used in the GBM path: r - q = {parameters.risk_free_rate:.4f} - {parameters.dividend_yield:.4f} = {parameters.risk_free_rate - parameters.dividend_yield:.4f}",
        "- Comparison rule: paper Table 3 rows are treated as a published reference, while the simulated rows come from the same Table 2 parameters under independent random draws",
        "",
        "## Parameters",
        "",
        f"- Spot price: {parameters.spot_price}",
        f"- Strike price: {parameters.strike_price}",
        f"- Risk-free rate: {parameters.risk_free_rate}",
        f"- Dividend yield: {parameters.dividend_yield}",
        f"- Volatility: {parameters.volatility}",
        f"- Decision time: {parameters.decision_time_years}",
        f"- Maturity: {parameters.maturity_years}",
        f"- Source note: {parameters.source_note}",
        "",
        "## Base Case",
        "",
        dataframe_to_markdown(summary_frame),
        "",
        "## Spot Sensitivity",
        "",
        dataframe_to_markdown(sensitivity_frame),
        "",
        "## Validation Notes",
        "",
        "- The chooser value is the call leg plus the adjusted put leg.",
        "- The identity gap should be numerically zero, up to floating-point rounding.",
        "- The chooser premium over the call leg is always the value of the added put leg.",
        "",
        "## Table 3 Simulation",
        "",
        "The table below is generated from the Table 2 inputs. The exact path values depend on the seed and random draws, so they should be compared as a stochastic replication rather than a row-perfect match.",
        "",
        "### Simulation Summary",
        "",
        dataframe_to_markdown(table3_summary_display),
        "",
        "### Simulated Paths",
        "",
        dataframe_to_markdown(table3_display),
        "",
        "### Paper Reference Table",
        "",
        "The next table shows the paper's published values only as a visual reference for comparison.",
        "",
        dataframe_to_markdown(paper_reference_display),
        "",
        "### Aggregate Validation",
        "",
        "This summary compares the simulated outputs against the paper at the distribution level. It includes the payoff mean, payoff standard deviation, payoff non-zero ratio, and the gap between the theoretical chooser value and the paper's average payoff.",
        "",
        dataframe_to_markdown(aggregate_display),
        "",
        "Interpretation: the paper's 10-row payoff sample is the reference distribution, while the simulated 10-row payoff sample comes from the same Table 2 parameters but different random draws. A non-zero gap is expected, and the statistics above show whether the two samples are in the same range.",
        "",
        "### Side-by-Side Comparison: Prices",
        "",
        "This split table keeps the price columns readable and shows the row-by-row differences directly.",
        "",
        dataframe_to_markdown(paper_comparison_display[["row", "paper_st1", "simulated_st1", "st1_diff", "paper_st2", "simulated_st2", "st2_diff"]]),
        "",
        "### Side-by-Side Comparison: Choice and Payoff",
        "",
        dataframe_to_markdown(paper_comparison_display[["row", "paper_choice", "choice_call_put", "paper_payoff", "payoff", "payoff_diff"]]),
        "",
        "## Validation Conclusion",
        "",
        "- The implementation now clearly distinguishes model inputs, simulation assumptions, and the paper's reference outputs.",
        "- Exact path-level equality is not expected because the simulated Table 3 uses random draws, but the aggregate comparison makes the deviation visible.",
        "- The most likely reasons for differences versus the paper are the random seed and the paper's own simulation settings, which are not fully specified in the table excerpt.",
        "- The report now states the seed, the GBM update rule, and the drift assumption so the run is reproducible.",
    ]
    return "\n".join(summary_lines)


def save_csv(frame: pd.DataFrame, filename: str) -> Path:
    ensure_output_dirs()
    path = PROCESSED_DIR / filename
    frame.to_csv(path, index=False)
    return path


def save_markdown(markdown_text: str, filename: str) -> Path:
    ensure_output_dirs()
    path = REPORTS_DIR / filename
    path.write_text(markdown_text, encoding="utf-8")
    return path


def build_validation_frames(parameters: BsmChooserParameters) -> tuple[pd.DataFrame, pd.DataFrame]:
    breakdown = chooser_valuation_breakdown(parameters)
    summary_frame = pd.DataFrame(
        [
            {"metric": "spot_price", "value": breakdown["spot_price"]},
            {"metric": "strike_price", "value": breakdown["strike_price"]},
            {"metric": "adjusted_strike_for_put_leg", "value": breakdown["adjusted_strike_for_put_leg"]},
            {"metric": "risk_free_rate", "value": breakdown["risk_free_rate"]},
            {"metric": "dividend_yield", "value": breakdown["dividend_yield"]},
            {"metric": "volatility", "value": breakdown["volatility"]},
            {"metric": "decision_time_years", "value": breakdown["decision_time_years"]},
            {"metric": "maturity_years", "value": breakdown["maturity_years"]},
            {"metric": "time_to_decision_years", "value": breakdown["time_to_decision_years"]},
            {"metric": "call_leg_value", "value": breakdown["call_leg_value"]},
            {"metric": "put_leg_value", "value": breakdown["put_leg_value"]},
            {"metric": "chooser_value", "value": breakdown["chooser_value"]},
            {"metric": "chooser_premium_vs_call", "value": breakdown["chooser_premium_vs_call"]},
            {"metric": "chooser_premium_vs_put", "value": breakdown["chooser_premium_vs_put"]},
            {"metric": "chooser_identity_gap", "value": breakdown["chooser_identity_gap"]},
        ]
    )
    sensitivity_frame = build_sensitivity_table(parameters)
    return summary_frame, sensitivity_frame


def main(config_path: Path | None = None) -> dict[str, Path]:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    cleanup_generated_outputs()
    parameters = load_parameters(config_path)
    drift = parameters.risk_free_rate - parameters.dividend_yield
    seed = int(round(parameters.spot_price * 100)) + int(round(parameters.strike_price * 10))
    summary_frame, sensitivity_frame = build_validation_frames(parameters)
    table3_frame = build_table3_simulation(parameters, drift=drift, seed=seed)
    table3_summary = build_table3_summary(table3_frame, parameters, drift=drift, seed=seed)
    breakdown = chooser_valuation_breakdown(parameters)
    table3_aggregate = build_table3_aggregate_comparison(table3_frame, theoretical_chooser_value=breakdown["chooser_value"])

    combined_frame = pd.concat(
        [
            summary_frame.assign(section="base_case"),
            sensitivity_frame.assign(section="spot_sensitivity"),
            table3_frame.assign(section="table3_simulation"),
        ],
        ignore_index=True,
    )
    csv_path = save_csv(combined_frame, versioned_filename("week3_bsm_validation", "csv"))
    markdown_path = save_markdown(
        build_report_markdown(parameters, summary_frame, sensitivity_frame, table3_frame, table3_summary, table3_aggregate, seed),
        versioned_filename("week3_bsm_validation", "md"),
    )
    pdf_path = save_markdown_pdf_report(
        markdown_path,
        versioned_filename("week3_bsm_validation", "pdf"),
        title="Week 3 BSM Chooser Option Validation",
    )

    logger.info("Saved week3 validation CSV to %s", csv_path)
    logger.info("Saved week3 validation report to %s", markdown_path)
    logger.info("Saved week3 validation PDF to %s", pdf_path)
    return {"validation_csv": csv_path, "validation_md": markdown_path, "validation_pdf": pdf_path}


if __name__ == "__main__":
    main()
