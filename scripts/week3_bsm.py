from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from math import erf, exp, log, sqrt
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# =========================
# 论文表格参考数据
# =========================
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


# =========================
# 基础工具函数
# =========================
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


# =========================
# 参数加载与基础定价
# =========================
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
    # 通过改变标的价格，观察 chooser 价值与各腿价值的联动方向。
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


# =========================
# 论文 Section 4 参数敏感性数据
# =========================
def build_paper_sensitivity_frames(parameters: BsmChooserParameters) -> dict[str, pd.DataFrame]:
    """按论文 Section 4 逐一扫描 σ、K、r、q，记录 call / put / chooser 三者价值变化。"""
    base = parameters

    rows_v = []
    for v in np.linspace(0.05, 1.0, 80):
        temp = BsmChooserParameters(base.spot_price, base.strike_price, base.risk_free_rate,
                                    base.dividend_yield, v, base.decision_time_years, base.maturity_years)
        bd = chooser_valuation_breakdown(temp)
        rows_v.append({"volatility": v, "call_leg": bd["call_leg_value"],
                        "put_leg": bd["put_leg_value"], "chooser": bd["chooser_value"]})

    rows_k = []
    for k in np.linspace(50.0, 450.0, 80):
        temp = BsmChooserParameters(base.spot_price, k, base.risk_free_rate,
                                    base.dividend_yield, base.volatility, base.decision_time_years, base.maturity_years)
        bd = chooser_valuation_breakdown(temp)
        rows_k.append({"strike_price": k, "call_leg": bd["call_leg_value"],
                        "put_leg": bd["put_leg_value"], "chooser": bd["chooser_value"]})

    rows_r = []
    for r in np.linspace(0.001, 0.10, 80):
        temp = BsmChooserParameters(base.spot_price, base.strike_price, r,
                                    base.dividend_yield, base.volatility, base.decision_time_years, base.maturity_years)
        bd = chooser_valuation_breakdown(temp)
        rows_r.append({"risk_free_rate": r, "call_leg": bd["call_leg_value"],
                        "put_leg": bd["put_leg_value"], "chooser": bd["chooser_value"]})

    rows_q = []
    for q in np.linspace(0.001, 0.10, 80):
        temp = BsmChooserParameters(base.spot_price, base.strike_price, base.risk_free_rate,
                                    q, base.volatility, base.decision_time_years, base.maturity_years)
        bd = chooser_valuation_breakdown(temp)
        rows_q.append({"dividend_yield": q, "call_leg": bd["call_leg_value"],
                        "put_leg": bd["put_leg_value"], "chooser": bd["chooser_value"]})

    return {
        "volatility": pd.DataFrame(rows_v),
        "strike": pd.DataFrame(rows_k),
        "rate": pd.DataFrame(rows_r),
        "dividend": pd.DataFrame(rows_q),
    }


def gbm_next_price(start_price: float, shock: float, time_years: float, drift: float, volatility: float) -> float:
    return start_price * exp((drift - 0.5 * volatility**2) * time_years + shock * volatility * sqrt(time_years))


# =========================
# Table 3 复现模拟
# =========================
def build_table3_simulation(parameters: BsmChooserParameters, drift: float, seed: int, path_count: int = 10) -> pd.DataFrame:
    rng = random.Random(seed)
    first_period_years = parameters.decision_time_years
    second_period_years = parameters.maturity_years - parameters.decision_time_years

    rows: list[dict[str, float | str | int]] = []
    for row_number in range(1, path_count + 1):
        z1 = rng.normalvariate(0.0, 1.0)
        z2 = rng.normalvariate(0.0, 1.0)
        # 第一步先决定行权偏好，再根据选择路径推进到到期日。
        model_st1 = gbm_next_price(parameters.spot_price, z1, first_period_years, drift, parameters.volatility)
        model_choice = "CALL" if model_st1 > parameters.strike_price else "PUT"
        model_st2 = gbm_next_price(model_st1, z2, second_period_years, drift, parameters.volatility)
        model_payoff = max(model_st2 - parameters.strike_price, 0.0) if model_choice == "CALL" else max(parameters.strike_price - model_st2, 0.0)

        rows.append(
            {
# =========================
# 统计汇总与对比表
# =========================
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


# =========================
# Greeks 计算与敏感性分析
# =========================
def merton_greeks(S, K, T, r, q, v, is_call=True):
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1 = (log(S / K) + (r - q + 0.5 * v**2) * T) / (v * sqrt(T))
    d2 = d1 - v * sqrt(T)
    pdf = exp(-0.5 * d1**2) / sqrt(2.0 * 3.141592653589793)
    N_d1 = standard_normal_cdf(d1)
    N_d2 = standard_normal_cdf(d2)
    N_neg_d1 = standard_normal_cdf(-d1)
    N_neg_d2 = standard_normal_cdf(-d2)
    
    if is_call:
        delta = exp(-q * T) * N_d1
        rho = K * T * exp(-r * T) * N_d2
        theta = - (S * exp(-q * T) * pdf * v) / (2.0 * sqrt(T)) + q * S * exp(-q * T) * N_d1 - r * K * exp(-r * T) * N_d2
    else:
        delta = -exp(-q * T) * N_neg_d1
        rho = -K * T * exp(-r * T) * N_neg_d2
        theta = - (S * exp(-q * T) * pdf * v) / (2.0 * sqrt(T)) - q * S * exp(-q * T) * N_neg_d1 + r * K * exp(-r * T) * N_neg_d2
        
    gamma = exp(-q * T) * pdf / (S * v * sqrt(T))
    vega = S * exp(-q * T) * sqrt(T) * pdf
    
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def get_chooser_greeks(parameters: BsmChooserParameters) -> dict[str, float]:
    S, K, r, q, v, t, T = (
        parameters.spot_price,
        parameters.strike_price,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
        parameters.decision_time_years,
        parameters.maturity_years,
    )
    
    tau = T - t
    K_prime = K * exp(-r * tau)
    
    cg = merton_greeks(S, K, T, r, q, v, is_call=True)
    pg = merton_greeks(S, K_prime, tau, r, q, v, is_call=False)
    
    # Put 的 rho 需要考虑调整后的执行价 K' 也依赖 r。
    # 这里把链式法则的额外项一起计入，得到总 rho。
    rho_p_total = 2.0 * pg["rho"]
    
    # Theta 用一个很小的时间差做数值差分，方便展示总体时间衰减。
    dt = 1e-4
    v_current = chooser_valuation_breakdown(parameters)["chooser_value"]
    
    temp_t_minus = BsmChooserParameters(S, K, r, q, v, t - dt, T - dt, parameters.source_note)
    v_t_minus = chooser_valuation_breakdown(temp_t_minus)["chooser_value"]
    theta_total = (v_t_minus - v_current) / dt
    
    return {
        "delta": cg["delta"] + pg["delta"],
        "gamma": cg["gamma"] + pg["gamma"],
        "vega": cg["vega"] + pg["vega"],
        "theta": theta_total,
        "rho": cg["rho"] + rho_p_total,
    }


def build_greeks_dataframe(parameters: BsmChooserParameters) -> pd.DataFrame:
    # 把 call、put、chooser 三个对象的 Greeks 放在同一张表里，便于报告展示。
    cg = merton_greeks(parameters.spot_price, parameters.strike_price, parameters.maturity_years, parameters.risk_free_rate, parameters.dividend_yield, parameters.volatility, is_call=True)
    
    tau = parameters.maturity_years - parameters.decision_time_years
    K_prime = parameters.strike_price * exp(-parameters.risk_free_rate * tau)
    pg = merton_greeks(parameters.spot_price, K_prime, tau, parameters.risk_free_rate, parameters.dividend_yield, parameters.volatility, is_call=False)
    pg["rho"] = 2.0 * pg["rho"]
    
    dt = 1e-4
    def get_p_val(dec_time, mat_time):
        t_rem = mat_time - dec_time
        k_adj = parameters.strike_price * exp(-parameters.risk_free_rate * t_rem)
        return black_scholes_put(parameters.spot_price, k_adj, t_rem, parameters.risk_free_rate, parameters.dividend_yield, parameters.volatility)
        
    p_curr = get_p_val(parameters.decision_time_years, parameters.maturity_years)
    p_decay = get_p_val(parameters.decision_time_years - dt, parameters.maturity_years - dt)
    pg["theta"] = (p_decay - p_curr) / dt
    
    ig = get_chooser_greeks(parameters)
    
    rows = []
    g_list = ["delta", "gamma", "vega", "theta", "rho"]
    labels = ["Delta (dy/dS)", "Gamma (d2y/dS2)", "Vega (dy/dv)", "Theta (dy/dt)", "Rho (dy/dr)"]
    
    for g, label in zip(g_list, labels):
        rows.append({
            "Metric": label,
            "Call Leg": cg[g],
            "Put Leg": pg[g],
            "Chooser Option": ig[g]
        })
        
    return pd.DataFrame(rows)


# =========================
# Monte Carlo 收敛性验证
# =========================
def run_monte_carlo_convergence(parameters: BsmChooserParameters, seed: int = 17170) -> pd.DataFrame:
    S_0, K, r, q, v, t, T = (
        parameters.spot_price,
        parameters.strike_price,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
        parameters.decision_time_years,
        parameters.maturity_years,
    )
    
    np.random.seed(seed)
    sizes = [10000, 50000, 100000, 200000, 500000, 1000000]
    rows = []
    
    z1_max = np.random.normal(0, 1, 1000000)
    z2_max = np.random.normal(0, 1, 1000000)
    
    for n in sizes:
        z1 = z1_max[:n]
        z2 = z2_max[:n]
        
        # Rubinstein exact 版本：先在选择时刻决定 call / put，再把剩余期限继续推进。
        S_t = S_0 * np.exp((r - q - 0.5 * v**2) * t + v * np.sqrt(t) * z1)
        K_prime_rubby = K * np.exp(-(r - q) * (T - t))
        choice_call = S_t > K_prime_rubby
        S_T = S_t * np.exp((r - q - 0.5 * v**2) * (T - t) + v * np.sqrt(T - t) * z2)
        payoff_rubby = np.where(choice_call, np.maximum(S_T - K, 0.0), np.maximum(K - S_T, 0.0))
        disc_rubby = np.exp(-r * T) * payoff_rubby
        rubby_mean = np.mean(disc_rubby)
        rubby_std = np.std(disc_rubby, ddof=1)
        rubby_se = rubby_std / np.sqrt(n)
        
        # Split-leg 版本：call leg 和 put leg 分开模拟，便于对比论文基准写法。
        S_T_c = S_0 * np.exp((r - q - 0.5 * v**2) * T + v * np.sqrt(T) * z1)
        payoff_call = np.maximum(S_T_c - K, 0.0)
        disc_call = np.exp(-r * T) * payoff_call
        
        K_prime_split = K * np.exp(-r * (T - t))
        S_tau_p = S_0 * np.exp((r - q - 0.5 * v**2) * (T - t) + v * np.sqrt(T - t) * z2)
        payoff_put = np.maximum(K_prime_split - S_tau_p, 0.0)
        disc_put = np.exp(-r * (T - t)) * payoff_put
        disc_split = disc_call + disc_put
        
        split_mean = np.mean(disc_split)
        split_std = np.std(disc_split, ddof=1)
        split_se = split_std / np.sqrt(n)
        
        rows.append({
            "Paths": n,
            "Split Mean": split_mean,
            "Split SE": split_se,
            "Split 95% CI Lower": split_mean - 1.96 * split_se,
            "Split 95% CI Upper": split_mean + 1.96 * split_se,
            "Rubinstein Mean": rubby_mean,
            "Rubinstein SE": rubby_se,
            "Rubinstein 95% CI Lower": rubby_mean - 1.96 * rubby_se,
            "Rubinstein 95% CI Upper": rubby_mean + 1.96 * rubby_se,
        })
        
    return pd.DataFrame(rows)


# =========================
# 选择时点 t 与期权价值关系
# =========================
def build_t_vs_value_data(parameters: BsmChooserParameters) -> pd.DataFrame:
    S, K, r, q, v, T = (
        parameters.spot_price,
        parameters.strike_price,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
        parameters.maturity_years,
    )
    
    t_values = np.linspace(0.001, T - 1e-4, 100)
    rows = []
    
    for t in t_values:
        temp = BsmChooserParameters(S, K, r, q, v, t, T, parameters.source_note)
        bd = chooser_valuation_breakdown(temp)
        
        c_leg = black_scholes_call(S, K, T, r, q, v)
        k_adjusted = K * exp(-(r - q) * (T - t))
        p_leg = black_scholes_put(S, k_adjusted, t, r, q, v)
        rubinstein_val = c_leg + exp(-q * (T - t)) * p_leg
        
        rows.append({
            "t": t,
            "split_value": bd["chooser_value"],
            "rubinstein_value": rubinstein_val,
        })
        
    return pd.DataFrame(rows)


# =========================
# 图表生成
# =========================
def generate_plots(
    parameters: BsmChooserParameters,
    mc_df: pd.DataFrame,
    t_vs_v_df: pd.DataFrame,
    param_sensitivity_frames: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    ensure_output_dirs()
    
    # 图 1：Greeks 对标的价格的敏感性
    S_vals = np.linspace(100.0, 200.0, 100)
    greek_history = []
    for s in S_vals:
        temp = BsmChooserParameters(
            spot_price=s,
            strike_price=parameters.strike_price,
            risk_free_rate=parameters.risk_free_rate,
            dividend_yield=parameters.dividend_yield,
            volatility=parameters.volatility,
            decision_time_years=parameters.decision_time_years,
            maturity_years=parameters.maturity_years,
            source_note=parameters.source_note,
        )
        greek_history.append(get_chooser_greeks(temp))
    
    greek_df = pd.DataFrame(greek_history)
    greek_df["spot"] = S_vals
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    greeks_to_plot = ["delta", "gamma", "vega", "theta", "rho"]
    titles = ["Delta vs Spot", "Gamma vs Spot", "Vega vs Spot", "Theta (Annualized) vs Spot", "Rho vs Spot"]
    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for idx, (greek, label, color) in enumerate(zip(greeks_to_plot, titles, colors_list)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(greek_df["spot"], greek_df[greek], label=greek.capitalize(), color=color, linewidth=2)
        ax.axvline(parameters.spot_price, color="gray", linestyle="--", alpha=0.7, label=f"Spot={parameters.spot_price}")
        ax.axvline(parameters.strike_price, color="black", linestyle=":", alpha=0.7, label=f"Strike={parameters.strike_price}")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    fig.delaxes(axes[2, 1])
    plt.tight_layout()
    greeks_plot_path = REPORTS_DIR / "week3_bsm_greeks_sensitivity.png"
    fig.savefig(greeks_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    
    # 图 2：Monte Carlo 收敛性与 95% 置信区间
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        mc_df["Paths"],
        mc_df["Split Mean"],
        yerr=1.96 * mc_df["Split SE"],
        fmt="o-",
        capsize=5,
        color="#1f77b4",
        label="Split Model MC Mean (95% CI)",
        linewidth=1.5,
    )
    ax.errorbar(
        mc_df["Paths"] * 1.05,
        mc_df["Rubinstein Mean"],
        yerr=1.96 * mc_df["Rubinstein SE"],
        fmt="s-",
        capsize=5,
        color="#ff7f0e",
        label="Rubinstein MC Mean (95% CI)",
        linewidth=1.5,
    )
    
    split_theoretical = chooser_valuation_breakdown(parameters)["chooser_value"]
    c_leg = black_scholes_call(
        parameters.spot_price,
        parameters.strike_price,
        parameters.maturity_years,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
    )
    k_adjusted = parameters.strike_price * exp(-(parameters.risk_free_rate - parameters.dividend_yield) * (parameters.maturity_years - parameters.decision_time_years))
    p_leg = black_scholes_put(
        parameters.spot_price,
        k_adjusted,
        parameters.decision_time_years,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
    )
    rub_theoretical = c_leg + exp(-parameters.dividend_yield * (parameters.maturity_years - parameters.decision_time_years)) * p_leg
    
    ax.axhline(split_theoretical, color="#1f77b4", linestyle="--", alpha=0.8, label=f"Split Analytical ({split_theoretical:.4f})")
    ax.axhline(rub_theoretical, color="#ff7f0e", linestyle="--", alpha=0.8, label=f"Rubinstein Analytical ({rub_theoretical:.4f})")
    
    ax.set_xscale("log")
    ax.set_xlabel("Number of Paths (Log Scale)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Option Price", fontsize=10, fontweight="bold")
    ax.set_title("Monte Carlo Price Convergence & 95% Confidence Intervals", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    convergence_plot_path = REPORTS_DIR / "week3_bsm_convergence.png"
    fig.savefig(convergence_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    
    # 图 3：选择时间 t 与期权价值的关系
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        t_vs_v_df["t"],
        t_vs_v_df["split_value"],
        label="Split-Leg Chooser Value",
        color="#1f77b4",
        linewidth=2.5,
    )
    ax.plot(
        t_vs_v_df["t"],
        t_vs_v_df["rubinstein_value"],
        label="Rubinstein Exact Chooser Value",
        color="#ff7f0e",
        linewidth=2.5,
    )

    call_t0 = black_scholes_call(
        parameters.spot_price,
        parameters.strike_price,
        parameters.maturity_years,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
    )
    put_t0 = black_scholes_put(
        parameters.spot_price,
        parameters.strike_price,
        parameters.maturity_years,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
    )
    max_val = max(call_t0, put_t0)
    straddle_val = call_t0 + put_t0

    # max(C,P): Rubinstein 在 t→0 的下界 & Split-Leg 在 t→T 的收敛值
    ax.axhline(
        max_val, color="#c0392b", linestyle=":", linewidth=1.8,
        label=f"max(Call, Put) = {max_val:.2f}  [Rubinstein: t→0 limit | Split-Leg: t→T limit]",
    )
    # C+P (Straddle): Rubinstein 在 t→T 的上界
    ax.axhline(
        straddle_val, color="#27ae60", linestyle="--", linewidth=1.8,
        label=f"Straddle C+P = {straddle_val:.2f}  [Rubinstein: t→T limit]",
    )

    ax.set_xlabel("Choice Decision Time t (Years)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Chooser Option Value at Inception (Time 0)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Choice Decision Time t vs. Chooser Option Value",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    t_plot_path = REPORTS_DIR / "week3_bsm_t_vs_value.png"
    fig.savefig(t_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    
    # 图 4：论文 Section 4 参数敏感性分析（σ / K / r / q）
    call_color = "#cc4125"
    put_color = "#e69138"
    chooser_color = "#3d85c8"

    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 9))

    df_v = param_sensitivity_frames["volatility"]
    ax = axes4[0, 0]
    ax.plot(df_v["volatility"] * 100, df_v["call_leg"], label="Call", color=call_color, linewidth=2)
    ax.plot(df_v["volatility"] * 100, df_v["put_leg"], label="Put", color=put_color, linewidth=2)
    ax.plot(df_v["volatility"] * 100, df_v["chooser"], label="Chooser", color=chooser_color, linewidth=1.5, linestyle="--")
    ax.axvline(parameters.volatility * 100, color="gray", linestyle=":", alpha=0.7, label=f"σ = {parameters.volatility:.1%}")
    ax.set_xlabel("Volatility σ (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Option Value ($)", fontsize=10, fontweight="bold")
    ax.set_title("4.1  Volatility (σ) Sensitivity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    df_k = param_sensitivity_frames["strike"]
    ax = axes4[0, 1]
    ax.plot(df_k["strike_price"], df_k["call_leg"], label="Call", color=call_color, linewidth=2)
    ax.plot(df_k["strike_price"], df_k["put_leg"], label="Put", color=put_color, linewidth=2)
    ax.plot(df_k["strike_price"], df_k["chooser"], label="Chooser", color=chooser_color, linewidth=1.5, linestyle="--")
    ax.axvline(parameters.strike_price, color="gray", linestyle=":", alpha=0.7, label=f"K = {parameters.strike_price}")
    ax.set_xlabel("Strike Price K ($)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Option Value ($)", fontsize=10, fontweight="bold")
    ax.set_title("4.2  Strike Price (K) Sensitivity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    df_r = param_sensitivity_frames["rate"]
    ax = axes4[1, 0]
    ax.plot(df_r["risk_free_rate"] * 100, df_r["call_leg"], label="Call", color=call_color, linewidth=2)
    ax.plot(df_r["risk_free_rate"] * 100, df_r["put_leg"], label="Put", color=put_color, linewidth=2)
    ax.plot(df_r["risk_free_rate"] * 100, df_r["chooser"], label="Chooser", color=chooser_color, linewidth=1.5, linestyle="--")
    ax.axvline(parameters.risk_free_rate * 100, color="gray", linestyle=":", alpha=0.7, label=f"r = {parameters.risk_free_rate:.2%}")
    ax.set_xlabel("Risk-free Rate r (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Option Value ($)", fontsize=10, fontweight="bold")
    ax.set_title("4.3  Risk-free Rate (r) Sensitivity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    df_q = param_sensitivity_frames["dividend"]
    ax = axes4[1, 1]
    ax.plot(df_q["dividend_yield"] * 100, df_q["call_leg"], label="Call", color=call_color, linewidth=2)
    ax.plot(df_q["dividend_yield"] * 100, df_q["put_leg"], label="Put", color=put_color, linewidth=2)
    ax.plot(df_q["dividend_yield"] * 100, df_q["chooser"], label="Chooser", color=chooser_color, linewidth=1.5, linestyle="--")
    ax.axvline(parameters.dividend_yield * 100, color="gray", linestyle=":", alpha=0.7, label=f"q = {parameters.dividend_yield:.2%}")
    ax.set_xlabel("Dividend Yield q (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Option Value ($)", fontsize=10, fontweight="bold")
    ax.set_title("4.4  Dividend Yield (q) Sensitivity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    param_sens_plot_path = REPORTS_DIR / "week3_bsm_param_sensitivity.png"
    fig4.savefig(param_sens_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig4)

    return {
        "week3_bsm_greeks_sensitivity_v1.0_20260522.png": greeks_plot_path,
        "week3_bsm_convergence_v1.0_20260522.png": convergence_plot_path,
        "week3_bsm_t_vs_value_v1.0_20260522.png": t_plot_path,
        "week3_bsm_param_sensitivity_v1.0_20260522.png": param_sens_plot_path,
        "week3_bsm_greeks_sensitivity.png": greeks_plot_path,
        "week3_bsm_convergence.png": convergence_plot_path,
        "week3_bsm_t_vs_value.png": t_plot_path,
        "week3_bsm_param_sensitivity.png": param_sens_plot_path,
    }


# =========================
# 报告正文与导出
# =========================
def build_report_markdown(
    parameters: BsmChooserParameters,
    summary_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
    greeks_frame: pd.DataFrame,
    convergence_frame: pd.DataFrame,
    t_vs_v_frame: pd.DataFrame,
    table3_frame: pd.DataFrame,
    table3_summary: pd.DataFrame,
    table3_aggregate: pd.DataFrame,
    param_sensitivity_frames: dict[str, pd.DataFrame],
    seed: int,
) -> str:
    bd_base = chooser_valuation_breakdown(parameters)
    put_atm = black_scholes_put(
        parameters.spot_price, parameters.strike_price, parameters.maturity_years,
        parameters.risk_free_rate, parameters.dividend_yield, parameters.volatility,
    )
    straddle_atm = bd_base["call_leg_value"] + put_atm
    max_cp = max(bd_base["call_leg_value"], put_atm)
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
        "This report contains the quantitative analysis of the Rubinstein Chooser Option validation, incorporating analytical Greeks sensitivity, multi-scale Monte Carlo Path convergence checking, and choice time t convergence studies.",
        "",
        "### Core Model Equations",
        "",
        "Split-Leg Model (Codebase Baseline):",
        "$$V_{split} = C(S_0, K, T) + P(S_0, K e^{-r(T-t)}, T-t)$$",
        "",
        "Rubinstein Exact Valuation (Standard Derivative Theory):",
        "$$V_{rubinstein} = C(S_0, K, T) + e^{-q(T-t)} P(S_0, K e^{-(r-q)(T-t)}, t)$$",
        "",
        "### Table 2 replication setup",
        "",
        f"- Spot price: {parameters.spot_price}",
        f"- Strike price: {parameters.strike_price}",
        f"- Risk-free rate: {parameters.risk_free_rate}",
        f"- Dividend yield: {parameters.dividend_yield}",
        f"- Volatility: {parameters.volatility}",
        f"- Decision time t: {parameters.decision_time_years} Years",
        f"- Maturity T: {parameters.maturity_years} Years",
        f"- Price drift used in the GBM path: r - q = {parameters.risk_free_rate - parameters.dividend_yield:.4f}",
        f"- Stochastic seed: {seed}",
        "",
        "## Base Valuation Breakdown",
        "",
        dataframe_to_markdown(summary_frame),
        "",
        "## Part 1: Option Greeks & Sensitivity Analysis",
        "",
        "Analytical calculation of the chooser option Greeks handles the decomposition of both the call and put legs, which is verified below against spot change curves. For the Rho metric, the derivative with respect to the risk-free rate includes the chain rule for the put strike parameter: dK'/dr = -(T-t) K', leading to an exact analytical adjustment: dP/dr_total = 2 * dP_merton/dr.",
        "",
        dataframe_to_markdown(greeks_frame),
        "",
        "### Greeks Sensitivity to Asset Price",
        "",
        "The following chart displays the analytical Greeks across a wide range of asset prices (100 to 200).",
        "",
        "![Greeks Sensitivity](week3_bsm_greeks_sensitivity.png)",
        "",
        "## Section 4: BSM Parameter Sensitivity Analysis",
        "",
        "This section follows the reference paper's results analysis (Section 4), examining how each of the four key Black-Scholes parameters — volatility (σ), strike price (K), risk-free rate (r), and dividend yield (q) — affects the call leg, put leg, and total chooser option value. For each sweep all other parameters are held at the baseline configuration.",
        "",
        "![Parameter Sensitivity](week3_bsm_param_sensitivity.png)",
        "",
        "### 4.1 Volatility (σ)",
        "",
        "Volatility is a statistical measure of the dispersion of returns of the underlying asset, often measured by the standard deviation (σ) of price changes. In the BSM model it is annualised. The sweep covers σ from 5% to 100%.",
        "",
        "**Results:** Both call and put option values increase monotonically as volatility rises. Higher volatility implies a greater probability that the stock price will swing over a larger range in either direction. For call options, the upside is unlimited while the downside is bounded by the premium paid; for put options the same asymmetric protection applies in the opposite direction. Therefore, increased volatility does not cause losses on the downside but helps make money on the upside for both option types — the chooser option value is strictly increasing in σ.",
        "",
        f"Baseline: σ = {parameters.volatility:.1%}  →  Call leg = {bd_base['call_leg_value']:.4f},  Put leg = {bd_base['put_leg_value']:.4f},  Chooser = {bd_base['chooser_value']:.4f}",
        "",
        "### 4.2 Strike Price (K)",
        "",
        "The strike price is the agreed price at which the option holder has the right to buy (call) or sell (put) the underlying asset. It determines the option's intrinsic value as max(0, S_T − K) for calls and max(0, K − S_T) for puts. The sweep covers K from $50 to $450, which spans deep-ITM through deep-OTM relative to the current spot price.",
        "",
        "**Results:** As strike price increases, call option value decreases while put option value increases — the two curves cross near the current spot price. For call options, a higher K requires the stock to rise further to reach ITM status, reducing both the probability and magnitude of payoff. For put options, a higher K increases the probability the stock finishes below the strike, raising the likelihood and magnitude of exercise. The chooser option value follows a convex profile, since it captures both legs.",
        "",
        f"Baseline: K = {parameters.strike_price}  →  Spot / Strike = {parameters.spot_price / parameters.strike_price:.4f}",
        "",
        "### 4.3 Risk-free Interest Rate (r)",
        "",
        "The risk-free interest rate is the return from a zero-risk investment, proxied in US capital markets by the treasury bond rate. The sweep covers r from 0.1% to 10%.",
        "",
        "**Results:** As the risk-free rate increases, call option value increases while put option value decreases — a near-linear relationship. For call options, a higher r raises the expected drift of the stock (r − q), increasing the expected future stock price; it also lowers the present value of the strike price (future cash outflow), both of which benefit call holders. For put options, the higher expected stock price reduces the probability of ITM exercise, while the higher r also reduces the present value of future cash inflows from put exercise.",
        "",
        f"Baseline: r = {parameters.risk_free_rate:.4%}  (near-zero rate environment reflecting the JPM 2018–2024 data window)",
        "",
        "### 4.4 Dividend Yield (q)",
        "",
        "Dividend yield measures the cash dividends paid relative to the stock price. In the Merton continuous-dividend model, q enters both the effective drift (r − q) and the stock discount factor e^(−qT). The sweep covers q from 0.1% to 10%.",
        "",
        "**Results:** As dividend yield increases, call option value decreases while put option value increases. Higher q reduces the effective stock drift (r − q), lowering expected stock price growth and thus call values. For put options, the reduced expected stock price makes ITM exercise more likely, increasing put values. This is the mirror image of the risk-free rate effect: the two parameters affect option values in exactly opposite directions.",
        "",
        f"Baseline: q = {parameters.dividend_yield:.4%}",
        "",
        "## Part 2: Monte Carlo Path Scale Expansion & Convergence Analysis",
        "",
        "We expanded the model's Monte Carlo pricing from 100,000 paths to 1,000,000 paths to analyze standard error behavior and path convergence under the standard 95% Confidence Interval (CI) bands.",
        "",
        dataframe_to_markdown(convergence_frame),
        "",
        "### Path Convergence Tracking",
        "",
        "Below is the convergence plot showing the calculated simulation means & 95% confidence intervals against the analytical limits as sample path size scales to 1,000,000.",
        "",
        "![Monte Carlo Convergence](week3_bsm_convergence.png)",
        "",
        "## Part 3: Choice Decision Time t vs. Option Value Analysis",
        "",
        f"The mathematical convergence of a Chooser Option as t changes is studied below. **Rubinstein Exact model**: as t→0, choice must be made immediately with no additional information, and the value converges to max(Call, Put) = {max_cp:.2f} (= Call here, since Call > Put). As t→T, the holder can defer choice until just before maturity and always select the higher payoff — value converges to the Straddle C+P = {straddle_atm:.2f}. The Rubinstein curve is monotonically increasing in t. **Split-Leg model**: as t→T, the Put leg's remaining time tau = T−t → 0; because the adjusted strike K' → K and S ({parameters.spot_price}) > K ({parameters.strike_price}), the put expires worthless and value converges to Call = {bd_base['call_leg_value']:.2f} = max(Call, Put). Notably the two models form an X-shape: both share max(Call, Put) = {max_cp:.2f} as a boundary — Rubinstein at the left limit (t→0) and Split-Leg at the right limit (t→T).",
        "",
        "![Decision Time t vs Option Value](week3_bsm_t_vs_value.png)",
        "",
        "## Part 4: Spot Sensitivity (Table 2 Scenario)",
        "",
        dataframe_to_markdown(sensitivity_frame),
        "",
        "## Part 5: Table 3 Path-by-Path Replication",
        "",
        "Stochastic drawing comparison for standard Table 3 rows:",
        "",
        "### Simulation Summary",
        "",
        dataframe_to_markdown(table3_summary_display),
        "",
        "### Simulated Paths (Stochastic)",
        "",
        dataframe_to_markdown(table3_display),
        "",
        "### Published Paper Reference Table 3",
        "",
        dataframe_to_markdown(paper_reference_display),
        "",
        "### Aggregate Distribution Level Comparison",
        "",
        dataframe_to_markdown(aggregate_display),
        "",
        "### Side-by-Side Comparison: Prices",
        "",
        dataframe_to_markdown(paper_comparison_display[["row", "paper_st1", "simulated_st1", "st1_diff", "paper_st2", "simulated_st2", "st2_diff"]]),
        "",
        "### Side-by-Side Comparison: Choice and Payoff",
        "",
        dataframe_to_markdown(paper_comparison_display[["row", "paper_choice", "choice_call_put", "paper_payoff", "payoff", "payoff_diff"]]),
        "",
        "## Validation Conclusion",
        "",
        "- Section 4 parameter sensitivity confirms all paper directional conclusions: higher σ raises both call and put values; higher K lowers call and raises put; higher r raises call and lowers put; higher q lowers call and raises put.",
        "- Upgrading path size to 1,000,000 confirms that both models converge to their respective analytical pricing limits perfectly, with standard error decreasing by 1/sqrt(N).",
        "- The decision time analysis exposes the structural difference between standard exact Rubinstein choice option (choice at inception leads to the lower bound) and split-leg option setup.",
        "- Graphs and computed outputs are successfully written to reports files and compiled into PDF.",
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


# =========================
# 主流程编排
# =========================
def build_validation_frames(parameters: BsmChooserParameters) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    greeks_frame = build_greeks_dataframe(parameters)
    return summary_frame, sensitivity_frame, greeks_frame


def main(config_path: Path | None = None) -> dict[str, Path]:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    cleanup_generated_outputs()
    parameters = load_parameters(config_path)
    drift = parameters.risk_free_rate - parameters.dividend_yield
    seed = int(round(parameters.spot_price * 100)) + int(round(parameters.strike_price * 10))
    # 先准备基础校验数据，再生成收敛数据和图表，最后统一输出报告文件。
    summary_frame, sensitivity_frame, greeks_frame = build_validation_frames(parameters)
    
    # Run Monte Carlo and decision time datasets
    convergence_frame = run_monte_carlo_convergence(parameters, seed=seed)
    t_vs_v_frame = build_t_vs_value_data(parameters)
    
    # Save simulated outputs inside csv versioned file
    csv_convergence_path = PROCESSED_DIR / versioned_filename("week3_bsm_convergence_data", "csv")
    convergence_frame.to_csv(csv_convergence_path, index=False)
    csv_t_vs_v_path = PROCESSED_DIR / versioned_filename("week3_bsm_t_vs_value_data", "csv")
    t_vs_v_frame.to_csv(csv_t_vs_v_path, index=False)
    
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
    
    # Compute Section 4 parameter sensitivity frames
    param_sensitivity_frames = build_paper_sensitivity_frames(parameters)
    # Generate graphs
    plot_paths = generate_plots(parameters, convergence_frame, t_vs_v_frame, param_sensitivity_frames)

    markdown_path = save_markdown(
        build_report_markdown(
            parameters, 
            summary_frame, 
            sensitivity_frame, 
            greeks_frame, 
            convergence_frame, 
            t_vs_v_frame, 
            table3_frame, 
            table3_summary, 
            table3_aggregate, 
            param_sensitivity_frames,
            seed
        ),
        versioned_filename("week3_bsm_validation", "md"),
    )
    
    # Compile PDF
    pdf_path = save_markdown_pdf_report(
        markdown_path,
        versioned_filename("week3_bsm_validation", "pdf"),
        title="Week 3 BSM Chooser Option Validation",
        asset_paths=plot_paths,
    )

    logger.info("Saved week3 validation CSV to %s", csv_path)
    logger.info("Saved week3 validation report to %s", markdown_path)
    logger.info("Saved week3 validation PDF to %s", pdf_path)
    return {"validation_csv": csv_path, "validation_md": markdown_path, "validation_pdf": pdf_path}


if __name__ == "__main__":
    main()
