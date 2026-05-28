# Week 4 – Model Validation Report: BSM Error Metrics

**Run date**: 20260528  |  **Pipeline version**: v1.0

## Methodology

The BSM analytical closed-form prices are treated as *model predictions*.
Monte Carlo (MC) simulation prices (N=10,000 paths, GBM under risk-neutral
measure, seed=42) serve as the independent benchmark ("actual prices").
MAE and RMSE measure how closely the BSM formula approximates the MC benchmark
across different market regimes.

Parameters are derived from JPM historical market data (2018–2024):
- **S**: JPM daily close price
- **σ**: 20-day rolling historical annualised volatility
- **r**: US 10-year Treasury yield (DGS10)
- **q**: trailing-twelve-month dividend yield
- **VIX regime**: Low < 20.0, Medium 20.0–30.0, High ≥ 30.0

Evaluation grid: 3 maturities × 3 moneyness levels × 2 option types,
sampled monthly → **1,008 pricing observations** over **56 dates**.

---

## 1. Overall Error Metrics

| Metric | Value |
|--------|-------|
| MAE (overall) | 0.112135 |
| RMSE (overall) | 0.174916 |
| Max |BSM − MC| | 2.106593 |
| Total observations | 1,008 |

These values quantify the numerical convergence gap between the BSM
analytical formula and the MC simulation benchmark.  Both are generated
under identical GBM assumptions, so deviations arise purely from MC
sampling variance (which decreases as ∝ 1/√N).

---

## 2. Error by VIX Regime (Failure Mode Analysis)

| Regime | MAE | RMSE | n |
|--------|-----|------|---|
| Low (VIX < 20.0) | 0.087432 | 0.128842 | 630 |
| Medium (20.0–30.0) | 0.137720 | 0.183733 | 288 |
| High (VIX ≥ 30.0) | 0.203184 | 0.344156 | 90 |

**Interpretation**: Higher VIX regimes produce larger absolute errors because
the MC payoff distribution widens with higher volatility, amplifying sampling
noise.  Under a fixed N=10,000 paths, the standard error of the MC estimate
scales as σ × √(T/N), so high-σ, long-T options show the largest gaps.

---

## 3. Error by Maturity and Option Type

```
          group   n      MAE     RMSE  max_abs_err
maturity=0.25yr 336 0.080286 0.120687     0.580171
 maturity=0.5yr 336 0.104617 0.155197     1.430548
 maturity=1.0yr 336 0.151502 0.230511     2.106593
```

```
    group   n     MAE     RMSE  max_abs_err
type=call 504 0.13068 0.209066     2.106593
 type=put 504 0.09359 0.132221     0.583721
```

Longer maturities accumulate more GBM variance, making MC estimates noisier and
increasing |BSM − MC|.  Calls and puts exhibit similar error levels due to
put-call parity symmetry.

---

## 4. Sentiment Impact Gap Analysis

| Metric | Value |
|--------|-------|
| Pearson corr(sentiment, mean |BSM−MC|) | 0.215158 |
| Spearman corr(sentiment, mean |BSM−MC|) | 0.251674 |
| Mean error on positive-sentiment days | 0.12149 |
| Mean error on negative-sentiment days | 0.072511 |

**Interpretation**: BSM does not incorporate sentiment as an input; sentiment is
a proxy for market-moving information events that the model structurally ignores.
A non-zero correlation between sentiment and |BSM − MC| indicates that, on days
with strong news signal, market-implied volatility may deviate from the backward-
looking historical volatility used in BSM, widening the BSM–MC gap.

---

## 5. Charts

![Error Time Series](week4_bsm_error_timeseries.png)

![Regime Boxplot](week4_bsm_regime_boxplot.png)

![Sentiment Scatter](week4_bsm_sentiment_scatter.png)
