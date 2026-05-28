# Week 4 – BSM Performance Benchmark Documentation

**Run date**: 20260528  |  **Pipeline version**: v1.0

This document records the Black-Scholes-Merton (BSM) model performance baseline
for JPM options pricing over the 2018–2024 period.  All figures here serve as
the reference point for any future model enhancements.

---

## 1. Benchmark Setup

| Parameter | Value |
|-----------|-------|
| Underlying asset | JPM (JPMorgan Chase) |
| Evaluation period | 2018-01-01 – 2024-12-31 |
| Sampling frequency | Monthly (month-start) |
| Maturities | [0.25, 0.5, 1.0] years |
| Moneyness levels (K/S) | [0.9, 1.0, 1.1] |
| Option types | Call, Put |
| Total observations | 1,008 |
| MC benchmark paths | 10,000 (seed=42) |
| Historical vol window | 20 trading days |
| Risk-free rate source | FRED DGS10 |
| Dividend yield | Trailing-twelve-month |

---

## 2. Headline Baseline Metrics

These are the official BSM baseline figures.  Future models must beat these
numbers to demonstrate improvement.

| Metric | Baseline value |
|--------|----------------|
| **Overall MAE** | **0.112135** |
| **Overall RMSE** | **0.174916** |
| Max absolute error | 2.106593 |
| Low-VIX MAE  (VIX < 20.0) | 0.087432 |
| Mid-VIX MAE  (20.0–30.0) | 0.137720 |
| **High-VIX MAE (VIX ≥ 30.0)** | **0.203184** |
| Low-VIX RMSE | 0.128842 |
| Mid-VIX RMSE | 0.183733 |
| High-VIX RMSE | 0.344156 |
| Sentiment–error Pearson corr | 0.215158 |

---

## 3. Full Breakdown by Group

All sub-group MAE / RMSE values for complete traceability:

```
          group    n      MAE     RMSE  max_abs_err
        overall 1008 0.112135 0.174916     2.106593
     regime=low  630 0.087432 0.128842     0.624741
  regime=medium  288 0.137720 0.183733     0.756761
    regime=high   90 0.203184 0.344156     2.106593
maturity=0.25yr  336 0.080286 0.120687     0.580171
 maturity=0.5yr  336 0.104617 0.155197     1.430548
 maturity=1.0yr  336 0.151502 0.230511     2.106593
      type=call  504 0.130680 0.209066     2.106593
       type=put  504 0.093590 0.132221     0.583721
  moneyness=0.9  336 0.108003 0.159285     0.756761
  moneyness=1.0  336 0.112518 0.159001     0.624741
  moneyness=1.1  336 0.115883 0.202814     2.106593
```

---

## 4. Key Limitations Identified

1. **High-volatility failure**: MAE in high-VIX regime (0.2032) is
   232% of the low-VIX baseline (0.0874).
   BSM underprices risk during market stress because it uses backward-looking
   historical volatility rather than forward-looking implied volatility.

2. **Maturity effect**: Error scales with maturity (T=1yr RMSE is materially
   larger than T=0.25yr), reflecting accumulated GBM path uncertainty that
   BSM's closed form cannot fully capture when σ is volatile.

3. **Sentiment gap**: Pearson correlation of 0.215158 between
   news sentiment and pricing error confirms BSM's inability to respond to
   information events.  Positive-sentiment days show higher mean error
   (0.12149) vs negative-sentiment days
   (0.072511), suggesting bullish news events
   drive larger deviations from the simulation benchmark.

---

## 5. Improvement Targets for Future Models

| Target | Current baseline | Goal |
|--------|-----------------|------|
| Overall MAE | 0.112135 | < 0.089708 (−20%) |
| High-VIX MAE | 0.203184 | < 0.152388 (−25%) |
| Sentiment correlation | 0.215158 | ≈ 0 (model absorbs sentiment) |
