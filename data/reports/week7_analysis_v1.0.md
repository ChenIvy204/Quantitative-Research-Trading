# Week 7 - Sensitivity analysis report

**Report date**: 20260619  |  **Pipeline version**: v1.0

## 1. Tool Configuration

- Model artifact: week6_approach2_neuralnetwork_v1.0.joblib
- Reference date: 2024-12-31
- Input feature count: 23
- Baseline model price: 24.423695
- Closed-form reference: 33.429823
- Monte Carlo reference: 33.602667

## 2. Extreme Scenario Tests

| scenario | model_price | delta | delta_pct | closed_form_quote | mc_quote |
| --- | --- | --- | --- | --- | --- |
| baseline | 24.423695 | 0 | 0 | 33.429823 | 33.602667 |
| volatility_spike_50pct | 25.944422 | 1.520727 | 0.062264 | 50.218133 | 49.108253 |
| rate_hike_2pct | 24.383972 | -0.039724 | -0.001626 | 33.48704 | 33.653419 |
| sentiment_shock | 28.007684 | 3.583989 | 0.146742 | 33.429823 | 33.602667 |
| combined_stress | 28.184373 | 3.760677 | 0.153977 | 50.116995 | 49.170447 |

## 3. Sensitivity Grid

| feature | value | model_price | delta | delta_pct |
| --- | --- | --- | --- | --- |
| vix | 12.145 | 23.137409 | -1.286287 | -0.052666 |
| vix | 15.615 | 23.886675 | -0.537021 | -0.021988 |
| vix | 17.35 | 24.423695 | 0 | 0 |
| vix | 19.085 | 25.074049 | 0.650353 | 0.026628 |
| vix | 26.025 | 29.027987 | 4.604292 | 0.188517 |
| sentiment_7d | 0 | 24.825158 | 0.401462 | 0.016437 |
| sentiment_7d | 0.005333 | 24.802069 | 0.378373 | 0.015492 |
| sentiment_7d | 0.105333 | 24.423695 | 0 | 0 |
| sentiment_7d | 0.205333 | 24.950535 | 0.526839 | 0.021571 |
| sentiment_7d | 0.355333 | 26.611322 | 2.187626 | 0.08957 |
| sentiment_20d | 0 | 28.578059 | 4.154364 | 0.170096 |
| sentiment_20d | 0.04583 | 27.292497 | 2.868801 | 0.11746 |
| sentiment_20d | 0.14583 | 24.423695 | 0 | 0 |
| sentiment_20d | 0.24583 | 23.190763 | -1.232933 | -0.050481 |
| sentiment_20d | 0.34583 | 24.649922 | 0.226227 | 0.009263 |
| r | 0.0358 | 25.324733 | 0.901038 | 0.036892 |
| r | 0.0458 | 24.423695 | 0 | 0 |
| r | 0.0558 | 24.398439 | -0.025256 | -0.001034 |
| r | 0.0658 | 24.383972 | -0.039724 | -0.001626 |
| q | 0.01419 | 26.400603 | 1.976908 | 0.080942 |
| q | 0.01919 | 24.423695 | 0 | 0 |
| q | 0.02419 | 23.034311 | -1.389385 | -0.056887 |
| T2 | 0.3 | 22.393345 | -2.03035 | -0.08313 |
| T2 | 0.4 | 23.36571 | -1.057985 | -0.043318 |
| T2 | 0.5 | 24.423695 | 0 | 0 |
| _7 more rows omitted_ |

## 4. SHAP Impact Summary

| feature | mean_abs_shap |
| --- | --- |
| vix_ma_ratio | 1.316436 |
| vix | 0.869999 |
| vix_jpm_corr_20d | 0.56082 |
| vol_ratio_20_60 | 0.346856 |
| sentiment_7d | 0.282197 |
| q | 0.276816 |
| vix_change_5d | 0.26088 |
| sentiment_20d | 0.244439 |
| vol_ratio_5_20 | 0.176465 |
| return_20d | 0.126048 |

## 5. Observations

- The pricing tool now exposes a direct quote, a closed-form reference, and a Monte Carlo reference for the same contract.
- Scenario tests include a 50% volatility spike, a 2% rate hike, a sentiment shock, and a combined stress case.
- Sensitivity grids are centered on the latest usable market row so the tool can be refreshed with new daily data.
