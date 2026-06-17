# Week 7 - Advanced Analysis & Tool Development

**Report date**: 20260618  |  **Pipeline version**: v1.0

## 1. Tool Configuration

- Model artifact: week6_approach2_xgboost_v1.0.joblib
- Reference date: 2024-12-31
- Input feature count: 23
- Baseline model price: 16.098978
- Closed-form reference: 33.429823
- Monte Carlo reference: 33.602667

## 2. Extreme Scenario Tests

| scenario | model_price | delta | delta_pct | closed_form_quote | mc_quote |
| --- | --- | --- | --- | --- | --- |
| baseline | 16.098978 | 0 | 0 | 33.429823 | 33.602667 |
| volatility_spike_50pct | 16.734989 | 0.636011 | 0.039506 | 33.429823 | 33.602667 |
| rate_hike_2pct | 16.098978 | 0 | 0 | 33.429823 | 33.602667 |
| sentiment_shock | 16.081207 | -0.017771 | -0.001104 | 33.429823 | 33.602667 |
| combined_stress | 17.38422 | 1.285242 | 0.079834 | 33.429823 | 33.602667 |

## 3. Sensitivity Grid

| feature | value | model_price | delta | delta_pct |
| --- | --- | --- | --- | --- |
| vix | 12.145 | 15.147018 | -0.95196 | -0.059132 |
| vix | 15.615 | 15.113082 | -0.985896 | -0.06124 |
| vix | 17.35 | 16.098978 | 0 | 0 |
| vix | 19.085 | 16.098978 | 0 | 0 |
| vix | 26.025 | 17.38422 | 1.285242 | 0.079834 |
| sentiment_7d | 0 | 16.098978 | 0 | 0 |
| sentiment_7d | 0.005333 | 16.098978 | 0 | 0 |
| sentiment_7d | 0.105333 | 16.098978 | 0 | 0 |
| sentiment_7d | 0.205333 | 16.098978 | 0 | 0 |
| sentiment_7d | 0.355333 | 16.098978 | 0 | 0 |
| sentiment_20d | 0 | 16.002396 | -0.096582 | -0.005999 |
| sentiment_20d | 0.04583 | 16.002396 | -0.096582 | -0.005999 |
| sentiment_20d | 0.14583 | 16.098978 | 0 | 0 |
| sentiment_20d | 0.24583 | 16.098978 | 0 | 0 |
| sentiment_20d | 0.34583 | 16.098978 | 0 | 0 |
| r | 0.0358 | 17.633144 | 1.534166 | 0.095296 |
| r | 0.0458 | 16.098978 | 0 | 0 |
| r | 0.0558 | 16.098978 | 0 | 0 |
| r | 0.0658 | 16.098978 | 0 | 0 |
| q | 0.01419 | 16.098978 | 0 | 0 |
| q | 0.01919 | 16.098978 | 0 | 0 |
| q | 0.02419 | 16.098978 | 0 | 0 |
| T2 | 0.3 | 16.098978 | 0 | 0 |
| T2 | 0.4 | 16.098978 | 0 | 0 |
| T2 | 0.5 | 16.098978 | 0 | 0 |
| _7 more rows omitted_ |

## 4. SHAP Impact Summary

| feature | mean_abs_shap |
| --- | --- |
| vix_ma_ratio | 0.696841 |
| vix | 0.46509 |
| S | 0.101307 |
| q | 0.064257 |
| vix_jpm_corr_20d | 0.033735 |
| vol_ratio_20_60 | 0.025271 |
| news_count_7d | 0.023548 |
| return_5d | 0.002082 |
| sentiment_20d | 0.00005 |
| sentiment_7d | 0 |

## 5. Observations

- The pricing tool now exposes a direct quote, a closed-form reference, and a Monte Carlo reference for the same contract.
- Scenario tests include a 50% volatility spike, a 2% rate hike, a sentiment shock, and a combined stress case.
- Sensitivity grids are centered on the latest usable market row so the tool can be refreshed with new daily data.
