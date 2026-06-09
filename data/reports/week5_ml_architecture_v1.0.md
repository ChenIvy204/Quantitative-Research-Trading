# Week 5 – Machine Learning Model Architecture Design

**Report date**: 20260609  |  **Pipeline version**: v1.0

> This is the active Week 5 report for the current run. Older dated Week 5
> report files in `data/reports/` are treated as superseded outputs.

---

## 1. Executive Summary

This document describes the machine learning architecture designed and
implemented in Week 5 of the Quantitative Research & Trading project.
Two complementary approaches price chooser options on JPM stock:

- **Approach 1 (ML + chooser closed form)**: ML models predict 20-day
    forward realised volatility; the predicted σ is fed into the chooser
    pricing formula.
- **Approach 2 (End-to-End)**: ML models directly map chooser contract
    parameters and market features to chooser prices.

---

## 2. Problem Formulation

### Approach 1

$$\hat{V}_{\text{chooser}} = C\!\left(S, K, T_2, r, q, \hat{\sigma}_{\text{ML}}\right) + P\!\left(S, K e^{-r(T_2-T_1)}, T_2-T_1, r, q, \hat{\sigma}_{\text{ML}}\right)$$

where $\hat{\sigma}_{\text{ML}}$ is the ML-predicted 20-day forward
realised volatility.

**Target**: $\sigma_{\text{fwd,20d}}[t] = \sqrt{252} \cdot
\text{std}\!\left(\ln\frac{S_{t+i}}{S_{t+i-1}}\right)_{i=1}^{20}$

### Approach 2

$$\hat{V}_{\text{chooser}} = f_{\theta}\!\left(S,\,K,\,T_1,\,T_2,\,r,\,q,\,m,\,\mathbf{x}_{\text{market}}\right)$$

where $f_{\theta}$ is a trained ML model and $\mathbf{x}_{\text{market}}$
is the market feature vector.

**Target**: chooser price benchmarked with Monte Carlo-valued call/put legs.

---

## 3. Data & Feature Engineering

### 3.1 Raw Data Sources

| Source | Description | Frequency |
|--------|-------------|-----------|
| `yahoo_jpm_2018_2024.csv` | JPM daily close prices | Daily |
| `fred_DGS10_2018_2024.csv` | US 10-yr Treasury yield | Daily |
| `fred_VIXCLS_2018_2024.csv` | CBOE VIX index | Daily |
| `jpm_dividends_2018_2024.csv` | JPM dividend payments | Per event |
| `alphavantage_news_jpm_2018_2024.csv` | News sentiment scores | Per article |

### 3.2 Market Features (Approach 1 – Volatility Dataset)

All features are strictly **backward-looking** to prevent look-ahead bias.
Dataset: 1,681 trading days.

  - `hist_vol_5d`
  - `hist_vol_10d`
  - `hist_vol_20d`
  - `hist_vol_40d`
  - `hist_vol_60d`
  - `vol_ratio_5_20`
  - `vol_ratio_20_60`
  - `vol_20d_change`
  - `return_1d`
  - `return_5d`
  - `return_20d`
  - `price_to_ma_20d`
  - `price_to_ma_60d`
  - `vix`
  - `vix_change_5d`
  - `vix_ma_ratio`
  - `vix_jpm_corr_20d`
  - `r`
  - `q`
  - `sentiment_7d`
  - `sentiment_20d`
  - `news_count_7d`
  - `drawdown_20d`

### 3.3 Features (Approach 2 – Chooser Pricing Dataset)

All market features above, plus chooser-specific parameters.
Dataset: 17,850+ rows (daily dates × chooser contracts).

  - `return_1d`
  - `return_5d`
  - `return_20d`
  - `price_to_ma_20d`
  - `price_to_ma_60d`
  - `vix`
  - `vix_change_5d`
  - `vix_ma_ratio`
  - `vix_jpm_corr_20d`
  - `r`
  - `q`
  - `sentiment_7d`
  - `sentiment_20d`
  - `news_count_7d`
  - `drawdown_20d`
  - `S`
  - `K`
  - `moneyness`
  - `T1`
  - `T2`

### 3.4 Target Variables

| Approach | Target | Description |
|----------|--------|-------------|
| 1 | `fwd_vol_20d` | Annualised realised vol over next 20 trading days |
| 2 | `chooser_price` | Chooser price using Monte Carlo benchmark |

---

## 4. Time-Series Validation Framework

Data is split **chronologically** (never randomly) to prevent look-ahead bias.
All features are scaled using `RobustScaler` fitted **only** on the training set.

| Split | Date Range | Fraction |
|-------|-----------|----------|
| Train | 2018-03-29 → 2022-11-28 | 70% |
| Validation | 2022-11-29  → 2023-11-29 | 15% |
| Test | 2023-11-30  → 2024-12-02 | 15% |

---

## 5. Model Architectures

### 5.1 Approach 1 – ML Volatility Prediction

#### Random Forest
- `n_estimators=300`, `max_depth=8`, `min_samples_leaf=5`
- Preprocessing: RobustScaler
- Input: 23 market features (1 row per trading day)

#### XGBoost
- `n_estimators=500`, `max_depth=5`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- Preprocessing: RobustScaler

#### LSTM
  - Architecture: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)
  - Lookback window: 20 trading days
  - Optimizer: Adam | Loss: MSE | Early stopping (patience=10)
  - Input shape: (batch, 20, 23)

### 5.2 Approach 2 – End-to-End Supervised Chooser Pricing

#### Linear Regression
- Standard OLS with intercept, Preprocessing: RobustScaler
- Input: 20 features (market + chooser parameters)

#### XGBoost
- `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`

#### Neural Network (MLP)
- Architecture: Dense(128) → ReLU → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1)
- Optimizer: Adam (`lr=1e-3`), early stopping (`patience` via `validation_fraction=0.1`)
- `max_iter=500`

---

## 6. Performance Summary (Test Set)

### 6.1 Approach 1 – Volatility Prediction

| Model | Vol MAE | Vol RMSE | Option MAE | Option RMSE |
|-------|---------|----------|------------|-------------|
| RandomForest | 0.06895 | 0.10087 | 8.2373 | 11.6896 |
| XGBoost | 0.07625 | 0.10392 | 7.9809 | 11.5073 |
| LSTM | 0.05985 | 0.08653 | 7.8128 | 12.2689 |

*Vol MAE/RMSE: annualised vol units. Chooser MAE/RMSE: USD.*

### 6.2 Approach 2 – End-to-End Chooser Pricing

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|----------|-----|
| LinearRegression | 14.1912 | 16.9857 | -0.7884 |
| XGBoost | 11.1124 | 14.3564 | -0.2776 |
| NeuralNetwork | 8.0512 | 9.3497 | 0.4581 |

*Target: chooser price (hist_vol_20d as σ). Evaluation grid: 2T1 × 3T2 × 3K.*

---

## 7. Limitations & Recommended Next Steps

### Current Limitations

#### Data & Targets
1. **No market-implied vol**: Targets are derived from historical/closed-form
    prices, not actual market option quotes. This is a fundamental constraint:
    Approach 2 learns to predict a synthetic chooser surface, not a real market
    surface. High R² values (e.g., MLP R²=0.89) reflect model fit to the
    synthetic label, NOT market predictability.
2. **Synthetic chooser labels (Approach 2)**: Targets are computed as
    `chooser_price(S, K, T1, T2, r, q, hist_vol_match(T2))`. Since this is a
    deterministic closed-form calculation, linear models and neural networks
    with sufficient capacity can still fit the surface, but the direct
    historical-vol feature leakage into the input set has been removed.
3. **Volatility term-structure matching**: Vol used for each maturity is now
    matched to option T2, preventing term-structure mismatch. The pricing
    features themselves exclude all historical vol windows and vol-derived
    ratios, so Approach 2 must learn from price/momentum/VIX/rate/sentiment
    signals only.

#### Model Architecture
4. **Simplified chooser grid**: 2 decision times × 3 maturities × 3 moneyness levels.
5. **Static features**: No real-time microstructure data (bid-ask, volume).
6. **No market data feedback**: Models do not adjust based on actual prices
    observed at decision time T1.

### Model-Specific Notes

#### Approach 2 – Route 2 Pricing
- **LinearRegression & MLP**: R² should be materially lower now that all
    historical-vol input leakage has been removed. Any remaining fit comes from
    genuine market-state variables rather than a direct vol proxy.
- **XGBoost/GradientBoosting**: Lower R² (0.40-0.50) may indicate that tree
  models struggle with the smooth closed-form surface; alternatively, the
  current feature set lacks sufficient flexibility for tree splits. This is
  NOT a sign of data leakage or feature engineering failure—it reflects
  model-surface mismatch.

### Recommended Next Steps (Week 6+)
1. Incorporate implied volatility data for more realistic targets.
2. Add Greeks (delta, gamma, vega) as engineered input features.
3. Implement Bayesian hyperparameter optimisation.
4. Extend LSTM to multi-step ahead vol forecasting.
5. Build ensemble that combines Approach 1 and Approach 2 predictions.
6. Evaluate on 2025+ data for out-of-sample performance.
7. **Collect real market chooser prices** and retrain Approach 2 models
   on actual traded prices instead of synthetic targets.

---

*Generated by `week5_ml_models.py` | v1.0 | 20260609*
