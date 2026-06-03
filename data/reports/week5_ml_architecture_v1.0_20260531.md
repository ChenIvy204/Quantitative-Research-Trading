# Week 5 – Machine Learning Model Architecture Design

Run date: 20260531  |  Pipeline version: v1.0

---

## 1. Executive Summary

This document describes the machine learning architecture designed and
implemented in Week 5 of the Quantitative Research & Trading project.
The work follows two complementary paths for pricing European options on
JPM stock.

In the first path, the model predicts the next 20 trading days of realised
volatility, and that prediction is then passed into the Black-Scholes-Merton
pricing formula.

In the second path, the model learns the option price directly from the
market data and contract settings, without using an intermediate pricing
formula.

---

## 2. Problem Formulation

### Approach 1

The first approach estimates the 20-day forward realised volatility and
uses that estimate inside the Black-Scholes-Merton model to obtain the
option price. The target is the annualised realised volatility over the
next 20 trading days.

### Approach 2

The second approach predicts the option price directly from the market
feature set and contract variables such as underlying price, strike,
expiry, interest rate, dividend yield, moneyness, and call-or-put type.
Its target is the Black-Scholes-Merton price computed with 20-day
historical volatility.

---

## 3. Data & Feature Engineering

### 3.1 Raw Data Sources

| Source | Description | Frequency |
|--------|-------------|-----------|
| yahoo_jpm_2018_2024.csv | JPM daily close prices | Daily |
| fred_DGS10_2018_2024.csv | US 10-yr Treasury yield | Daily |
| fred_VIXCLS_2018_2024.csv | CBOE VIX index | Daily |
| jpm_dividends_2018_2024.csv | JPM dividend payments | Per event |
| alphavantage_news_jpm_2018_2024.csv | News sentiment scores | Per article |

### 3.2 Market Features (Approach 1 – Volatility Dataset)

All features are strictly backward-looking to prevent look-ahead bias.
Dataset: 1,681 trading days.

  - hist_vol_5d
  - hist_vol_20d
  - hist_vol_60d
  - vol_ratio_5_20
  - vol_ratio_20_60
  - vol_20d_change
  - return_1d
  - return_5d
  - return_20d
  - price_to_ma_20d
  - price_to_ma_60d
  - vix
  - vix_change_5d
  - vix_ma_ratio
  - vix_jpm_corr_20d
  - r
  - q
  - sentiment_7d
  - sentiment_20d
  - news_count_7d
  - drawdown_20d

### 3.3 Features (Approach 2 – Option Pricing Dataset)

All market features above, plus option-specific parameters.
Dataset: 21,420+ rows (daily dates × 18 option contracts each).

  - hist_vol_5d
  - hist_vol_20d
  - hist_vol_60d
  - vol_ratio_5_20
  - vol_ratio_20_60
  - vol_20d_change
  - return_1d
  - return_5d
  - return_20d
  - price_to_ma_20d
  - price_to_ma_60d
  - vix
  - vix_change_5d
  - vix_ma_ratio
  - vix_jpm_corr_20d
  - r
  - q
  - sentiment_7d
  - sentiment_20d
  - news_count_7d
  - drawdown_20d
  - S
  - K
  - moneyness
  - T
  - is_call

### 3.4 Target Variables

| Approach | Target | Description |
|----------|--------|-------------|
| 1 | fwd_vol_20d | Annualised realised volatility over the next 20 trading days |
| 2 | bsm_price | Black-Scholes-Merton price using 20-day historical volatility |

---

## 4. Time-Series Validation Framework

Data is split chronologically, never randomly, to prevent look-ahead bias.
All features are scaled using RobustScaler fitted only on the training set.

| Split | Date Range | Fraction |
|-------|-----------|----------|
| Train | 2018-03-29 → 2022-11-28 | 70% |
| Validation | 2022-11-29  → 2023-11-29 | 15% |
| Test | 2023-11-30  → 2024-12-02 | 15% |

---

## 5. Model Architectures

### 5.1 Approach 1 – ML Volatility Prediction

#### Random Forest
- Uses 300 trees with a maximum depth of 8 and a minimum of 5 samples per leaf.
- The input is scaled with RobustScaler before training.
- Input: 21 market features, one row per trading day.

#### XGBoost
- Uses 500 boosting rounds with depth 5 and a learning rate of 0.05.
- Row and feature subsampling are both set to 0.8.
- The input is scaled with RobustScaler before training.

#### LSTM
  - Architecture: two LSTM layers with 64 and 32 units, followed by two dense layers with 16 units and 1 output.
  - Lookback window: 20 trading days.
  - Optimizer: Adam with mean squared error loss and early stopping after 10 stagnant validation epochs.
  - Input shape: batches of 20-day sequences with 21 features.

### 5.2 Approach 2 – End-to-End Supervised Pricing

#### Linear Regression
- Standard ordinary least squares with an intercept term.
- Input is scaled with RobustScaler.
- Input: 26 features covering market variables and option parameters.

#### XGBoost
- Uses 500 boosting rounds with depth 6 and a learning rate of 0.05.
- Row and feature subsampling are both set to 0.8.

#### Neural Network (MLP)
- Architecture: dense layers with 128, 64, and 32 units, followed by a single output unit.
- Optimizer: Adam with a learning rate of 0.001 and early stopping through internal validation.
- Maximum iterations: 500.

---

## 6. Performance Summary (Test Set)

### 6.1 Approach 1 – Volatility Prediction

| Model | Vol MAE | Vol RMSE | Option MAE | Option RMSE |
|-------|---------|----------|------------|-------------|
| RandomForest | 0.07052 | 0.10154 | 4.1866 | 6.3097 |
| XGBoost | 0.07394 | 0.09706 | 4.2210 | 6.1274 |
| LSTM | 0.08047 | 0.10774 | 4.3614 | 6.3925 |

Vol MAE/RMSE are measured in annualised volatility units. Option MAE/RMSE are measured in USD for an at-the-money six-month call.

### 6.2 Approach 2 – End-to-End Option Pricing

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|----------|-----|
| LinearRegression | 7.9447 | 9.0996 | 0.3070 |
| XGBoost | 4.5111 | 6.5016 | 0.6462 |
| NeuralNetwork | 1.7683 | 2.5551 | 0.9454 |

The target is the Black-Scholes-Merton price based on 20-day historical volatility. The evaluation grid covers three maturities, three moneyness levels, and both call and put contracts.

---

## 7. Limitations & Recommended Next Steps

### Current Limitations
1. No market-implied volatility is used. The targets are derived from historical or Black-Scholes-Merton prices rather than live option quotes.
2. The option grid is simplified to three maturities, three moneyness levels, and two contract types.
3. The feature set is static and does not include real-time microstructure information such as bid-ask spread or trading volume.
4. The second approach mainly learns the Black-Scholes-Merton surface, while real market prices may still show skew or smile effects.

### Recommended Next Steps (Week 6+)
1. Incorporate implied volatility data to make the targets closer to market pricing.
2. Add Greeks such as delta, gamma, and vega as engineered features.
3. Introduce Bayesian hyperparameter optimisation.
4. Extend the LSTM to forecast volatility over multiple future steps.
5. Build an ensemble that combines the two approaches.
6. Evaluate the methods on 2025 and later data for a cleaner out-of-sample test.

---

Generated by week5_ml_models.py | v1.0 | 20260531
