# Feature Engineering Reference

This project builds a daily modeling table from JPM market data, FRED rates, VIX, and Alpha Vantage news sentiment inputs.

## Data preparation

- Aligns JPM, DGS10, and VIX series on a shared daily index.
- Forward-fills macro series and interpolates price fields where appropriate.
- Produces a data quality report with per-feature missing rate, min, max, mean, standard deviation, and outlier counts.
- Identifies outliers with the 3σ rule and replaces flagged values with the feature median.
- Documents the missing-value strategy for price, macro, dividend, and news-derived features.
- Exports a combined PDF report that places the written summary and the boxplot figure in one file.
- Builds the optimized feature set by pruning features with absolute Pearson correlation above 0.8 and keeping only features with absolute IC above 0.03.

## Engineered features

- `jpm_return_1d`: one-day adjusted close return.
- `jpm_return_5d`: five-day adjusted close return.
- `jpm_vol_5d`: 5-day rolling annualized volatility.
- `jpm_vol_20d`: 20-day rolling annualized volatility.
- `jpm_vol_60d`: 60-day rolling annualized volatility.
- `jpm_vol_20d_change_1d`: one-day difference in 20-day volatility.
- `jpm_vol_20d_change_rate_1d`: one-day percentage change in 20-day volatility.
- `jpm_ma_20d`: 20-day rolling mean of adjusted close.
- `jpm_price_to_ma_20d`: adjusted close divided by the 20-day moving average.
- `jpm_dividend_ttm`: trailing 12-month dividend per share.
- `jpm_dividend_yield_ttm`: trailing 12-month dividend divided by current adjusted close.
- `jpm_dividend_growth_yoy`: year-over-year dividend growth based on trailing dividend sums.
- `dgs10_change_1d`: one-day change in the 10-year Treasury yield.
- `dgs10_momentum_5d`: five-day momentum in the 10-year Treasury yield.
- `vix_change_1d`: one-day change in VIX.
- `vix_jpm_corr_20d`: 20-day rolling correlation between JPM returns and VIX changes.
- `rolling_high_20d`: 20-day rolling high of adjusted close.
- `drawdown_20d`: distance from the rolling high.

## News features

- `news_article_count`: daily article count.
- `news_sentiment_mean`: daily average sentiment score on a 0-1 scale.
- `news_sentiment_std`: daily sentiment dispersion.
- `news_7d_article_count`: 7-day rolling article count.
- `news_7d_sentiment_mean`: 7-day rolling average sentiment on a 0-1 scale.
- `news_7d_sentiment_trend`: day-over-day change in the rolling sentiment average.

## Output files

- `data/processed/week2_feature_dataset_2018_2024.csv`
- `data/processed/week2_feature_dataset_2018_2024.parquet`
- `data/processed/week2_feature_dataset_v1.0_YYYYMMDD.csv`
- `data/processed/week2_feature_dataset_v1.0_YYYYMMDD.parquet`
- `data/processed/week2_data_quality_report_v1.0_YYYYMMDD.csv`
- `data/processed/week2_data_quality_report_v1.0_YYYYMMDD.pdf`
- `data/processed/week2_feature_optimization_report_v1.0_YYYYMMDD.csv`
- `data/processed/week2_feature_ic_report_v1.0_YYYYMMDD.csv`
- `data/processed/week2_feature_correlation_matrix_v1.0_YYYYMMDD.csv`
- `data/processed/week2_data_quality_report_v1.0_YYYYMMDD.md`
- `data/processed/week2_data_quality_report_boxplot_v1.0_YYYYMMDD.png`
- `data/processed/week2_feature_optimization_report_v1.0_YYYYMMDD.pdf`