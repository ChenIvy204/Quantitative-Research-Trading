# Feature Engineering Reference

This project builds a daily modeling table from JPM market data, FRED rates, VIX, and Alpha Vantage news sentiment inputs.

## Data preparation

- Aligns JPM, DGS10, and VIX series on a shared daily index.
- Forward-fills macro series and interpolates price fields where appropriate.
- Caps obvious outliers in key return and change series with IQR clipping.

## Engineered features

- `jpm_return_1d`: one-day adjusted close return.
- `jpm_return_5d`: five-day adjusted close return.
- `jpm_vol_20d`: 20-day rolling annualized volatility.
- `jpm_ma_20d`: 20-day rolling mean of adjusted close.
- `jpm_price_to_ma_20d`: adjusted close divided by the 20-day moving average.
- `jpm_dividend_ttm`: trailing 12-month dividend per share.
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