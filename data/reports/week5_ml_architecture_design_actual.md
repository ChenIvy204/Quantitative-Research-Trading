# ML Architecture Design Document – Week 5 Actual Implementation

## 1. 文档概述

- 目的：明确本项目 Week 5 的机器学习架构设计目标、范围与约束。
- 适用场景：JPM 股票相关期权定价任务，覆盖 2018-2024 年的日频市场数据。
- 关联模块：Week 2 的数据预处理与特征工程流水线，Week 3-4 的 BSM 基准与评估模块，Week 5 的机器学习建模脚本 `scripts/week5_ml_models.py`。

本版本文档对应的是仓库中已经实现并可复现的 Week 5 产物，而不是概念性框架。当前代码实现的是 JPM 欧洲期权定价两路线方案，而不是 Chooser Option 实盘定价；因此，文中的变量、特征与结果均以实际代码为准。

## 2. 问题定义与建模目标

- 输入：
  - 标的资产历史价格及其滚动统计特征
  - 历史波动率、波动率期限结构特征
  - VIX、利率、股息率、新闻情绪等市场状态变量
  - 期权参数：标的价格 S、行权价 K、到期时间 T、是否看涨 `is_call`、平值/虚实值水平 `moneyness`
- 输出：
  - 路线一：预测未来 20 个交易日的实现波动率 `fwd_vol_20d`
  - 路线二：直接输出 BSM 价格 `bsm_price`
- 目标：
  - 最小化测试集上的价格误差，重点指标为 MAE、RMSE 和 R²
  - 与纯 BSM 基准比较，验证机器学习方法在定价近似任务上的有效性

实际实现中，路线一的目标不是市场隐含波动率，而是从历史收益率构造的前瞻实现波动率；路线二的目标不是 CME 的真实成交价，而是用 20 日历史波动率计算得到的 BSM 价格。这意味着当前 Week 5 更接近“模型架构验证与替代定价表面学习”，而不是“直接拟合真实市场期权报价”。

## 3. 数据流与特征工程架构

- 数据源：
  - Yahoo Finance：`yahoo_jpm_2018_2024.csv`
  - FRED：`fred_DGS10_2018_2024.csv`、`fred_VIXCLS_2018_2024.csv`
  - JPM dividends：`jpm_dividends_2018_2024.csv`
  - Alpha Vantage news sentiment：`alphavantage_news_jpm_2018_2024.csv`
- 数据划分策略：
  - 按时间顺序切分为 70% 训练集、15% 验证集、15% 测试集
  - 采用 chronological split，不打乱顺序，以避免未来信息泄露
- 特征预处理流水线：
  - 缺失值处理：主要通过前向/后向填充和丢弃无效样本完成
  - 特征构造：使用滚动窗口统计量、历史收益率、均线偏离、VIX 变化、VIX 与 JPM 收益相关性、情绪滚动统计等
  - 归一化：主要由 `RobustScaler` 完成，对树模型和线性/神经网络输入统一处理
  - 序列特征：LSTM 使用 20 天 lookback 序列
- 特征存储：
  - 导出为 CSV：`data/processed/week5_feature_dataset_v1.0_20260531.csv`
  - 其它模型结果也统一输出到 `data/processed/` 与 `data/reports/`

Week 5 实际使用的核心特征包括：
- 历史波动率：`hist_vol_5d`、`hist_vol_20d`、`hist_vol_60d`
- 波动率结构特征：`vol_ratio_5_20`、`vol_ratio_20_60`、`vol_20d_change`
- 价格动量特征：`return_1d`、`return_5d`、`return_20d`
- 趋势特征：`price_to_ma_20d`、`price_to_ma_60d`
- 风险与情绪特征：`vix`、`vix_change_5d`、`vix_ma_ratio`、`vix_jpm_corr_20d`、`r`、`q`、`sentiment_7d`、`sentiment_20d`、`news_count_7d`、`drawdown_20d`
- 期权参数：`S`、`K`、`moneyness`、`T`、`is_call`

## 4. 模型架构设计（两种路线）

### 路线一：ML 波动率预测 + BSM 定价

- 步骤：
  1. 使用 ML 模型预测未来 20 个交易日的实现波动率 `fwd_vol_20d`
  2. 将预测波动率输入 BSM 公式，得到对应的期权价格近似
- 候选 ML 模型：
  - Random Forest
  - XGBoost；如果环境中缺失 xgboost，则退化为 sklearn 的 GradientBoostingRegressor
  - LSTM（仅在 TensorFlow 可用时训练）
- 损失函数：
  - 波动率预测评估使用 MAE 和 RMSE
- 输出接口：
  - 波动率预测值 → BSM 定价函数 → 路线一的价格误差评估

实际代码中，路线一采用的是 21 个市场特征输入，目标是 `fwd_vol_20d`。LSTM 使用 20 天序列，结构为双层 LSTM + 全连接层。

### 路线二：端到端监督学习定价

- 模型输入：
  - 全部市场特征 + 期权参数
- 模型输出：
  - 直接回归 BSM 价格 `bsm_price`
- 候选模型：
  - LinearRegression
  - XGBoost；如果环境中缺失 xgboost，则退化为 GradientBoostingRegressor
  - MLP 神经网络
- 损失函数：
  - 价格回归误差，使用 MAE、RMSE 和 R² 进行评价
- 正则化：
  - 线性/树模型依赖稳健缩放和早停或深度约束
  - MLP 依赖 early stopping 和固定迭代上限

实际代码中，路线二并不是直接拟合真实市场成交价，而是学习“由历史波动率驱动的 BSM 价格面”。这使得路线二更像是 BSM 定价表面的监督学习近似。

## 5. 训练与验证策略

- 避免时间泄露：
  - 所有特征仅使用预测日之前的数据构造
  - 前瞻波动率 `fwd_vol_20d` 仅作为标签，不作为输入
  - 数据切分按日期顺序执行，不使用随机划分
- 超参数策略：
  - 当前 Week 5 代码中使用的是手工设定的固定参数，而不是完整的网格搜索流程
  - LSTM 通过 early stopping 终止训练
  - MLP 采用 internal validation 和 early stopping
- 评估指标：
  - 路线一：Vol MAE、Vol RMSE、Option MAE、Option RMSE
  - 路线二：MAE、RMSE、R²
- 基准对比：
  - Week 4 的 BSM 评估模块作为基线参考

实际数据划分范围如下：
- 训练集：2018-03-29 → 2022-11-28
- 验证集：2022-11-29 → 2023-11-29
- 测试集：2023-11-30 → 2024-12-02

## 6. 可解释性设计

- 全局解释：
  - 代码中已实现树模型的 feature importance 提取
  - 输出图为 `data/reports/week5_feature_importance.png`
- 局部解释：
  - 当前 Week 5 代码没有集成 SHAP 或 LIME
  - 如果后续需要，可以在树模型上补充 SHAP summary plot 和单样本解释
- 当前可支持的解释结论：
  - 历史波动率、VIX 状态、收益动量和情绪类特征都被纳入候选重要特征
  - 具体重要性需要以树模型输出为准，而不是先验假设

从仓库实际实现看，Week 5 的解释性是“可视化 + 特征重要性”级别，而不是完整 SHAP/LIME 分析流水线。

## 7. 工程实现与部署考虑

- 开发环境：
  - Python 脚本为主，`scripts/week5_ml_models.py` 是主入口
  - 可以在 Jupyter Notebook 中做探索，但最终产物由脚本批处理生成
- 主要库：
  - 数据处理：Pandas、NumPy
  - 模型：scikit-learn、XGBoost（可选）、TensorFlow（可选）
  - 可视化：Matplotlib
- 模型版本管理：
  - 当前实现通过日期与版本号命名输出文件，例如 `v1.0` 和 `20260531`
- 部署接口：
  - 目前还没有正式封装成 REST API 或 `predict_price(features_dict) -> float`
  - 现阶段更像是离线实验脚本与结果导出框架
- 实时更新：
  - 当前代码没有自动增量学习机制
  - 如需扩展，可在 `scripts/pipeline.py` 中追加定期重训练逻辑

## 8. 预期输出与验收标准

- 文档输出：
  - 架构设计文档
  - 特征列表与处理逻辑说明
  - 模型架构与结果摘要
- 代码输出：
  - 可运行的 Week 5 ML 脚本 `scripts/week5_ml_models.py`
  - 训练、验证与结果导出流程
  - 结果图表与 CSV 产物
- 实际产物：
  - `data/processed/week5_feature_dataset_v1.0_20260531.csv`
  - `data/processed/week5_vol_results_v1.0_20260531.csv`
  - `data/processed/week5_pricing_results_v1.0_20260531.csv`
  - `data/processed/week5_model_comparison_v1.0_20260531.csv`
  - `data/reports/week5_ml_architecture_v1.0_20260531.md`
  - `data/reports/week5_feature_importance.png`
  - `data/reports/week5_vol_prediction_comparison.png`
  - `data/reports/week5_pricing_comparison.png`
  - `data/reports/week5_model_performance.png`
- 验收标准：
  - 两条路线都能完成训练和测试集评估
  - 结果文件能够稳定生成
  - 路线二在当前任务中表现最好，测试集 R² 达到 0.9454
  - 路线一能够稳定给出波动率预测与基于 BSM 的价格误差评估

## 9. 风险与缓解措施

- 过拟合风险：
  - 通过时间序列切分、模型深度限制和 early stopping 缓解
- 数据非平稳性：
  - 通过滚动特征和时间窗切分尽量降低影响
- 波动率预测失准：
  - 路线一高度依赖前瞻波动率预测，因此误差会向 BSM 定价传导
- 计算资源限制：
  - 代码对 TensorFlow 和 xgboost 都做了可选导入处理，缺失时可以退化到较轻量的模型

与用户给定框架相比，当前仓库的最大差异是：
1. 任务对象是 JPM 欧洲期权，而不是 Chooser Option。
2. 目标价格是由代码中构造的 BSM 价格和前瞻实现波动率，而不是 CME 实盘成交价。
3. 当前实现重点在于可复现的研究流水线，不是线上部署系统。

---

Generated from actual Week 5 code outputs in `scripts/week5_ml_models.py`.
