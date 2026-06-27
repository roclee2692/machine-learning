# 集成算法复杂非线性回归

California Housing 数据集上的集成算法回归对比实验。

## 快速开始

```bash
python ensemble_nonlinear_regression.py
```

预计耗时 **1~2 分钟**，图表自动保存到当前目录。

## 数据预处理

| 步骤 | 方法 |
|------|------|
| 异常值 | IQR 裁剪（训练集计算边界，统一应用到测试集） |
| 缺失值 | 中位数填充 |
| 偏态变换 | `log1p`（自动检测 `|skew| > 1` 的特征） |
| 特征工程 | 4 个领域特征（户均房间、人均房间、卧室占比、户均人口） |
| 标准化 | `StandardScaler`（所有模型统一使用） |

特征数：8（原始） → 12（领域特征）

## 模型对比

| 模型 | 类型 | 核心参数 |
|------|------|---------|
| LinearRegression | 基线 | - |
| LinearSVR | 线性 SVM | C=1.0 |
| RandomForest | Bagging | n_estimators=100, max_depth=15 |
| GBDT | Boosting | n_estimators=100, lr=0.1, max_depth=5 |
| XGBoost | 高级 Boosting | GridSearchCV 调参（27组参数） |

## 输出文件

| 文件 | 内容 |
|------|------|
| `01_predictions_comparison.png` | 真实值 vs 预测值散点图（6个模型） |
| `02_metrics_comparison.png` | R² / RMSE / MAE / MSE 柱状图 |
| `03_feature_importance.png` | RandomForest 和 XGBoost 特征重要性 Top-15 |
| `04_error_distribution.png` | 预测误差分布直方图 |

## 注意事项

- California Housing 数据集目标变量在 **5.0 处有截断**（$500,000 封顶）
- 固定随机种子 `RANDOM_SEED=42`，结果完全可复现
- 所有模型统一使用标准化数据（含树模型），因为领域特征和原始特征尺度差异大
