# California Housing Linear Regression / 加州房价线性回归

English | 中文

## English

This project implements a basic machine learning regression pipeline using the
California Housing dataset from scikit-learn.

## 中文

本项目使用 scikit-learn 内置的 California Housing 数据集，实现一个基础的机器学习
回归实验流程。

## Goal / 项目目标

English:
Predict median house value based on housing-related features.

中文：
根据房屋相关特征预测加州地区的房屋价值中位数。

## Workflow / 实验流程

1. Load dataset / 加载数据集
2. Split train/test data / 划分训练集和测试集
3. Standardize features / 标准化特征
4. Train Linear Regression model / 训练线性回归模型
5. Evaluate with MAE, MSE, RMSE, and R2 / 使用 MAE、MSE、RMSE、R2 评估
6. Visualize prediction results and residuals / 可视化预测结果和残差

## Dataset / 数据集

English:
The dataset is loaded from `sklearn.datasets.fetch_california_housing`, so no
manual download is required. It contains 20,640 samples and 8 features.

中文：
数据集直接来自 `sklearn.datasets.fetch_california_housing`，无需手动下载。
它包含 20,640 条样本和 8 个特征。

English:
The target value is median house value in units of 100,000 dollars. The target
is capped near 5.0, which affects the visualizations.

中文：
目标值表示房屋价值中位数，单位为 10 万美元。目标值在约 5.0 处被截断，
这会影响可视化图像中的分布形状。

## Project Structure / 项目结构

```text
california-housing-linear-regression/
├─ README.md
├─ requirements.txt
├─ src/
│  └─ train_linear_regression.py
├─ outputs/
│  └─ figures/
│     ├─ true_vs_predicted.png
│     └─ residual_plot.png
└─ reports/
   └─ experiment_report.md
```

## Installation / 安装依赖

```bash
pip install -r requirements.txt
```

## Run / 运行方法

```bash
python src/train_linear_regression.py
```

If you use the local Conda environment / 如果使用本地 Conda 环境：

```bash
/mnt/c/Users/Raelon/miniconda3/envs/ai_env/python.exe src/train_linear_regression.py
```

## Metrics / 评估指标

| Metric / 指标 | Meaning / 含义 |
| --- | --- |
| MAE | Mean Absolute Error / 平均绝对误差 |
| MSE | Mean Squared Error / 均方误差 |
| RMSE | Root Mean Squared Error / 均方根误差 |
| R2 | Coefficient of Determination / 决定系数 |

Current experiment results / 当前实验结果：

| Metric / 指标 | Value / 数值 |
| --- | ---: |
| MAE | 0.5332 |
| MSE | 0.5559 |
| RMSE | 0.7456 |
| R2 | 0.5758 |

## Outputs / 输出文件

The script saves figures to `outputs/figures/`.

脚本会将图像保存到 `outputs/figures/`。

- `true_vs_predicted.png`
  - English: Compares real target values with predicted values.
  - 中文：比较真实目标值和模型预测值。
- `residual_plot.png`
  - English: Shows prediction errors against predicted values.
  - 中文：展示预测值对应的残差误差。

## Notes / 说明

English:
The vertical band near true value 5.0 in the true-vs-predicted plot is expected
because the dataset target is capped. Some predictions are below 0 or above 5
because plain Linear Regression does not constrain its output range.

中文：
真实值与预测值图中，真实值约 5.0 附近的竖直带状分布是正常现象，因为数据集目标值
被截断。部分预测值小于 0 或大于 5，是因为普通线性回归没有限制输出范围。
