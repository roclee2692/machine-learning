"""Train and evaluate Linear Regression on California Housing.

English:
This script is a small, readable baseline machine learning experiment. It loads
the California Housing dataset, trains a Linear Regression model, evaluates the
model, and saves diagnostic plots.

中文：
这个脚本是一个清晰的小型机器学习基线实验。它会加载 California Housing
数据集，训练线性回归模型，评估模型表现，并保存诊断图像。

Workflow / 实验流程：
1. Load dataset / 加载数据集
2. Split train/test data / 划分训练集和测试集
3. Standardize features / 标准化特征
4. Train Linear Regression / 训练线性回归模型
5. Evaluate with MAE, MSE, RMSE, and R2 / 使用 MAE、MSE、RMSE、R2 评估
6. Save diagnostic plots / 保存诊断图像

Dataset note / 数据集说明：
The target is median house value in units of 100,000 dollars. For example,
2.5 means about $250,000. The target is capped near 5.0, so a vertical band at
true value 5.0 is expected in the true-vs-predicted plot.

目标值表示房屋价值中位数，单位是 10 万美元。例如 2.5 约等于 25 万美元。
该数据集的目标值在约 5.0 处被截断，所以真实值与预测值图中出现 True Value
约等于 5.0 的竖直带状分布是正常现象。
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# English: Force UTF-8 console output when Windows Python is called from WSL.
# 中文：当从 WSL 调用 Windows Python 时，强制使用 UTF-8 输出，避免中文乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# English: Resolve paths from this file, not from the terminal directory.
# 中文：基于当前脚本定位项目路径，而不是依赖终端所在目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# English: Prefer common Chinese fonts when available, then fall back to DejaVu.
# 中文：优先使用常见中文字体；如果没有，则回退到 Matplotlib 默认字体。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    """Run the complete regression experiment. / 运行完整回归实验。"""

    # English:
    # Load the dataset as pandas objects. as_frame=True keeps feature names,
    # which makes coefficient analysis easier later.
    #
    # 中文：
    # 以 pandas 格式加载数据。as_frame=True 会保留特征名称，方便后续分析系数。
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    # English: Print a small preview to confirm the dataset loaded correctly.
    # 中文：打印前几行数据，确认数据集已正确加载。
    print("Features / 特征:")
    print(X.head())
    print("\nTarget / 目标值:")
    print(y.head())

    # English:
    # Split the data into training and test sets. test_size=0.2 keeps 20% of
    # data for testing. random_state=42 makes the result reproducible.
    #
    # 中文：
    # 将数据划分为训练集和测试集。test_size=0.2 表示 20% 数据用于测试。
    # random_state=42 用于保证每次运行划分结果一致，方便复现实验。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # English:
    # Standardize features to zero mean and unit variance. This makes feature
    # coefficients more comparable after training.
    #
    # Important: fit_transform is used only on training data, and transform is
    # used on test data. This avoids data leakage.
    #
    # 中文：
    # 对特征做标准化，使每个特征接近 0 均值、1 标准差。这样训练后的系数更便于比较。
    #
    # 注意：训练集使用 fit_transform，测试集只使用 transform。这样可以避免测试集信息
    # 泄漏到训练过程中。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # English:
    # Train an ordinary least squares Linear Regression model. The model learns
    # one coefficient per feature plus one intercept.
    #
    # 中文：
    # 训练普通最小二乘线性回归模型。模型会学习每个特征对应的系数，以及一个截距项。
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # English: Predict target values for unseen test samples.
    # 中文：对测试集进行预测，用于评估模型泛化能力。
    y_pred = model.predict(X_test_scaled)

    # English:
    # Evaluate model performance with complementary regression metrics.
    # MAE is easy to interpret, MSE/RMSE penalize large errors, and R2 describes
    # how much variance the model explains.
    #
    # 中文：
    # 使用多个互补的回归指标评估模型。MAE 易解释，MSE/RMSE 对大误差更敏感，
    # R2 用于衡量模型解释了多少目标变量方差。
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation / 模型评估:")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    # English:
    # Inspect feature coefficients. Because features were standardized, larger
    # absolute coefficients usually mean stronger influence in this linear
    # model. This is model interpretation, not proof of real-world causality.
    #
    # 中文：
    # 查看特征系数。由于特征已经标准化，绝对值更大的系数通常表示该特征在当前线性
    # 模型中影响更强。但这只是模型解释，不等于现实中的因果关系证明。
    coef_df = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": model.coef_,
        }
    ).sort_values(by="coefficient", key=abs, ascending=False)

    print("\nFeature coefficients / 特征系数:")
    print(coef_df)

    # English:
    # Basic checks verify that the full dataset is used and explain unusual
    # points in the plots. Linear Regression is unconstrained, so it can predict
    # values below 0 or above the dataset target cap.
    #
    # 中文：
    # 基础检查用于确认使用了完整数据集，并解释图中异常点。线性回归没有输出范围约束，
    # 所以可能预测出小于 0 或大于数据集上限的值。
    capped_targets = (y_test >= 5).sum()
    negative_predictions = (y_pred < 0).sum()
    high_predictions = (y_pred > 5).sum()

    print("\nData check / 数据检查:")
    print(f"Dataset shape / 数据集形状       : {X.shape}")
    print(f"Train shape / 训练集形状         : {X_train.shape}")
    print(f"Test shape / 测试集形状          : {X_test.shape}")
    print(f"Test target range / 测试目标范围 : {y_test.min():.4f} to {y_test.max():.4f}")
    print(f"Prediction range / 预测值范围    : {y_pred.min():.4f} to {y_pred.max():.4f}")
    print(f"Targets capped near 5 / 约为 5 的目标值数量 : {capped_targets}")
    print(f"Predictions below 0 / 小于 0 的预测数量     : {negative_predictions}")
    print(f"Predictions above 5 / 大于 5 的预测数量     : {high_predictions}")

    # English:
    # Plot 1: true values vs. predicted values.
    # This plot checks prediction accuracy visually. Points close to the dashed
    # y=x line are accurate; points far from the line have larger errors.
    #
    # 中文：
    # 图 1：真实值 vs 预测值。
    # 这张图用于直观看预测是否接近理想情况。点越靠近虚线 y=x，预测越准确；
    # 离虚线越远，误差越大。
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, label="Test samples / 测试样本")
    plt.xlabel("True Value / 真实值")
    plt.ylabel("Predicted Value / 预测值")
    plt.title("True vs Predicted House Value / 真实值与预测值")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Ideal prediction / 理想预测",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "true_vs_predicted.png", dpi=300)
    plt.close()

    # English:
    # Residual means prediction error after fitting:
    # residual = true value - predicted value.
    #
    # 中文：
    # 残差表示模型预测之后剩下的误差：
    # 残差 = 真实值 - 预测值。
    residuals = y_test - y_pred

    # English:
    # Plot 2: residuals vs. predicted values.
    # This plot checks whether errors are randomly centered around zero. Clear
    # curves, funnels, or diagonal bands suggest model limitations or target
    # clipping effects.
    #
    # 中文：
    # 图 2：残差 vs 预测值。
    # 这张图用于检查误差是否大致围绕 0 随机分布。如果出现明显曲线、漏斗形或斜线
    # 带状结构，可能说明线性模型过于简单，或目标值截断影响了结果。
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.4, label="Residuals / 残差")
    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Zero error / 零误差",
    )
    plt.xlabel("Predicted Value / 预测值")
    plt.ylabel("Residual / 残差")
    plt.title("Residual Plot / 残差图")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residual_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
