"""
线性回归 - 完整实现
包含从零实现和scikit-learn实现两种方式

相关方法对比
方法	特点	何时用
线性回归	无正则化	数据干净，特征独立
岭回归	L2 正则化（平方项）	多重共线性
Lasso	L1 正则化（绝对值）	需要特征选择
弹性网	L1+L2 混合	结合两者优点
简单说：岭回归 = 普通线性回归 + "不要让参数太大"的约束 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 1. 从零实现线性回归 ==========

class LinearRegressionFromScratch:
    """从零实现线性回归（梯度下降法）"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """训练模型

        Args:
            X: shape (m, n) - m个样本，n个特征
            y: shape (m,) - m个目标值
        """
        m, n = X.shape

        # 初始化参数
        self.weights = np.zeros(n)
        self.bias = 0

        # 梯度下降
        for i in range(self.n_iterations):
            # 预测
            y_pred = self.predict(X)

            # 计算损失
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)

            # 计算梯度
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # 每100次迭代打印一次
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias

    def plot_cost(self):
        """绘制损失函数曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失函数 J')
        plt.title('损失函数随迭代次数的变化')
        plt.grid(True)
        plt.show()


class LinearRegressionNormalEquation:
    """从零实现线性回归（正规方程法）"""

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """使用正规方程训练模型

        w = (X^T X)^(-1) X^T y
        """
        # 在X前面添加一列1（用于偏置项）
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # 正规方程
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias


# ========== 2. 数据生成 ==========

def generate_linear_data(n_samples=100, n_features=1, noise=10, random_state=42):
    """生成线性回归数据

    Args:
        n_samples: 样本数量
        n_features: 特征数量
        noise: 噪声强度
        random_state: 随机种子
    """
    np.random.seed(random_state)

    # 生成特征
    X = 2 * np.random.rand(n_samples, n_features)

    # 生成真实权重
    true_weights = np.random.randn(n_features) * 10
    true_bias = np.random.randn() * 10

    # 生成目标值 (加上噪声)
    y = np.dot(X, true_weights) + true_bias + noise * np.random.randn(n_samples)

    print(f"真实权重: {true_weights}")
    print(f"真实偏置: {true_bias:.2f}")

    return X, y, true_weights, true_bias


# ========== 3. 评估函数 ==========

def evaluate_model(y_true, y_pred, model_name="Model"):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} 评估结果:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_predictions(X, y_true, y_pred, title="预测结果对比"):
    """可视化预测结果"""
    plt.figure(figsize=(10, 6))
    
    # 确保X是二维的，并处理单列或多列的情况
    if len(X.shape) == 1 or X.shape[1] == 1:
        # 单变量情况，按X值排序
        if len(X.shape) == 1:
            X_plot = X
        else:
            X_plot = X[:, 0]
        
        # 按X值排序用于绘制拟合线
        sort_idx = np.argsort(X_plot)
        plt.scatter(X_plot, y_true, color='blue', alpha=0.5, label='真实值')
        plt.plot(X_plot[sort_idx], y_pred[sort_idx], color='red', linewidth=2, label='预测值')
    else:
        # 多变量情况，用第一个特征绘制
        X_plot = X[:, 0]
        plt.scatter(X_plot, y_true, color='blue', alpha=0.5, label='真实值')
        plt.scatter(X_plot, y_pred, color='red', alpha=0.5, label='预测值')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_residuals(y_true, y_pred):
    """绘制残差图"""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 残差散点图
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('预测值')
    axes[0].set_ylabel('残差')
    axes[0].set_title('残差散点图')
    axes[0].grid(True)

    # 残差直方图
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('残差')
    axes[1].set_ylabel('频数')
    axes[1].set_title('残差分布')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# ========== 4. 示例：单特征线性回归 ==========

def example1_simple_linear_regression():
    """示例1：简单线性回归（一元）"""
    print("="*50)
    print("示例1：简单线性回归")
    print("="*50)

    # 生成数据
    X, y, true_w, true_b = generate_linear_data(n_samples=100, n_features=1, noise=10)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 方法1：梯度下降
    print("\n方法1：梯度下降法")
    model_gd = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=1000)
    model_gd.fit(X_train, y_train)
    print(f"\n学到的权重: {model_gd.weights}")
    print(f"学到的偏置: {model_gd.bias:.2f}")

    y_pred_gd = model_gd.predict(X_test)
    evaluate_model(y_test, y_pred_gd, "梯度下降")

    # 绘制损失函数
    model_gd.plot_cost()

    # 方法2：正规方程
    print("\n方法2：正规方程法")
    model_ne = LinearRegressionNormalEquation()
    model_ne.fit(X_train, y_train)
    print(f"\n学到的权重: {model_ne.weights}")
    print(f"学到的偏置: {model_ne.bias:.2f}")

    y_pred_ne = model_ne.predict(X_test)
    evaluate_model(y_test, y_pred_ne, "正规方程")

    # 方法3：sklearn
    print("\n方法3：sklearn实现")
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train, y_train)
    print(f"\n学到的权重: {model_sklearn.coef_}")
    print(f"学到的偏置: {model_sklearn.intercept_:.2f}")

    y_pred_sklearn = model_sklearn.predict(X_test)
    evaluate_model(y_test, y_pred_sklearn, "sklearn")

    # 可视化
    plot_predictions(X_test, y_test, y_pred_sklearn, "sklearn线性回归预测")
    plot_residuals(y_test, y_pred_sklearn)


# ========== 5. 示例：多特征线性回归 ==========

def example2_multiple_linear_regression():
    """示例2：多元线性回归"""
    print("\n" + "="*50)
    print("示例2：多元线性回归")
    print("="*50)

    # 生成多特征数据
    X, y, true_w, true_b = generate_linear_data(n_samples=200, n_features=3, noise=5)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print(f"\n学到的权重: {model.coef_}")
    print(f"学到的偏置: {model.intercept_:.2f}")

    # 预测和评估
    y_pred = model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred, "多元线性回归")

    # 绘制特征重要性
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model.coef_)), np.abs(model.coef_))
    plt.xlabel('特征索引')
    plt.ylabel('权重绝对值')
    plt.title('特征重要性')
    plt.grid(True)
    plt.show()


# ========== 6. 示例：正则化 ==========

def example3_regularization():
    """示例3：正则化（Ridge和Lasso）"""
    print("\n" + "="*50)
    print("示例3：正则化回归")
    print("="*50)

    # 生成容易过拟合的数据
    X, y, _, _ = generate_linear_data(n_samples=50, n_features=20, noise=15)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 普通线性回归
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    # Ridge回归 (L2正则化)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    # Lasso回归 (L1正则化)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)

    # 评估对比
    evaluate_model(y_test, y_pred_lr, "普通线性回归")
    evaluate_model(y_test, y_pred_ridge, "Ridge回归")
    evaluate_model(y_test, y_pred_lasso, "Lasso回归")

    # 对比权重
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.bar(range(len(lr.coef_)), lr.coef_)
    plt.title('普通线性回归权重')
    plt.xlabel('特征索引')
    plt.ylabel('权重值')

    plt.subplot(1, 3, 2)
    plt.bar(range(len(ridge.coef_)), ridge.coef_)
    plt.title('Ridge回归权重')
    plt.xlabel('特征索引')
    plt.ylabel('权重值')

    plt.subplot(1, 3, 3)
    plt.bar(range(len(lasso.coef_)), lasso.coef_)
    plt.title('Lasso回归权重')
    plt.xlabel('特征索引')
    plt.ylabel('权重值')

    plt.tight_layout()
    plt.show()

    print(f"\nLasso回归中被置为0的特征数: {np.sum(lasso.coef_ == 0)}")


# ========== 7. 示例：多项式回归 ==========

def example4_polynomial_regression():
    """示例4：多项式回归"""
    print("\n" + "="*50)
    print("示例4：多项式回归")
    print("="*50)

    from sklearn.preprocessing import PolynomialFeatures

    # 生成非线性数据
    np.random.seed(42)
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

    # 1次（线性）
    model_1 = LinearRegression()
    model_1.fit(X, y)

    # 2次多项式
    poly_2 = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_2 = poly_2.fit_transform(X)
    model_2 = LinearRegression()
    model_2.fit(X_poly_2, y)

    # 10次多项式（过拟合）
    poly_10 = PolynomialFeatures(degree=10, include_bias=False)
    X_poly_10 = poly_10.fit_transform(X)
    model_10 = LinearRegression()
    model_10.fit(X_poly_10, y)

    # 可视化
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_pred_1 = model_1.predict(X_test)
    y_pred_2 = model_2.predict(poly_2.transform(X_test))
    y_pred_10 = model_10.predict(poly_10.transform(X_test))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X_test, y_pred_1, 'r-', linewidth=2)
    plt.title('1次（欠拟合）')
    plt.xlabel('X')
    plt.ylabel('y')

    plt.subplot(1, 3, 2)
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X_test, y_pred_2, 'r-', linewidth=2)
    plt.title('2次（刚好）')
    plt.xlabel('X')
    plt.ylabel('y')

    plt.subplot(1, 3, 3)
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X_test, y_pred_10, 'r-', linewidth=2)
    plt.title('10次（过拟合）')
    plt.xlabel('X')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()


# ========== 主函数 ==========

if __name__ == "__main__":
    # 运行所有示例
    example1_simple_linear_regression()
    example2_multiple_linear_regression()
    example3_regularization()
    example4_polynomial_regression()

    print("\n" + "="*50)
    print("所有示例运行完成！")
    print("="*50)
