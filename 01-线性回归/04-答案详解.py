"""
线性回归练习题 - 答案详解
包含所有练习题的完整答案和详细解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 练习1：房价预测（基础）==========

def exercise1_house_price_basic():
    """练习1答案"""
    print("="*50)
    print("练习1：房价预测（基础）")
    print("="*50)

    # 数据
    areas = np.array([50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
    bedrooms = np.array([1, 2, 2, 2, 3, 3, 3, 4]).reshape(-1, 1)
    ages = np.array([5, 3, 8, 2, 10, 1, 6, 4]).reshape(-1, 1)
    prices = np.array([200, 250, 260, 310, 300, 380, 350, 420])

    # 合并特征
    X = np.hstack([areas, bedrooms, ages])

    # 任务1：训练模型
    model = LinearRegression()
    model.fit(X, prices)

    print(f"模型系数: {model.coef_}")
    print(f"模型截距: {model.intercept_:.2f}")

    # 任务2：预测
    new_house = np.array([[80, 2, 5]])  # 80平米, 2卧室, 5年房龄
    predicted_price = model.predict(new_house)
    print(f"\n预测价格: {predicted_price[0]:.2f}万元")

    # 任务3：R²分数
    y_pred = model.predict(X)
    r2 = r2_score(prices, y_pred)
    print(f"R²分数: {r2:.4f}")

    # 任务4：可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(prices, y_pred, alpha=0.7)
    plt.plot([prices.min(), prices.max()], [prices.min(), prices.max()], 'r--', lw=2)
    plt.xlabel('真实价格（万元）')
    plt.ylabel('预测价格（万元）')
    plt.title('真实值 vs 预测值')
    plt.grid(True)
    plt.show()

    print("\n分析：")
    print(f"- 面积系数为 {model.coef_[0]:.2f}，表示每增加1平米，房价增加{model.coef_[0]:.2f}万元")
    print(f"- 卧室系数为 {model.coef_[1]:.2f}，表示每增加1个卧室，房价增加{model.coef_[1]:.2f}万元")
    print(f"- 房龄系数为 {model.coef_[2]:.2f}，表示每增加1年房龄，房价变化{model.coef_[2]:.2f}万元")
    print(f"- R²={r2:.4f}，模型解释了{r2*100:.2f}%的价格变化")


# ========== 练习2：手写梯度下降（进阶）==========

def exercise2_gradient_descent():
    """练习2答案"""
    print("\n" + "="*50)
    print("练习2：手写梯度下降")
    print("="*50)

    # 数据
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
        """梯度下降实现"""
        m = len(X)
        w = 0
        b = 0
        cost_history = []

        for i in range(n_iterations):
            # 预测
            y_pred = w * X + b

            # 损失
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            cost_history.append(cost)

            # 梯度
            dw = (1/m) * np.sum((y_pred - y) * X)
            db = (1/m) * np.sum(y_pred - y)

            # 更新
            w -= learning_rate * dw
            b -= learning_rate * db

        return w, b, cost_history

    # 测试不同学习率
    learning_rates = [0.001, 0.01, 0.1]
    plt.figure(figsize=(15, 5))

    for idx, lr in enumerate(learning_rates, 1):
        w, b, cost_history = gradient_descent(X, y, learning_rate=lr, n_iterations=1000)

        print(f"\n学习率 = {lr}:")
        print(f"  最终权重 w = {w:.6f}")
        print(f"  最终偏置 b = {b:.6f}")
        print(f"  最终损失 = {cost_history[-1]:.6f}")

        plt.subplot(1, 3, idx)
        plt.plot(cost_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失')
        plt.title(f'学习率 = {lr}')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n分析：")
    print("- 学习率0.001: 收敛很慢，需要更多迭代")
    print("- 学习率0.01: 收敛速度适中")
    print("- 学习率0.1: 收敛很快，但可能不稳定")


# ========== 练习3：加州房价预测（实战）==========

def exercise3_california_housing():
    """练习3答案"""
    print("\n" + "="*50)
    print("练习3：加州房价预测")
    print("="*50)

    # 1. 数据加载与探索
    print("\n1. 数据加载与探索")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"数据维度: {X.shape}")
    print(f"特征名称: {feature_names}")
    print(f"\n目标值统计:\n{pd.Series(y).describe()}")

    # 创建DataFrame方便分析
    df = pd.DataFrame(X, columns=feature_names)
    df['Price'] = y

    # 相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.show()

    # 2. 数据预处理
    print("\n2. 数据预处理")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 3. 模型训练
    print("\n3. 模型训练")

    # 普通线性回归
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Ridge回归
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    # Lasso回归
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)

    # 4. 模型评估
    print("\n4. 模型评估")

    models = {
        '线性回归': lr,
        'Ridge回归': ridge,
        'Lasso回归': lasso
    }

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {'MSE': mse, 'RMSE': rmse, 'R²': r2}

        print(f"\n{name}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")

    # 残差图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test_scaled)
        residuals = y_test - y_pred

        axes[idx].scatter(y_pred, residuals, alpha=0.5)
        axes[idx].axhline(y=0, color='r', linestyle='--')
        axes[idx].set_xlabel('预测值')
        axes[idx].set_ylabel('残差')
        axes[idx].set_title(f'{name} 残差图')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()

    # 5. 特征重要性
    print("\n5. 特征重要性分析")

    # 使用Ridge的系数（更稳定）
    coef_abs = np.abs(ridge.coef_)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coef_abs
    }).sort_values('Importance', ascending=False)

    print("\n特征重要性排名:")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, ridge.coef_)
    plt.xlabel('权重')
    plt.title('Ridge回归特征权重')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n结论：")
    best_model = max(results, key=lambda x: results[x]['R²'])
    print(f"- 最佳模型：{best_model}")
    print(f"- 最重要的3个特征：{', '.join(feature_importance.head(3)['Feature'].values)}")


# ========== 练习4：多项式特征（进阶）==========

def exercise4_polynomial_features():
    """练习4答案"""
    print("\n" + "="*50)
    print("练习4：多项式特征")
    print("="*50)

    # 生成数据
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 + 3*X + 0.5*X**2 - 0.1*X**3 + np.random.randn(50, 1) * 3
    y = y.ravel()

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 测试不同阶数
    degrees = [1, 2, 3, 5, 10]
    results = []

    plt.figure(figsize=(18, 10))

    for idx, degree in enumerate(degrees, 1):
        # 创建多项式特征
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # 训练模型
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # 评估
        train_score = model.score(X_train_poly, y_train)
        test_score = model.score(X_test_poly, y_test)

        results.append({
            'Degree': degree,
            'Train R²': train_score,
            'Test R²': test_score
        })

        print(f"\n{degree}次多项式:")
        print(f"  训练R²: {train_score:.4f}")
        print(f"  测试R²: {test_score:.4f}")

        # 可视化
        X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
        X_plot_poly = poly.transform(X_plot)
        y_plot = model.predict(X_plot_poly)

        plt.subplot(2, 3, idx)
        plt.scatter(X_train, y_train, alpha=0.5, label='训练集')
        plt.scatter(X_test, y_test, alpha=0.5, label='测试集')
        plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='拟合曲线')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'{degree}次多项式 (测试R²={test_score:.3f})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 对比图
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Degree'], results_df['Train R²'], 'o-', label='训练R²')
    plt.plot(results_df['Degree'], results_df['Test R²'], 's-', label='测试R²')
    plt.xlabel('多项式阶数')
    plt.ylabel('R²分数')
    plt.title('不同阶数的模型性能对比')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n分析：")
    print("- 1次：欠拟合，训练和测试R²都较低")
    print("- 2-3次：拟合良好，测试R²最高")
    print("- 5-10次：过拟合，训练R²很高但测试R²下降")


# ========== 练习5：学习曲线分析（高级）==========

def exercise5_learning_curve():
    """练习5答案"""
    print("\n" + "="*50)
    print("练习5：学习曲线分析")
    print("="*50)

    # 加载数据
    data = fetch_california_housing()
    X, y = data.data, data.target

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def plot_learning_curve(model, X, y, title="学习曲线"):
        """绘制学习曲线"""
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_sizes = np.linspace(0.1, 1.0, 10)
        train_errors = []
        val_errors = []

        for size in train_sizes:
            # 采样
            n_samples = int(len(X_train_full) * size)
            X_train = X_train_full[:n_samples]
            y_train = y_train_full[:n_samples]

            # 训练
            model.fit(X_train, y_train)

            # 评估
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_test)

            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_test, y_val_pred)

            train_errors.append(train_mse)
            val_errors.append(val_mse)

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes * len(X_train_full), train_errors, 'o-', label='训练误差')
        plt.plot(train_sizes * len(X_train_full), val_errors, 's-', label='验证误差')
        plt.xlabel('训练样本数')
        plt.ylabel('MSE')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

        # 分析
        if train_errors[-1] > val_errors[-1] * 0.8:
            print("分析：存在高偏差（欠拟合）问题")
            print("建议：增加模型复杂度或添加更多特征")
        elif val_errors[-1] > train_errors[-1] * 1.5:
            print("分析：存在高方差（过拟合）问题")
            print("建议：增加训练数据或使用正则化")
        else:
            print("分析：模型拟合良好")

    # 测试不同模型
    plot_learning_curve(LinearRegression(), X_scaled, y, "线性回归学习曲线")
    plot_learning_curve(Ridge(alpha=1.0), X_scaled, y, "Ridge回归学习曲线")


# ========== 练习6：特征工程（综合）==========

def exercise6_feature_engineering():
    """练习6答案"""
    print("\n" + "="*50)
    print("练习6：特征工程")
    print("="*50)

    # 原始数据
    engine_size = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    weight = np.array([1000, 1200, 1400, 1600, 1800, 2000, 2200])
    mpg = np.array([35, 32, 28, 25, 22, 20, 18])

    # 基础模型
    X_basic = np.column_stack([engine_size, weight])
    model_basic = LinearRegression()
    model_basic.fit(X_basic, mpg)
    score_basic = model_basic.score(X_basic, mpg)

    print(f"基础模型 R²: {score_basic:.4f}")

    # 特征工程
    power_weight_ratio = engine_size / weight * 1000  # 功率重量比
    interaction = engine_size * weight  # 交互特征
    engine_squared = engine_size ** 2  # 平方特征
    weight_squared = weight ** 2

    X_engineered = np.column_stack([
        engine_size,
        weight,
        power_weight_ratio,
        interaction,
        engine_squared,
        weight_squared
    ])

    model_engineered = LinearRegression()
    model_engineered.fit(X_engineered, mpg)
    score_engineered = model_engineered.score(X_engineered, mpg)

    print(f"特征工程后 R²: {score_engineered:.4f}")
    print(f"R²提升: {(score_engineered - score_basic):.4f}")

    # 特征重要性
    feature_names = ['engine_size', 'weight', 'power_weight_ratio',
                     'interaction', 'engine²', 'weight²']
    coef_abs = np.abs(model_engineered.coef_)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coef_abs)
    plt.xlabel('权重绝对值')
    plt.title('特征重要性')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n结论：")
    print("- 特征工程显著提升了模型性能")
    print(f"- 最重要的特征：{feature_names[np.argmax(coef_abs)]}")


# ========== 练习7：正则化参数调优（高级）==========

def exercise7_hyperparameter_tuning():
    """练习7答案"""
    print("\n" + "="*50)
    print("练习7：正则化参数调优")
    print("="*50)

    # 加载数据
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 网格搜索
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    print(f"\n最优alpha: {grid_search.best_params_['alpha']}")
    print(f"最优交叉验证分数: {grid_search.best_score_:.4f}")

    # 在测试集上评估
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"测试集R²: {test_score:.4f}")

    # 可视化
    results = pd.DataFrame(grid_search.cv_results_)

    plt.figure(figsize=(10, 6))
    plt.semilogx(param_grid['alpha'], results['mean_test_score'], 'o-')
    plt.fill_between(param_grid['alpha'],
                     results['mean_test_score'] - results['std_test_score'],
                     results['mean_test_score'] + results['std_test_score'],
                     alpha=0.3)
    plt.xlabel('Alpha')
    plt.ylabel('交叉验证分数（R²）')
    plt.title('Ridge回归参数调优')
    plt.grid(True)
    plt.show()


# ========== 练习8：工资预测（项目）==========

def exercise8_salary_prediction():
    """练习8答案"""
    print("\n" + "="*50)
    print("练习8：工资预测项目")
    print("="*50)

    # 数据
    years_experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    education_level = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4])
    age = np.array([22, 23, 25, 26, 27, 28, 30, 32, 33, 35])
    salary = np.array([5, 6, 8, 9, 10, 12, 15, 18, 20, 25])

    # 1. 数据探索
    print("\n1. 数据探索")
    df = pd.DataFrame({
        'Years': years_experience,
        'Education': education_level,
        'Age': age,
        'Salary': salary
    })
    print(df.describe())

    # 相关性
    print("\n相关性分析:")
    print(df.corr()['Salary'].sort_values(ascending=False))

    # 2. 特征工程
    X = np.column_stack([years_experience, education_level, age])

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 模型训练（使用交叉验证）
    models = {
        '线性回归': LinearRegression(),
        'Ridge': Ridge(alpha=0.1),
        'Lasso': Lasso(alpha=0.1)
    }

    print("\n交叉验证结果（5折）:")
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, salary, cv=5, scoring='r2')
        print(f"{name}: R² = {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 4. 选择最佳模型并训练
    best_model = LinearRegression()
    best_model.fit(X_scaled, salary)

    # 5. 预测新员工
    new_employee = np.array([[3, 2, 26]])  # 3年经验, 硕士, 26岁
    new_employee_scaled = scaler.transform(new_employee)
    predicted_salary = best_model.predict(new_employee_scaled)

    print(f"\n新员工预测工资: {predicted_salary[0]:.2f}万元/年")

    # 6. 分析报告
    print("\n" + "="*50)
    print("分析报告")
    print("="*50)
    print("\n主要发现:")
    print(f"1. 工作年限的系数: {best_model.coef_[0]:.2f}")
    print(f"2. 教育水平的系数: {best_model.coef_[1]:.2f}")
    print(f"3. 年龄的系数: {best_model.coef_[2]:.2f}")
    print("\n结论:")
    print("- 工作年限对工资影响最大")
    print("- 教育水平也有显著影响")
    print("- 年龄的影响相对较小（可能与工作年限高度相关）")


# ========== 挑战题：三种梯度下降对比 ==========

def challenge_gradient_descent_comparison():
    """挑战题答案"""
    print("\n" + "="*50)
    print("挑战题：梯度下降方法对比")
    print("="*50)

    # 生成数据
    np.random.seed(42)
    X = 2 * np.random.rand(1000, 1)
    y = 4 + 3 * X + np.random.randn(1000, 1)
    y = y.ravel()

    # 批量梯度下降
    def batch_gradient_descent(X, y, lr=0.01, n_iter=100):
        m = len(y)
        w, b = 0, 0
        cost_history = []

        for _ in range(n_iter):
            y_pred = w * X.ravel() + b
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            cost_history.append(cost)

            dw = (1/m) * np.sum((y_pred - y) * X.ravel())
            db = (1/m) * np.sum(y_pred - y)

            w -= lr * dw
            b -= lr * db

        return w, b, cost_history

    # 随机梯度下降
    def stochastic_gradient_descent(X, y, lr=0.01, n_epochs=100):
        m = len(y)
        w, b = 0, 0
        cost_history = []

        for epoch in range(n_epochs):
            for i in range(m):
                xi = X[i].ravel()[0]
                yi = y[i]

                y_pred = w * xi + b
                cost = 0.5 * (y_pred - yi)**2

                dw = (y_pred - yi) * xi
                db = y_pred - yi

                w -= lr * dw
                b -= lr * db

            # 计算整体损失
            y_pred_all = w * X.ravel() + b
            cost = (1/(2*m)) * np.sum((y_pred_all - y)**2)
            cost_history.append(cost)

        return w, b, cost_history

    # 小批量梯度下降
    def mini_batch_gradient_descent(X, y, lr=0.01, n_epochs=100, batch_size=32):
        m = len(y)
        w, b = 0, 0
        cost_history = []

        for epoch in range(n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                batch_size_actual = len(y_batch)
                y_pred = w * X_batch.ravel() + b

                dw = (1/batch_size_actual) * np.sum((y_pred - y_batch) * X_batch.ravel())
                db = (1/batch_size_actual) * np.sum(y_pred - y_batch)

                w -= lr * dw
                b -= lr * db

            # 计算整体损失
            y_pred_all = w * X.ravel() + b
            cost = (1/(2*m)) * np.sum((y_pred_all - y)**2)
            cost_history.append(cost)

        return w, b, cost_history

    # 运行三种方法
    print("\n训练中...")
    w_bgd, b_bgd, cost_bgd = batch_gradient_descent(X, y, lr=0.1, n_iter=100)
    w_sgd, b_sgd, cost_sgd = stochastic_gradient_descent(X, y, lr=0.01, n_epochs=100)
    w_mbgd, b_mbgd, cost_mbgd = mini_batch_gradient_descent(X, y, lr=0.1, n_epochs=100)

    print(f"\n批量梯度下降: w={w_bgd:.4f}, b={b_bgd:.4f}")
    print(f"随机梯度下降: w={w_sgd:.4f}, b={b_sgd:.4f}")
    print(f"小批量梯度下降: w={w_mbgd:.4f}, b={b_mbgd:.4f}")

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cost_bgd, label='BGD', linewidth=2)
    plt.plot(cost_sgd, label='SGD', alpha=0.7)
    plt.plot(cost_mbgd, label='Mini-batch GD', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('损失函数对比')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(cost_bgd[-20:], label='BGD', linewidth=2)
    plt.plot(cost_sgd[-20:], label='SGD', alpha=0.7)
    plt.plot(cost_mbgd[-20:], label='Mini-batch GD', linewidth=2)
    plt.xlabel('Epoch (last 20)')
    plt.ylabel('Cost')
    plt.title('收敛细节')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n分析:")
    print("- BGD: 收敛平滑，但每次迭代慢")
    print("- SGD: 收敛波动大，但每次迭代快")
    print("- Mini-batch GD: 兼顾两者优点，实践中最常用")


# ========== 主函数 ==========

if __name__ == "__main__":
    # 运行所有练习答案
    exercise1_house_price_basic()
    exercise2_gradient_descent()
    exercise3_california_housing()
    exercise4_polynomial_features()
    exercise5_learning_curve()
    exercise6_feature_engineering()
    exercise7_hyperparameter_tuning()
    exercise8_salary_prediction()
    challenge_gradient_descent_comparison()

    print("\n" + "="*50)
    print("所有练习答案演示完成！")
    print("="*50)
