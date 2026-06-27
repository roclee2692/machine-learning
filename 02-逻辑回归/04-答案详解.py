"""
逻辑回归 - 练习题答案详解
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, roc_auc_score)
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
import pandas as pd

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 练习1答案：信用卡欺诈检测 ==========

def answer1_fraud_detection():
    """练习1：信用卡欺诈检测"""
    print("="*60)
    print("练习1答案：信用卡欺诈检测")
    print("="*60)

    # 生成数据
    np.random.seed(42)
    n_samples = 200

    amounts = np.random.exponential(100, n_samples)
    times = np.random.uniform(0, 24, n_samples)
    distances = np.random.uniform(0, 5000, n_samples)

    fraud_proba = (amounts/100 * 0.3 + times/24 * 0.3 + distances/5000 * 0.4)
    fraud_proba = fraud_proba / fraud_proba.max()
    y = (np.random.rand(n_samples) < fraud_proba).astype(int)

    X = np.column_stack([amounts, times, distances])

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 评估指标
    print("\n评估指标:")
    print(f"准确率:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"精确率:  {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"召回率:  {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1分数:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"AUC:    {roc_auc_score(y_test, y_proba):.4f}")

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 混淆矩阵热力图
    axes[0].imshow(cm, cmap='Blues', aspect='auto')
    axes[0].set_title('混淆矩阵')
    axes[0].set_ylabel('真实标签')
    axes[0].set_xlabel('预测标签')
    for i in range(len(cm)):
        for j in range(len(cm)):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=14)

    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_score:.3f})', linewidth=2)
    axes[1].plot([0, 1], [0, 1], 'k--', label='随机分类器')
    axes[1].set_xlabel('假正率 (FPR)')
    axes[1].set_ylabel('真正率 (TPR)')
    axes[1].set_title('ROC曲线')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 讨论
    print("\n【讨论】在欺诈检测中应该重视什么？")
    print("-" * 60)
    print("答：应该重视'召回率'（检测出真实欺诈的比例）")
    print("原因：")
    print("  1. 漏掉一个欺诈交易的代价很高（用户损失钱）")
    print("  2. 误判为欺诈的代价相对较小（多验证一下）")
    print("  3. 因此，应该降低决策阈值，提高召回率")
    print("  4. 允许精确率偏低，但要保证召回率尽可能高")


# ========== 练习2答案：手写Sigmoid和梯度下降 ==========

def answer2_sigmoid_from_scratch():
    """练习2：手写Sigmoid函数和梯度下降"""
    print("\n" + "="*60)
    print("练习2答案：手写Sigmoid和梯度下降")
    print("="*60)

    def sigmoid(z):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_cost(X, y, w, b):
        """计算交叉熵损失"""
        m = X.shape[0]
        z = np.dot(X, w) + b
        h = sigmoid(z)
        cost = (-1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        return cost

    def gradient_descent(X, y, w, b, learning_rate=0.01, n_iterations=1000):
        """梯度下降"""
        m = X.shape[0]
        costs = []

        for i in range(n_iterations):
            z = np.dot(X, w) + b
            h = sigmoid(z)

            cost = compute_cost(X, y, w, b)
            costs.append(cost)

            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)

            w -= learning_rate * dw
            b -= learning_rate * db

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

        return w, b, costs

    # 测试数据
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([0, 0, 0, 1, 1])

    # 初始化
    w = np.zeros(X.shape[1])
    b = 0

    # 梯度下降
    w, b, costs = gradient_descent(X, y, w, b, learning_rate=0.01, n_iterations=500)

    print(f"\n最终权重: {w}")
    print(f"最终偏置: {b:.4f}")

    # 绘制损失函数
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('迭代次数')
    plt.ylabel('损失函数')
    plt.title('损失函数收敛曲线')
    plt.grid(True)
    plt.show()


# ========== 练习3答案：乳腺癌分类 ==========

def answer3_breast_cancer():
    """练习3：乳腺癌分类"""
    print("\n" + "="*60)
    print("练习3答案：乳腺癌分类")
    print("="*60)

    # 加载数据
    data = load_breast_cancer()
    X = data.data
    y = data.target

    print(f"\n数据维度: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")

    # 1. 数据探索
    df = pd.DataFrame(X, columns=data.feature_names)
    print("\n特征统计信息:")
    print(df.describe())

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 超参数调优
    print("\n超参数调优 (GridSearch):")
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)

    print(f"最佳C值: {grid_search.best_params_['C']}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 5. 最终模型
    model = grid_search.best_estimator_

    # 6. 评估
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n评估结果:")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"精确率: {precision_score(y_test, y_pred):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    # 7. 特征重要性
    feature_importance = np.abs(model.coef_[0])
    top_features_idx = np.argsort(feature_importance)[-5:][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(5), feature_importance[top_features_idx])
    plt.yticks(range(5), [data.feature_names[i] for i in top_features_idx])
    plt.xlabel('权重绝对值')
    plt.title('Top 5 重要特征')
    plt.tight_layout()
    plt.show()


# ========== 练习4答案：多分类 ==========

def answer4_multiclass():
    """练习4：多分类问题"""
    print("\n" + "="*60)
    print("练习4答案：多分类（虹膜数据集）")
    print("="*60)

    # 加载数据
    data = load_iris()
    X = data.data
    y = data.target

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One-vs-Rest
    print("\n1. One-vs-Rest (OvR):")
    model_ovr = LogisticRegression(max_iter=1000, multi_class='ovr')
    model_ovr.fit(X_train_scaled, y_train)
    y_pred_ovr = model_ovr.predict(X_test_scaled)
    acc_ovr = accuracy_score(y_test, y_pred_ovr)
    print(f"准确率: {acc_ovr:.4f}")

    # Softmax
    print("\n2. Softmax (Multinomial):")
    model_softmax = LogisticRegression(max_iter=1000, multi_class='multinomial')
    model_softmax.fit(X_train_scaled, y_train)
    y_pred_softmax = model_softmax.predict(X_test_scaled)
    acc_softmax = accuracy_score(y_test, y_pred_softmax)
    print(f"准确率: {acc_softmax:.4f}")

    # 对比
    print("\n对比:")
    print(f"OvR准确率:      {acc_ovr:.4f}")
    print(f"Softmax准确率:  {acc_softmax:.4f}")

    # 混淆矩阵
    cm_softmax = confusion_matrix(y_test, y_pred_softmax)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_softmax, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title('Softmax多分类混淆矩阵')
    for i in range(len(cm_softmax)):
        for j in range(len(cm_softmax)):
            plt.text(j, i, str(cm_softmax[i, j]), ha='center', va='center', color='white')
    plt.show()


# ========== 练习5答案：类别不平衡 ==========

def answer5_class_imbalance():
    """练习5：类别不平衡处理"""
    print("\n" + "="*60)
    print("练习5答案：类别不平衡处理")
    print("="*60)

    # 生成不平衡数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        weights=[0.9, 0.1],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 方法1：直接训练
    print("\n方法1：直接训练（不处理不平衡）")
    model1 = LogisticRegression(max_iter=1000)
    model1.fit(X_train_scaled, y_train)
    y_pred1 = model1.predict(X_test_scaled)
    y_proba1 = model1.predict_proba(X_test_scaled)[:, 1]

    print(f"精确率: {precision_score(y_test, y_pred1):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred1):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred1):.4f}")
    print(f"AUC:   {roc_auc_score(y_test, y_proba1):.4f}")

    # 方法2：样本加权
    print("\n方法2：样本加权")
    model2 = LogisticRegression(max_iter=1000, class_weight='balanced')
    model2.fit(X_train_scaled, y_train)
    y_pred2 = model2.predict(X_test_scaled)
    y_proba2 = model2.predict_proba(X_test_scaled)[:, 1]

    print(f"精确率: {precision_score(y_test, y_pred2):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred2):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred2):.4f}")
    print(f"AUC:   {roc_auc_score(y_test, y_proba2):.4f}")

    # 方法3：阈值调整
    print("\n方法3：阈值调整")
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred_thresh = (y_proba1 >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_pred3 = (y_proba1 >= best_threshold).astype(int)
    print(f"最优阈值: {best_threshold:.2f}")
    print(f"精确率: {precision_score(y_test, y_pred3):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred3):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred3):.4f}")

    # 可视化对比
    methods = ['直接训练', '样本加权', '阈值调整']
    precisions = [
        precision_score(y_test, y_pred1),
        precision_score(y_test, y_pred2),
        precision_score(y_test, y_pred3)
    ]
    recalls = [
        recall_score(y_test, y_pred1),
        recall_score(y_test, y_pred2),
        recall_score(y_test, y_pred3)
    ]
    f1_scores = [
        f1_score(y_test, y_pred1),
        f1_score(y_test, y_pred2),
        f1_score(y_test, y_pred3)
    ]

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label='精确率')
    plt.bar(x, recalls, width, label='召回率')
    plt.bar(x + width, f1_scores, width, label='F1分数')
    plt.xlabel('方法')
    plt.ylabel('分数')
    plt.title('不同处理方法的对比')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ========== 练习6答案：多项式特征 ==========

def answer6_polynomial_features():
    """练习6：特征交互与多项式特征"""
    print("\n" + "="*60)
    print("练习6答案：多项式特征")
    print("="*60)

    # 生成非线性数据
    np.random.seed(42)
    n_samples = 200
    X = np.random.uniform(-3, 3, (n_samples, 2))
    z = X[:, 0]**2 + X[:, 1]**2 - 2
    y = (z < 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 基础模型
    model_basic = LogisticRegression(max_iter=1000)
    model_basic.fit(X_train, y_train)
    acc_basic = accuracy_score(y_test, model_basic.predict(X_test))
    print(f"基础模型准确率: {acc_basic:.4f}")

    # 多项式特征
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model_poly = LogisticRegression(max_iter=1000)
    model_poly.fit(X_train_poly, y_train)
    acc_poly = accuracy_score(y_test, model_poly.predict(X_test_poly))
    print(f"多项式特征模型准确率: {acc_poly:.4f}")

    # 可视化决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 基础模型
    Z_basic = model_basic.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[0].contourf(xx, yy, Z_basic, alpha=0.3, cmap='RdYlGn')
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.5)
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], c='green', alpha=0.5)
    axes[0].set_title(f'基础模型 (准确率:{acc_basic:.2%})')
    axes[0].set_xlabel('特征1')
    axes[0].set_ylabel('特征2')

    # 多项式特征
    Z_poly = model_poly.predict(poly.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z_poly, alpha=0.3, cmap='RdYlGn')
    axes[1].scatter(X[y == 0, 0], X[y == 0, 1], c='red', alpha=0.5)
    axes[1].scatter(X[y == 1, 0], X[y == 1, 1], c='green', alpha=0.5)
    axes[1].set_title(f'多项式特征模型 (准确率:{acc_poly:.2%})')
    axes[1].set_xlabel('特征1')
    axes[1].set_ylabel('特征2')

    plt.tight_layout()
    plt.show()


# ========== 主函数 ==========

if __name__ == "__main__":
    answer1_fraud_detection()
    answer2_sigmoid_from_scratch()
    answer3_breast_cancer()
    answer4_multiclass()
    answer5_class_imbalance()
    answer6_polynomial_features()

    print("\n" + "="*60)
    print("所有答案已完成！")
    print("="*60)

