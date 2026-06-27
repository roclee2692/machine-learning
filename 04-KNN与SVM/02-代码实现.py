"""
KNN与SVM - 完整实现
包含K近邻算法和支持向量机
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, roc_auc_score)
import pandas as pd

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 1. 数据生成 ==========

def generate_classification_data(n_samples=None, random_state=42):
    """使用真实房价数据集 (California Housing)

    将房价转换为分类问题：
    - 0: 低房价 (低于中位数)
    - 1: 高房价 (高于或等于中位数)
    """
    # 加载加州房价数据集
    housing = fetch_california_housing()
    X_full = housing.data
    y_continuous = housing.target

    # 使用中位数作为分类阈值
    median_price = np.median(y_continuous)
    y = (y_continuous >= median_price).astype(int)

    # 如果指定了 n_samples，随机采样
    if n_samples is not None and n_samples < len(X_full):
        np.random.seed(random_state)
        indices = np.random.choice(len(X_full), n_samples, replace=False)
        X_full = X_full[indices]
        y = y[indices]

    # 使用全部8个特征（不再限制为2个）
    X = X_full

    print(f"数据集信息:")
    print(f"  - 样本数: {len(X)}")
    print(f"  - 特征数: {X.shape[1]}")
    print(f"  - 特征: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude")
    print(f"  - 类别: 0=低房价, 1=高房价")
    print(f"  - 中位房价阈值: ${median_price * 100000:.2f}")

    return X.astype(float), y.astype(int)


# ========== 2. 评估函数 ==========

def evaluate_classifier(y_true, y_pred, y_proba=None, model_name="Model"):
    """评估分类模型"""
    print(f"\n{model_name} 评估结果:")
    print("=" * 50)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数:            {f1:.4f}")

    if y_proba is not None:
        auc_score = roc_auc_score(y_true, y_proba)
        print(f"AUC分数:           {auc_score:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_decision_boundary(X, y, model, title="决策边界", use_top_features=True):
    """绘制决策边界

    Args:
        X: 特征数据
        y: 标签
        model: 训练好的模型
        title: 图表标题
        use_top_features: 是否使用最重要的2个特征（当特征数>2时）
    """
    # 如果特征数>2，使用最重要的2个特征（MedInc和Latitude）
    if use_top_features and X.shape[1] > 2:
        # 根据特征重要性，选择 MedInc（索引0）和 Latitude（索引6）
        X_2d = X[:, [0, 6]]
        print(f"  可视化特征: MedInc (收入) 和 Latitude (纬度)")
    else:
        X_2d = X

    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 对于2个特征，直接用这2个特征训练一个简化模型进行预测
    if use_top_features and X.shape[1] > 2:
        # 用这2个特征重新训练一个简化模型用于可视化
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        if hasattr(model, 'n_neighbors'):  # KNN
            viz_model = KNeighborsClassifier(n_neighbors=model.n_neighbors)
        else:  # SVM
            viz_model = SVC(kernel=model.kernel, C=model.C)
        viz_model.fit(X_2d, y)
        Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='red', marker='o', alpha=0.5, label='低房价')
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='green', marker='s', alpha=0.5, label='高房价')

    if use_top_features and X.shape[1] > 2:
        plt.xlabel('MedInc (收入中位数)')
        plt.ylabel('Latitude (纬度)')
    else:
        plt.xlabel('MedInc (收入中位数)')
        plt.ylabel('AveRooms (平均房间数)')

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ========== 3. 示例1：KNN基础 ==========

def example1_knn_basic():
    """示例1：KNN基础"""
    print("="*50)
    print("示例1：KNN基础")
    print("="*50)

    # 生成数据
    X, y = generate_classification_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化（重要！）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练KNN
    print("\n训练KNN (K=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # 预测
    y_pred = knn.predict(X_test_scaled)

    # 评估
    evaluate_classifier(y_test, y_pred, model_name="KNN (K=5)")

    # 决策边界
    plot_decision_boundary(X_test_scaled, y_test, knn, "KNN决策边界 (K=5)")


# ========== 4. 示例2：K值的影响 ==========

def example2_k_effect():
    """示例2：K值对性能的影响"""
    print("\n" + "="*50)
    print("示例2：K值对性能的影响")
    print("="*50)

    # 生成数据
    X, y = generate_classification_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 测试不同的K值
    k_values = range(1, 31)
    train_scores = []
    test_scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        train_scores.append(knn.score(X_train_scaled, y_train))
        test_scores.append(knn.score(X_test_scaled, y_test))

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_scores, 'o-', label='训练准确率')
    plt.plot(k_values, test_scores, 's-', label='测试准确率')
    plt.xlabel('K值')
    plt.ylabel('准确率')
    plt.title('K值对KNN性能的影响')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 找最优K值
    best_k = k_values[np.argmax(test_scores)]
    print(f"\n最优K值: {best_k}")
    print(f"该K值下的测试准确率: {test_scores[best_k-1]:.4f}")


# ========== 5. 示例3：SVM基础 ==========

def example3_svm_basic():
    """示例3：SVM基础"""
    print("\n" + "="*50)
    print("示例3：SVM基础")
    print("="*50)

    # 生成数据
    X, y = generate_classification_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练SVM（RBF核）
    print("\n训练SVM (RBF核)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(X_train_scaled, y_train)

    # 预测
    y_pred = svm.predict(X_test_scaled)
    y_proba = svm.predict_proba(X_test_scaled)[:, 1]

    # 评估
    evaluate_classifier(y_test, y_pred, y_proba, model_name="SVM (RBF核)")

    # 决策边界
    plot_decision_boundary(X_test_scaled, y_test, svm, "SVM决策边界 (RBF核)")


# ========== 6. 示例4：不同核函数 ==========

def example4_kernel_comparison():
    """示例4：不同核函数对比"""
    print("\n" + "="*50)
    print("示例4：不同核函数对比")
    print("="*50)

    # 生成数据
    X, y = generate_classification_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 不同核函数
    kernels = ['linear', 'rbf', 'poly']
    results = {}

    for kernel in kernels:
        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, C=1.0, probability=True)
        else:
            svm = SVC(kernel=kernel, C=1.0, probability=True)

        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[kernel] = accuracy
        print(f"{kernel}核 准确率: {accuracy:.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    kernels_plot = ['linear', 'rbf', 'poly']

    for idx, kernel in enumerate(kernels_plot):
        if kernel == 'poly':
            model = SVC(kernel=kernel, degree=3, C=1.0)
        else:
            model = SVC(kernel=kernel, C=1.0)

        model.fit(X_train_scaled, y_train)
        plot_decision_boundary_in_subplot(X_test_scaled, y_test, model, axes[idx],
                                         f"SVM决策边界 ({kernel}核)")


def plot_decision_boundary_in_subplot(X, y, model, ax, title, use_top_features=True):
    """在子图中绘制决策边界

    Args:
        X: 特征数据
        y: 标签
        model: 训练好的模型
        ax: matplotlib axes对象
        title: 图表标题
        use_top_features: 是否使用最重要的2个特征（当特征数>2时）
    """
    # 如果特征数>2，使用最重要的2个特征（MedInc和Latitude）
    if use_top_features and X.shape[1] > 2:
        # 根据特征重要性，选择 MedInc（索引0）和 Latitude（索引6）
        X_2d = X[:, [0, 6]]
    else:
        X_2d = X

    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 对于2个特征，直接用这2个特征训练一个简化模型进行预测
    if use_top_features and X.shape[1] > 2:
        # 用这2个特征重新训练一个简化模型用于可视化
        from sklearn.svm import SVC
        viz_model = SVC(kernel=model.kernel, C=model.C)
        viz_model.fit(X_2d, y)
        Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')

    if use_top_features and X.shape[1] > 2:
        ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='red', marker='o', alpha=0.5, label='低房价')
        ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='green', marker='s', alpha=0.5, label='高房价')
        ax.set_xlabel('MedInc (收入中位数)')
        ax.set_ylabel('Latitude (纬度)')
    else:
        ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='red', marker='o', alpha=0.5, label='低房价')
        ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='green', marker='s', alpha=0.5, label='高房价')
        ax.set_xlabel('MedInc (收入中位数)')
        ax.set_ylabel('AveRooms (平均房间数)')

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


# ========== 7. 示例5：KNN vs SVM ==========

def example5_knn_vs_svm():
    """示例5：KNN vs SVM对比"""
    print("\n" + "="*50)
    print("示例5：KNN vs SVM对比")
    print("="*50)

    # 使用加州房价数据集
    X, y = generate_classification_data(n_samples=500)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    knn_score = accuracy_score(y_test, knn.predict(X_test_scaled))

    # SVM
    svm = SVC(kernel='rbf')
    svm.fit(X_train_scaled, y_train)
    svm_score = accuracy_score(y_test, svm.predict(X_test_scaled))

    print(f"\nKNN (K=3) 准确率: {knn_score:.4f}")
    print(f"SVM (RBF) 准确率: {svm_score:.4f}")

    # 对比
    print("\n对比分析:")
    print(f"{'速度': <15}{'KNN': <10}{'SVM': <10}")
    print(f"{'预测速度': <15}{'慢': <10}{'快': <10}")
    print(f"{'训练速度': <15}{'快': <10}{'慢': <10}")
    print(f"{'内存占用': <15}{'大': <10}{'小': <10}")


# ========== 8. 示例6：SVM超参数调优 ==========

def example6_svm_tuning():
    """示例6：SVM超参数调优"""
    print("\n" + "="*50)
    print("示例6：SVM超参数调优")
    print("="*50)

    # 使用加州房价数据集 (更多样本用于调优)
    X, y = generate_classification_data(n_samples=1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }

    print("\n开始网格搜索... (这可能需要几分钟)")
    svm = SVC(probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    print(f"\n最佳超参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 最优模型
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"测试准确率: {test_score:.4f}")


# ========== 主函数 ==========

if __name__ == "__main__":
    # 运行所有示例
    example1_knn_basic()
    example2_k_effect()
    example3_svm_basic()
    example4_kernel_comparison()
    example5_knn_vs_svm()
    example6_svm_tuning()

    print("\n" + "="*50)
    print("所有示例运行完成！")
    print("="*50)

