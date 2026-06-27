"""
逻辑回归 - 完整实现
包含从零实现和scikit-learn实现
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)
from sklearn.datasets import make_classification, load_breast_cancer, load_iris
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 1. 从零实现逻辑回归 ==========

class LogisticRegressionFromScratch:
    """从零实现逻辑回归"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """训练模型"""
        m, n = X.shape

        # 初始化参数
        self.weights = np.zeros(n)
        self.bias = 0

        # 梯度下降
        for i in range(self.n_iterations):
            # 线性组合
            z = np.dot(X, self.weights) + self.bias

            # 预测概率
            y_pred = self.sigmoid(z)

            # 交叉熵损失
            cost = (-1/m) * np.sum(y * np.log(y_pred + 1e-15) +
                                   (1-y) * np.log(1-y_pred + 1e-15))
            self.cost_history.append(cost)

            # 梯度
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def plot_cost(self):
        """绘制损失函数"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('迭代次数')
        plt.ylabel('交叉熵损失')
        plt.title('损失函数收敛曲线')
        plt.grid(True)
        plt.show()


# ========== 2. 评估函数 ==========

def plot_confusion_matrix(y_true, y_pred, classes=['类别0', '类别1']):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

    return cm


def plot_roc_curve(y_true, y_proba):
    """绘制ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC曲线 (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='随机猜测')
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('ROC曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return auc


def plot_precision_recall_curve(y_true, y_proba):
    """绘制精确率-召回率曲线"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('精确率-召回率曲线')
    plt.grid(True, alpha=0.3)
    plt.show()


def evaluate_classification(y_true, y_pred, y_proba=None, model_name="模型"):
    """完整的分类评估"""
    print(f"\n{'='*50}")
    print(f"{model_name} 评估结果")
    print('='*50)

    # 基础指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = None

    print(f"\n准确率 (Accuracy):  {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall):    {rec:.4f}")
    print(f"F1分数:            {f1:.4f}")

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"AUC:              {auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n混淆矩阵:")
    print(cm)

    # 详细报告
    print(f"\n详细分类报告:")
    print(classification_report(y_true, y_pred,
                                target_names=['负类', '正类']))

    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc}


# ========== 3. 示例：二分类基础 ==========

def example1_binary_classification():
    """示例1：基础二分类 - Iris真实数据"""
    print("="*60)
    print("示例1：基础二分类 - Iris花卉分类（真实数据）")
    print("="*60)

    # 加载真实数据：Iris
    iris = load_iris()
    X_full = iris.data
    y_full = iris.target
    feature_names = iris.target_names
    features = iris.feature_names
    
    # 只取Setosa(0) vs Versicolor(1)的二分类任务
    mask = y_full < 2
    X = X_full[mask]
    y = y_full[mask]
    
    print(f"\n📊 数据信息:")
    print(f"样本数: {X.shape[0]}")
    print(f"特征数: {X.shape[1]}")
    print(f"特征名称: {list(features)}")
    print(f"分类任务: {feature_names[0]} vs {feature_names[1]}")
    print(f"类别分布: {feature_names[0]}={np.sum(y==0)}, {feature_names[1]}={np.sum(y==1)}")
    print(f"\n特征含义:")
    for i, name in enumerate(features):
        print(f"  特征{i+1}: {name} (如：花的{name})")
    print()

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练从零实现的模型
    print("\n方法1：从零实现")
    model_scratch = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
    model_scratch.fit(X_train_scaled, y_train)

    y_pred_scratch = model_scratch.predict(X_test_scaled)
    y_proba_scratch = model_scratch.predict_proba(X_test_scaled)

    evaluate_classification(y_test, y_pred_scratch, y_proba_scratch, "从零实现")
    model_scratch.plot_cost()

    # 训练sklearn模型
    print("\n方法2：sklearn实现")
    model_sklearn = LogisticRegression()
    model_sklearn.fit(X_train_scaled, y_train)

    y_pred_sklearn = model_sklearn.predict(X_test_scaled)
    y_proba_sklearn = model_sklearn.predict_proba(X_test_scaled)[:, 1]

    evaluate_classification(y_test, y_pred_sklearn, y_proba_sklearn, "sklearn")

    # 可视化决策边界（只用前两个特征）
    X_2d = X[:, :2]  # 只用前两个特征来画图
    X_train_2d, X_test_2d = X_train[:, :2], X_test[:, :2]
    
    # 用前两个特征重新训练sklearn模型用于可视化
    model_sklearn_2d = LogisticRegression()
    scaler_2d = StandardScaler()
    X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler_2d.transform(X_test_2d)
    model_sklearn_2d.fit(X_train_2d_scaled, y_test)
    
    plot_decision_boundary(X_test_2d_scaled, y_test, model_sklearn_2d, 
                          f"sklearn逻辑回归决策边界\n({iris.feature_names[0]} vs {iris.feature_names[1]})")

    # 可视化评估指标
    plot_confusion_matrix(y_test, y_pred_sklearn)
    plot_roc_curve(y_test, y_proba_sklearn)


def plot_decision_boundary(X, y, model, title="决策边界"):
    """绘制决策边界（仅适用于2D数据）"""
    if X.shape[1] != 2:
        print("只支持2维数据的决策边界可视化")
        return

    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black',
                         cmap='RdYlBu', s=50)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()


# ========== 4. 示例：乳腺癌分类 ==========

def example2_breast_cancer():
    """示例2：乳腺癌数据集分类"""
    print("\n" + "="*60)
    print("示例2：乳腺癌诊断（真实数据集）")
    print("="*60)

    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"数据维度: {X.shape}")
    print(f"特征: {feature_names[:5]}...")
    print(f"类别: 0-恶性, 1-良性")
    print(f"样本分布: 恶性={np.sum(y==0)}, 良性={np.sum(y==1)}")

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 评估
    results = evaluate_classification(y_test, y_pred, y_proba, "乳腺癌分类")

    # 可视化
    plot_confusion_matrix(y_test, y_pred, classes=['恶性', '良性'])
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)

    # 特征重要性（通过系数大小）
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': np.abs(model.coef_[0])
    }).sort_values('Coefficient', ascending=False)

    print("\n最重要的10个特征:")
    print(coef_df.head(10))

    plt.figure(figsize=(10, 8))
    coef_df.head(15).plot(x='Feature', y='Coefficient', kind='barh')
    plt.xlabel('系数绝对值')
    plt.title('特征重要性（Top 15）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ========== 5. 示例：多分类 ==========

def example3_multiclass():
    """示例3：多分类（One-vs-Rest）"""
    print("\n" + "="*60)
    print("示例3：多分类问题")
    print("="*60)

    # 生成3分类数据
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=500, centers=3, n_features=2,
                      random_state=42, cluster_std=2.0)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练多分类模型
    model = LogisticRegression(multi_class='ovr')  # One-vs-Rest
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred,
                                target_names=['类别0', '类别1', '类别2']))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['类别0', '类别1', '类别2'],
                yticklabels=['类别0', '类别1', '类别2'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('多分类混淆矩阵')
    plt.show()

    # 可视化决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black',
                         cmap='viridis', s=50)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('多分类决策边界')
    plt.colorbar(scatter)
    plt.show()


# ========== 6. 示例：阈值调整 ==========

def example4_threshold_tuning():
    """示例4：决策阈值调整"""
    print("\n" + "="*60)
    print("示例4：决策阈值对性能的影响")
    print("="*60)

    # 生成不平衡数据
    X, y = make_classification(n_samples=1000, n_features=20,
                               weights=[0.9, 0.1], random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 测试不同阈值
    thresholds = [0.3, 0.5, 0.7, 0.9]
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        print(f"\n阈值 = {threshold}:")
        print(f"  准确率: {acc:.4f}")
        print(f"  精确率: {prec:.4f}")
        print(f"  召回率: {rec:.4f}")
        print(f"  F1分数: {f1:.4f}")

    # 可视化
    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='准确率')
    plt.plot(results_df['threshold'], results_df['precision'], 's-', label='精确率')
    plt.plot(results_df['threshold'], results_df['recall'], '^-', label='召回率')
    plt.plot(results_df['threshold'], results_df['f1'], 'd-', label='F1分数')
    plt.xlabel('决策阈值')
    plt.ylabel('分数')
    plt.title('不同阈值对性能指标的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ========== 7. 示例：正则化 ==========

def example5_regularization():
    """示例5：L1和L2正则化"""
    print("\n" + "="*60)
    print("示例5：正则化对比")
    print("="*60)

    # 生成高维数据
    X, y = make_classification(n_samples=200, n_features=50,
                               n_informative=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 不同正则化
    models = {
        '无正则化': LogisticRegression(penalty=None, max_iter=10000),
        'L2正则化': LogisticRegression(penalty='l2', C=1.0, max_iter=10000),
        'L1正则化': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=10000)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'non_zero_coefs': np.sum(np.abs(model.coef_) > 0.01)
        }

        print(f"\n{name}:")
        print(f"  测试准确率: {acc:.4f}")
        print(f"  非零系数数量: {results[name]['non_zero_coefs']}")

    # 可视化系数
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, model) in enumerate(models.items()):
        axes[idx].bar(range(len(model.coef_[0])), model.coef_[0])
        axes[idx].set_xlabel('特征索引')
        axes[idx].set_ylabel('系数值')
        axes[idx].set_title(f'{name}\n非零系数: {results[name]["non_zero_coefs"]}')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ========== 主函数 ==========

if __name__ == "__main__":
    print("\n" + "🎯"*30)
    print("逻辑回归完整示例")
    print("🎯"*30 + "\n")

    # 运行所有示例
    example1_binary_classification()

    input("\n按回车键继续下一个示例...")
    example2_breast_cancer()

    input("\n按回车键继续下一个示例...")
    example3_multiclass()

    input("\n按回车键继续下一个示例...")
    example4_threshold_tuning()

    input("\n按回车键继续下一个示例...")
    example5_regularization()

    print("\n" + "="*60)
    print("✅ 所有示例运行完成！")
    print("="*60)
''