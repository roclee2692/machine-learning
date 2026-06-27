"""
决策树与随机森林 - 练习题答案详解
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 练习1答案：简单决策树 ==========

def answer1_simple_decision_tree():
    """练习1：简单决策树"""
    print("="*60)
    print("练习1答案：简单决策树")
    print("="*60)

    # 加载数据
    data = load_iris()
    X = data.data
    y = data.target

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练决策树
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)

    # 评估
    accuracy = accuracy_score(y_test, dt.predict(X_test))
    print(f"\n测试准确率: {accuracy:.4f}")

    # 可视化树
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=data.feature_names, class_names=data.target_names,
              filled=True, rounded=True, fontsize=10)
    plt.title('决策树结构 (depth=3)')
    plt.show()

    # 特征重要性
    print("\n特征重要性:")
    for i, importance in enumerate(dt.feature_importances_):
        print(f"{data.feature_names[i]}: {importance:.4f}")


# ========== 练习2答案：过拟合分析 ==========

def answer2_overfitting_analysis():
    """练习2：过拟合分析"""
    print("\n" + "="*60)
    print("练习2答案：过拟合分析")
    print("="*60)

    # 生成分类数据
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                              n_redundant=0, n_classes=2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 测试不同深度
    depths = range(1, 16)
    train_scores = []
    test_scores = []

    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        train_scores.append(dt.score(X_train, y_train))
        test_scores.append(dt.score(X_test, y_test))

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, 'o-', label='训练准确率')
    plt.plot(depths, test_scores, 's-', label='测试准确率')
    plt.xlabel('树的深度')
    plt.ylabel('准确率')
    plt.title('树深度对过拟合的影响')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 分析
    best_depth = depths[np.argmax(test_scores)]
    print(f"\n最优树深度: {best_depth}")
    print(f"该深度下的测试准确率: {test_scores[best_depth-1]:.4f}")
    print(f"过拟合开始于深度: {[i for i in range(len(depths)) if test_scores[i] < test_scores[i-1]][0] + 1 if any(test_scores[i] < test_scores[i-1] for i in range(1, len(test_scores))) else len(depths)}")


# ========== 练习3答案：随机森林 ==========

def answer3_random_forest():
    """练习3：随机森林"""
    print("\n" + "="*60)
    print("练习3答案：随机森林（乳腺癌数据集）")
    print("="*60)

    # 加载数据
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 评估
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n准确率: {accuracy:.4f}")
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['恶性', '良性']))

    # Top 5特征
    top_features_idx = np.argsort(rf.feature_importances_)[-5:][::-1]
    print(f"\nTop 5重要特征:")
    for i, idx in enumerate(top_features_idx):
        print(f"{i+1}. {data.feature_names[idx]}: {rf.feature_importances_[idx]:.4f}")


# ========== 练习4答案：树的数量影响 ==========

def answer4_num_trees_effect():
    """练习4：树的数量影响"""
    print("\n" + "="*60)
    print("练习4答案：树的数量对性能的影响")
    print("="*60)

    # 加载数据
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 测试不同数量的树
    n_trees = range(1, 101, 5)
    scores = []

    for n in n_trees:
        rf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        scores.append(rf.score(X_test, y_test))

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, scores, 'o-')
    plt.xlabel('树的数量')
    plt.ylabel('准确率')
    plt.title('随机森林中树的数量对性能的影响')
    plt.grid(True)
    plt.show()

    print(f"\n初始准确率 (1棵树): {scores[0]:.4f}")
    print(f"最终准确率 (100棵树): {scores[-1]:.4f}")
    print(f"改进幅度: {scores[-1] - scores[0]:.4f}")


# ========== 练习5答案：超参数调优 ==========

def answer5_hyperparameter_tuning():
    """练习5：超参数调优"""
    print("\n" + "="*60)
    print("练习5答案：GridSearchCV超参数调优")
    print("="*60)

    # 加载数据
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
    }

    print("\n开始网格搜索...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"\n最佳超参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 最优模型
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"测试准确率: {test_score:.4f}")


# ========== 练习6答案：决策树可视化 ==========

def answer6_tree_visualization():
    """练习6：决策树可视化"""
    print("\n" + "="*60)
    print("练习6答案：决策树可视化与分析")
    print("="*60)

    # 加载数据（只用前两个特征）
    data = load_iris()
    X = data.data[:, :2]
    y = data.target[:100]  # 二分类

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练决策树
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)

    # 可视化
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=data.feature_names[:2], class_names=['Setosa', 'Versicolor'],
              filled=True, rounded=True, fontsize=10)
    plt.title('决策树结构 (深度=3)')
    plt.show()

    print("\n树的分析:")
    print(f"树的深度: {dt.get_depth()}")
    print(f"叶子节点数: {dt.get_n_leaves()}")
    print(f"根节点的分割特征: {data.feature_names[:2][dt.tree_.feature[0]]}")
    print(f"根节点的分割阈值: {dt.tree_.threshold[0]:.4f}")


# ========== 练习7答案：多分类 ==========

def answer7_multiclass():
    """练习7：多分类决策树"""
    print("\n" + "="*60)
    print("练习7答案：多分类决策树")
    print("="*60)

    # 加载完整虹膜数据
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 决策树
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n准确率: {accuracy:.4f}")
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # 特征重要性
    print(f"\n特征重要性:")
    for i, importance in enumerate(dt.feature_importances_):
        print(f"{data.feature_names[i]}: {importance:.4f}")


# ========== 主函数 ==========

if __name__ == "__main__":
    answer1_simple_decision_tree()
    answer2_overfitting_analysis()
    answer3_random_forest()
    answer4_num_trees_effect()
    answer5_hyperparameter_tuning()
    answer6_tree_visualization()
    answer7_multiclass()

    print("\n" + "="*60)
    print("所有答案已完成！")
    print("="*60)

