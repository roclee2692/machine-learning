"""
鸢尾花全特征多分类 - 完整示例
===========================================
特点：
  - 使用全部4个特征（Sepal Length, Sepal Width, Petal Length, Petal Width）
  - 3分类（Setosa, Versicolor, Virginica）
  - 同时演示逻辑回归和朴素贝叶斯
  - 详细的模型评估和交叉验证
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score
)
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据准备 ====================
print("=" * 70)
print("鸢尾花全特征多分类 - 完整示例")
print("=" * 70)

iris = load_iris()
X = iris.data  # 全部4个特征
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print(f"\n数据集信息:")
print(f"  - 样本数: {X.shape[0]}")
print(f"  - 特征数: {X.shape[1]}")
print(f"  - 特征: {feature_names}")
print(f"  - 类别数: {len(target_names)}")
print(f"  - 类别: {target_names}")
print(f"  - 类别分布: {np.bincount(y)}")

# ==================== 2. 数据标准化 ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n数据划分:")
print(f"  - 训练集: {X_train.shape[0]} 样本")
print(f"  - 测试集: {X_test.shape[0]} 样本")

# ==================== 3. 模型训练与评估 ====================
print("\n" + "=" * 70)
print("模型训练与评估")
print("=" * 70)

# ========== 逻辑回归 ==========
print("\n【逻辑回归 - one-vs-rest】")
print("-" * 70)

lr_model = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)
lr_train_proba = lr_model.predict_proba(X_train)
lr_test_proba = lr_model.predict_proba(X_test)

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)

print(f"准确率:")
print(f"  - 训练集: {lr_train_acc:.4f}")
print(f"  - 测试集: {lr_test_acc:.4f}")

# 计算其他指标
lr_f1_macro = f1_score(y_test, lr_test_pred, average='macro')
lr_precision_macro = precision_score(y_test, lr_test_pred, average='macro')
lr_recall_macro = recall_score(y_test, lr_test_pred, average='macro')

print(f"宏平均指标:")
print(f"  - F1 Score: {lr_f1_macro:.4f}")
print(f"  - Precision: {lr_precision_macro:.4f}")
print(f"  - Recall: {lr_recall_macro:.4f}")

# 交叉验证
lr_cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"5折交叉验证准确率: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

# ========== 朴素贝叶斯 ==========
print("\n【朴素贝叶斯 - 高斯分布】")
print("-" * 70)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_train_pred = nb_model.predict(X_train)
nb_test_pred = nb_model.predict(X_test)
nb_train_proba = nb_model.predict_proba(X_train)
nb_test_proba = nb_model.predict_proba(X_test)

nb_train_acc = accuracy_score(y_train, nb_train_pred)
nb_test_acc = accuracy_score(y_test, nb_test_pred)

print(f"准确率:")
print(f"  - 训练集: {nb_train_acc:.4f}")
print(f"  - 测试集: {nb_test_acc:.4f}")

# 计算其他指标
nb_f1_macro = f1_score(y_test, nb_test_pred, average='macro')
nb_precision_macro = precision_score(y_test, nb_test_pred, average='macro')
nb_recall_macro = recall_score(y_test, nb_test_pred, average='macro')

print(f"宏平均指标:")
print(f"  - F1 Score: {nb_f1_macro:.4f}")
print(f"  - Precision: {nb_precision_macro:.4f}")
print(f"  - Recall: {nb_recall_macro:.4f}")

# 交叉验证
nb_cv_scores = cross_val_score(nb_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"5折交叉验证准确率: {nb_cv_scores.mean():.4f} ± {nb_cv_scores.std():.4f}")

# ==================== 4. 性能对比表 ====================
print("\n" + "=" * 70)
print("性能对比总结")
print("=" * 70)

comparison_df = pd.DataFrame({
    '指标': [
        '训练准确率',
        '测试准确率',
        'F1 Score (macro)',
        'Precision (macro)',
        'Recall (macro)',
        '交叉验证均值',
        '交叉验证标准差',
        '过拟合度 (train-test)'
    ],
    '逻辑回归': [
        f'{lr_train_acc:.4f}',
        f'{lr_test_acc:.4f}',
        f'{lr_f1_macro:.4f}',
        f'{lr_precision_macro:.4f}',
        f'{lr_recall_macro:.4f}',
        f'{lr_cv_scores.mean():.4f}',
        f'{lr_cv_scores.std():.4f}',
        f'{lr_train_acc - lr_test_acc:.4f}'
    ],
    '朴素贝叶斯': [
        f'{nb_train_acc:.4f}',
        f'{nb_test_acc:.4f}',
        f'{nb_f1_macro:.4f}',
        f'{nb_precision_macro:.4f}',
        f'{nb_recall_macro:.4f}',
        f'{nb_cv_scores.mean():.4f}',
        f'{nb_cv_scores.std():.4f}',
        f'{nb_train_acc - nb_test_acc:.4f}'
    ]
})

print("\n" + comparison_df.to_string(index=False))

# ==================== 5. 混淆矩阵对比 ====================
print("\n生成混淆矩阵对比...")

cm_lr = confusion_matrix(y_test, lr_test_pred)
cm_nb = confusion_matrix(y_test, nb_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 逻辑回归
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True,
           xticklabels=target_names, yticklabels=target_names)
axes[0].set_ylabel('True Label / 真标签')
axes[0].set_xlabel('Predicted Label / 预测标签')
axes[0].set_title(f'Logistic Regression\n逻辑回归 (Accuracy: {lr_test_acc:.2%})', fontweight='bold')

# 朴素贝叶斯
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=True,
           xticklabels=target_names, yticklabels=target_names)
axes[1].set_ylabel('True Label / 真标签')
axes[1].set_xlabel('Predicted Label / 预测标签')
axes[1].set_title(f'Naive Bayes\n朴素贝叶斯 (Accuracy: {nb_test_acc:.2%})', fontweight='bold')

plt.tight_layout()
plt.savefig('full_features_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：full_features_confusion_matrix.png")
plt.show()

# ==================== 6. 分类报告 ====================
print("\n" + "=" * 70)
print("详细分类报告")
print("=" * 70)

print("\n【逻辑回归分类报告】")
print("-" * 70)
print(classification_report(y_test, lr_test_pred, target_names=target_names))

print("\n【朴素贝叶斯分类报告】")
print("-" * 70)
print(classification_report(y_test, nb_test_pred, target_names=target_names))

# ==================== 7. 交叉验证对比 ====================
print("\n生成交叉验证对比图...")

fig, ax = plt.subplots(figsize=(10, 5))

x_pos = np.arange(5)
width = 0.35

ax.bar(x_pos - width/2, lr_cv_scores, width, label='Logistic Regression / 逻辑回归', alpha=0.8, color='skyblue', edgecolor='black')
ax.bar(x_pos + width/2, nb_cv_scores, width, label='Naive Bayes / 朴素贝叶斯', alpha=0.8, color='lightgreen', edgecolor='black')

ax.axhline(y=lr_cv_scores.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'LR Mean: {lr_cv_scores.mean():.4f}')
ax.axhline(y=nb_cv_scores.mean(), color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'NB Mean: {nb_cv_scores.mean():.4f}')

ax.set_xlabel('Fold', fontsize=11)
ax.set_ylabel('Accuracy / 准确率', fontsize=11)
ax.set_title('5-Fold Cross Validation Accuracy\n5折交叉验证准确率对比', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_ylim([0.9, 1.0])
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cross_validation_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：cross_validation_comparison.png")
plt.show()

# ==================== 8. 每个类别的性能对比 ====================
print("\n生成各类别性能对比图...")

# 计算每个类别的指标
from sklearn.metrics import precision_recall_fscore_support

lr_precision_per_class, lr_recall_per_class, lr_f1_per_class, _ = precision_recall_fscore_support(
    y_test, lr_test_pred, labels=range(3)
)
nb_precision_per_class, nb_recall_per_class, nb_f1_per_class, _ = precision_recall_fscore_support(
    y_test, nb_test_pred, labels=range(3)
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# F1 Score
x = np.arange(len(target_names))
width = 0.35

axes[0].bar(x - width/2, lr_f1_per_class, width, label='Logistic Regression / 逻辑回归', alpha=0.8, edgecolor='black')
axes[0].bar(x + width/2, nb_f1_per_class, width, label='Naive Bayes / 朴素贝叶斯', alpha=0.8, edgecolor='black')
axes[0].set_ylabel('F1 Score')
axes[0].set_title('F1 Score - Per Class')
axes[0].set_xticks(x)
axes[0].set_xticklabels(target_names)
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')
axes[0].set_ylim([0, 1.1])

# Precision
axes[1].bar(x - width/2, lr_precision_per_class, width, label='Logistic Regression / 逻辑回归', alpha=0.8, edgecolor='black')
axes[1].bar(x + width/2, nb_precision_per_class, width, label='Naive Bayes / 朴素贝叶斯', alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision - Per Class')
axes[1].set_xticks(x)
axes[1].set_xticklabels(target_names)
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')
axes[1].set_ylim([0, 1.1])

# Recall
axes[2].bar(x - width/2, lr_recall_per_class, width, label='Logistic Regression / 逻辑回归', alpha=0.8, edgecolor='black')
axes[2].bar(x + width/2, nb_recall_per_class, width, label='Naive Bayes / 朴素贝叶斯', alpha=0.8, edgecolor='black')
axes[2].set_ylabel('Recall')
axes[2].set_title('Recall - Per Class')
axes[2].set_xticks(x)
axes[2].set_xticklabels(target_names)
axes[2].legend()
axes[2].grid(alpha=0.3, axis='y')
axes[2].set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('per_class_performance.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：per_class_performance.png")
plt.show()

# ==================== 9. 预测概率分布 ====================
print("\n生成预测概率分布...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for i, name in enumerate(target_names):
    # 逻辑回归
    ax = axes[0, i]
    proba_lr = lr_test_proba[:, i]
    ax.hist(proba_lr[y_test == i], bins=15, alpha=0.7, label='True / 真实', color='blue', edgecolor='black')
    ax.hist(proba_lr[y_test != i], bins=15, alpha=0.7, label='False / 错误', color='red', edgecolor='black')
    ax.axvline(x=1/3, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel(f'P({name}|x)')
    ax.set_ylabel('Frequency / 频数')
    ax.set_title(f'Logistic Regression - {name}')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 朴素贝叶斯
    ax = axes[1, i]
    proba_nb = nb_test_proba[:, i]
    ax.hist(proba_nb[y_test == i], bins=15, alpha=0.7, label='True / 真实', color='blue', edgecolor='black')
    ax.hist(proba_nb[y_test != i], bins=15, alpha=0.7, label='False / 错误', color='red', edgecolor='black')
    ax.axvline(x=1/3, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel(f'P({name}|x)')
    ax.set_ylabel('Frequency / 频数')
    ax.set_title(f'Naive Bayes - {name}')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('probability_distributions_full.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：probability_distributions_full.png")
plt.show()

# ==================== 10. 一些实际预测示例 ====================
print("\n" + "=" * 70)
print("实际预测示例（前10个测试样本）")
print("=" * 70)

df_examples = pd.DataFrame({
    '特征': [f'样本{i}' for i in range(10)],
    '真标签': [target_names[y_test[i]] for i in range(10)],
    'LR预测': [target_names[lr_test_pred[i]] for i in range(10)],
    'LR概率': [f"{lr_test_proba[i, lr_test_pred[i]]:.4f}" for i in range(10)],
    'NB预测': [target_names[nb_test_pred[i]] for i in range(10)],
    'NB概率': [f"{nb_test_proba[i, nb_test_pred[i]]:.4f}" for i in range(10)],
    '结果': [
        '✓' if (lr_test_pred[i] == y_test[i] and nb_test_pred[i] == y_test[i]) 
        else '✗' if (lr_test_pred[i] != y_test[i] and nb_test_pred[i] != y_test[i])
        else '~'
        for i in range(10)
    ]
})

print("\n" + df_examples.to_string(index=False))
print("\n说明: ✓ = 两个模型都正确  ✗ = 两个模型都错误  ~ = 一个正确一个错误")

print("\n" + "=" * 70)
print("完成！所有图表已生成。")
print("=" * 70)
