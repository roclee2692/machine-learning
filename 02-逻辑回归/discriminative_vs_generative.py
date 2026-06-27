"""
区分性模型 vs 生成性模型：完整对比
===========================================
演示：
  1. 逻辑回归（Logistic Regression） - 区分性模型
  2. 朴素贝叶斯（Naive Bayes） - 生成性模型

核心对比：
  - 区分性：直接学习 P(y|x) 的决策边界
  - 生成性：先学 P(x|y) 和 P(y)，再用贝叶斯公式求 P(y|x)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据准备 ====================
print("=" * 70)
print("区分性模型 vs 生成性模型对比")
print("=" * 70)

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 二分类：Setosa vs 非Setosa
y_binary = (y != 0).astype(int)
binary_names = ['Setosa', 'Non-Setosa']

# 使用2个特征
feature_indices = [0, 2]  # Sepal Length, Petal Length
X_2d = X[:, feature_indices]

scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

X_train, X_test, y_train, y_test = train_test_split(
    X_2d_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print(f"\n数据集: {X_train.shape[0]} 训练样本, {X_test.shape[0]} 测试样本")
print(f"类别分布: Setosa={np.sum(y_binary==0)}, Non-Setosa={np.sum(y_binary==1)}")

# ==================== 2. 模型训练 ====================
print("\n" + "=" * 70)
print("2. 模型训练与评估")
print("=" * 70)

# ========== 逻辑回归（区分性模型） ==========
print("\n【逻辑回归 - 区分性模型】")
print("-" * 70)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)

print(f"训练准确率: {lr_train_acc:.4f}")
print(f"测试准确率: {lr_test_acc:.4f}")
print(f"模型参数:")
print(f"  - 权重 w: {lr_model.coef_[0]}")
print(f"  - 截距 b: {lr_model.intercept_[0]:.4f}")
print(f"  - 决策函数: P(y|x) = σ(w·x + b)")

# ========== 朴素贝叶斯（生成性模型） ==========
print("\n【朴素贝叶斯 - 生成性模型】")
print("-" * 70)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_train_acc = accuracy_score(y_train, nb_model.predict(X_train))
nb_test_acc = accuracy_score(y_test, nb_model.predict(X_test))
nb_pred = nb_model.predict(X_test)
nb_proba = nb_model.predict_proba(X_test)

print(f"训练准确率: {nb_train_acc:.4f}")
print(f"测试准确率: {nb_test_acc:.4f}")
print(f"模型参数（per class均值和方差）:")
for i in range(2):
    print(f"  - 类别 {binary_names[i]}:")
    print(f"    均值: {nb_model.theta_[i]}")
    print(f"    方差: {nb_model.var_[i]}")
print(f"  - 先验概率 P(y):")
for i, name in enumerate(binary_names):
    print(f"    P({name}) = {nb_model.class_prior_[i]:.4f}")

# ==================== 3. 性能对比 ====================
print("\n" + "=" * 70)
print("3. 模型性能对比")
print("=" * 70)

comparison_data = {
    '指标': ['训练准确率', '测试准确率', '过拟合度(train-test)'],
    '逻辑回归': [f'{lr_train_acc:.4f}', f'{lr_test_acc:.4f}', f'{lr_train_acc-lr_test_acc:.4f}'],
    '朴素贝叶斯': [f'{nb_train_acc:.4f}', f'{nb_test_acc:.4f}', f'{nb_train_acc-nb_test_acc:.4f}']
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

# ==================== 4. 决策边界对比可视化 ====================
print("\n生成决策边界对比图...")

h = 0.02
x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ========== 左：逻辑回归 ==========
Z_lr = lr_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lr = Z_lr.reshape(xx.shape)

ax = axes[0]
ax.contourf(xx, yy, Z_lr, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.contour(xx, yy, Z_lr, levels=[0.5], colors='black', linewidths=2)

ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
          c='blue', marker='o', label='Setosa', s=50, edgecolors='k', alpha=0.7)
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
          c='red', marker='o', label='Non-Setosa', s=50, edgecolors='k', alpha=0.7)

ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
          c='blue', marker='s', s=100, edgecolors='darkblue')
ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
          c='red', marker='s', s=100, edgecolors='darkred')

ax.set_xlabel(iris.feature_names[feature_indices[0]], fontsize=11)
ax.set_ylabel(iris.feature_names[feature_indices[1]], fontsize=11)
ax.set_title(f'Logistic Regression (Discriminative)\n逻辑回归（区分性）\nAccuracy: {lr_test_acc:.2%}', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

# ========== 右：朴素贝叶斯 ==========
Z_nb = nb_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_nb = Z_nb.reshape(xx.shape)

ax = axes[1]
ax.contourf(xx, yy, Z_nb, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.contour(xx, yy, Z_nb, levels=[0.5], colors='black', linewidths=2)

ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
          c='blue', marker='o', label='Setosa', s=50, edgecolors='k', alpha=0.7)
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
          c='red', marker='o', label='Non-Setosa', s=50, edgecolors='k', alpha=0.7)

ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
          c='blue', marker='s', s=100, edgecolors='darkblue')
ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
          c='red', marker='s', s=100, edgecolors='darkred')

ax.set_xlabel(iris.feature_names[feature_indices[0]], fontsize=11)
ax.set_ylabel(iris.feature_names[feature_indices[1]], fontsize=11)
ax.set_title(f'Naive Bayes (Generative)\n朴素贝叶斯（生成性）\nAccuracy: {nb_test_acc:.2%}', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('discriminative_vs_generative.png', dpi=150, bbox_inches='tight')
print("[OK] 已保存：discriminative_vs_generative.png")
plt.show()  # 用户关闭窗口后继续

# ==================== 5. 混淆矩阵对比 ====================
print("\n生成混淆矩阵对比图...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 逻辑回归混淆矩阵
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True,
           xticklabels=binary_names, yticklabels=binary_names)
axes[0].set_ylabel('True Label / 真标签')
axes[0].set_xlabel('Predicted Label / 预测标签')
axes[0].set_title(f'Logistic Regression\n逻辑回归 (Accuracy: {lr_test_acc:.2%})', fontweight='bold')

# 朴素贝叶斯混淆矩阵
cm_nb = confusion_matrix(y_test, nb_pred)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=True,
           xticklabels=binary_names, yticklabels=binary_names)
axes[1].set_ylabel('True Label / 真标签')
axes[1].set_xlabel('Predicted Label / 预测标签')
axes[1].set_title(f'Naive Bayes\n朴素贝叶斯 (Accuracy: {nb_test_acc:.2%})', fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
print("[OK] 已保存：confusion_matrix_comparison.png")
plt.show()

# ==================== 6. 预测概率分布对比 ====================
print("\n生成预测概率分布对比图...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 逻辑回归
ax = axes[0]
proba_lr_setosa = lr_proba[:, 0]
ax.hist(proba_lr_setosa[y_test == 0], bins=15, alpha=0.6, label='True Setosa / 真实Setosa', color='blue', edgecolor='black')
ax.hist(proba_lr_setosa[y_test == 1], bins=15, alpha=0.6, label='True Non-Setosa / 真实非Setosa', color='red', edgecolor='black')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold / 决策阈值')
ax.set_xlabel('P(Setosa|x)', fontsize=11)
ax.set_ylabel('Frequency / 频数')
ax.set_title('Logistic Regression - Probability Distribution\n逻辑回归 - 预测概率分布', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# 朴素贝叶斯
ax = axes[1]
proba_nb_setosa = nb_proba[:, 0]
ax.hist(proba_nb_setosa[y_test == 0], bins=15, alpha=0.6, label='True Setosa / 真实Setosa', color='blue', edgecolor='black')
ax.hist(proba_nb_setosa[y_test == 1], bins=15, alpha=0.6, label='True Non-Setosa / 真实非Setosa', color='red', edgecolor='black')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold / 决策阈值')
ax.set_xlabel('P(Setosa|x)', fontsize=11)
ax.set_ylabel('Frequency / 频数')
ax.set_title('Naive Bayes - Probability Distribution\n朴素贝叶斯 - 预测概率分布', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('probability_distribution_comparison.png', dpi=150, bbox_inches='tight')
print("[OK] 已保存：probability_distribution_comparison.png")
plt.show()

# ==================== 7. 概率热力图对比 ====================
print("\n生成概率热力图对比...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 逻辑回归概率
Z_lr_proba = lr_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
Z_lr_proba = Z_lr_proba.reshape(xx.shape)

ax = axes[0]
contourf = ax.contourf(xx, yy, Z_lr_proba, levels=15, cmap='RdBu_r', alpha=0.8)
ax.scatter(X_2d_scaled[y_binary == 0, 0], X_2d_scaled[y_binary == 0, 1],
          c='blue', marker='o', s=40, edgecolors='white', linewidth=0.5, alpha=0.7)
ax.scatter(X_2d_scaled[y_binary == 1, 0], X_2d_scaled[y_binary == 1, 1],
          c='red', marker='^', s=40, edgecolors='white', linewidth=0.5, alpha=0.7)
ax.set_xlabel(iris.feature_names[feature_indices[0]], fontsize=11)
ax.set_ylabel(iris.feature_names[feature_indices[1]], fontsize=11)
ax.set_title('Logistic Regression - P(Setosa|x) Heatmap\n逻辑回归 - P(Setosa|x) 热力图', fontsize=12, fontweight='bold')
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label('Probability / 概率', fontsize=10)

# 朴素贝叶斯概率
Z_nb_proba = nb_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
Z_nb_proba = Z_nb_proba.reshape(xx.shape)

ax = axes[1]
contourf = ax.contourf(xx, yy, Z_nb_proba, levels=15, cmap='RdBu_r', alpha=0.8)
ax.scatter(X_2d_scaled[y_binary == 0, 0], X_2d_scaled[y_binary == 0, 1],
          c='blue', marker='o', s=40, edgecolors='white', linewidth=0.5, alpha=0.7)
ax.scatter(X_2d_scaled[y_binary == 1, 0], X_2d_scaled[y_binary == 1, 1],
          c='red', marker='^', s=40, edgecolors='white', linewidth=0.5, alpha=0.7)
ax.set_xlabel(iris.feature_names[feature_indices[0]], fontsize=11)
ax.set_ylabel(iris.feature_names[feature_indices[1]], fontsize=11)
ax.set_title('Naive Bayes - P(Setosa|x) Heatmap\n朴素贝叶斯 - P(Setosa|x) 热力图', fontsize=12, fontweight='bold')
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label('Probability / 概率', fontsize=10)

plt.tight_layout()
plt.savefig('probability_heatmap_comparison.png', dpi=150, bbox_inches='tight')
print("[OK] 已保存：probability_heatmap_comparison.png")
plt.show()

# ==================== 8. 详细分类报告 ====================
print("\n" + "=" * 70)
print("4. 详细分类报告")
print("=" * 70)

print("\n【逻辑回归】")
print("-" * 70)
print(classification_report(y_test, lr_pred, target_names=binary_names))

print("\n【朴素贝叶斯】")
print("-" * 70)
print(classification_report(y_test, nb_pred, target_names=binary_names))

# ==================== 9. 概念总结表 ====================
print("\n" + "=" * 70)
print("5. 区分性模型 vs 生成性模型 - 概念总结")
print("=" * 70)

summary_table = pd.DataFrame({
    '维度': [
        '建模目标',
        '数学形式',
        '核心思想',
        '代表模型',
        '关键特点',
        '优点',
        '缺点',
        '适用场景'
    ],
    '区分性模型 (Discriminative)': [
        '直接学习决策边界 P(y|x)',
        'P(C₁|x) = σ(w·x + b)',
        '不关心数据是怎么生成的，只找 "怎么区分" 的规则',
        '逻辑回归、SVM、决策树、神经网络',
        '1. 专注分类边界，区分能力强\n2. 不需要对数据分布假设\n3. 不能生成新样本',
        '准确率通常较高，特别是边界区分',
        '对边界外的数据泛化能力弱，容易过拟合',
        '分类、回归等预测任务'
    ],
    '生成性模型 (Generative)': [
        '先建模数据分布 P(x|y)，再求 P(y|x)',
        'P(y|x) = P(x|y)P(y) / P(x)',
        '先学每一类数据 "长什么样"，再看新样本更像哪一类的分布',
        '朴素贝叶斯、GDA、HMM、GAN、VAE',
        '1. 对数据分布有明确假设\n2. 抗噪声、对小数据鲁棒\n3. 可以生成新的模拟样本',
        '在数据不足、噪声多时更稳定；可生成数据',
        '分布假设错误时性能下降；计算量通常较大',
        '数据不足、噪声多的场景；数据生成任务'
    ]
})

print("\n" + summary_table.to_string(index=False))

print("\n" + "=" * 70)
print("完成！已生成所有对比图表。")
print("=" * 70)
