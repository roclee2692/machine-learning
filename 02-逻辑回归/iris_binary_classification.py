"""
鸢尾花（Iris）数据集 - 二分类示例
===========================================
目标：区分 Setosa 和非 Setosa（二分类）  有刚毛和无刚毛
使用：逻辑回归（区分性模型）

区分性模型特点：
  - 直接学习决策边界 P(y|x)
  - 不需要对数据分布做假设
  - 专注分类边界的区分能力
  
  萼片长度原始值：4.7 cm ~ 7.9 cm
标准化后：-2.0 ~ +2.5（包含负数！）
为什么有负数？因为减去了均值，所以比均值小的都变成负数
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端确保窗口显示
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体或默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 1. 数据加载与准备 ====================
print("=" * 60)
print("鸢尾花二分类：区分 Setosa vs 非 Setosa")
print("=" * 60)

# 加载数据
iris = load_iris()
X = iris.data  # 特征：(150, 4)
y = iris.target  # 标签：0=Setosa有刚毛的  山鸢尾, 1=Versicolor 变色鸢尾 色彩斑斓的, 2=Virginica  弗吉尼亚鸢尾（一种花的品种）

# 转成二分类：Setosa(0) vs 非Setosa(1)
y_binary = (y != 0).astype(int)

print(f"\n原始数据shape: {X.shape}")
print(f"类别分布: {np.bincount(y_binary)}")
print(f"  - Setosa (0): {np.sum(y_binary == 0)}")
print(f"  - 非Setosa (1): {np.sum(y_binary == 1)}")

# ==================== 2. 特征选择与可视化 ====================
# 使用2个特征便于可视化：花萼长度(Sepal Length) 和 花瓣长度(Petal Length)
feature_indices = [0, 2]  # Sepal Length, Petal Length
X_2d = X[:, feature_indices]

print(f"\n选择特征：{iris.feature_names[feature_indices[0]]} 和 {iris.feature_names[feature_indices[1]]}")

# ==================== 3. 数据标准化 ====================
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

# ==================== 4. 数据划分 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_2d_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ==================== 5. 训练逻辑回归模型 ====================
print("\n" + "=" * 60)
print("训练逻辑回归模型")
print("=" * 60)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 模型参数
print(f"\n模型参数:")
print(f"  - 权重 w: {model.coef_[0]}")
print(f"  - 截距 b: {model.intercept_[0]}")
print(f"  - 决策函数: P(Setosa|x) = σ(w·x + b)")

# ==================== 6. 预测与评估 ====================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train) #精度
test_acc = accuracy_score(y_test, y_pred_test)
''''国际通用的区分：

术语	中文最好翻译	代码函数
Accuracy	准确率/精度	accuracy_score()
Precision	精确率	precision_score()  书上也叫准确率
Recall	召回率/灵敏度	recall_score() 查全率  
精度 (Accuracy)     = 全对/总数      ← 全体评价
精确率 (Precision)  = 预测对/预测总数 ← 预测质量
召回率 (Recall)     = 预测对/真实总数 ← 捕捉能力

accuracy_score = 整体对不对 ✅❌
precision = 你说对了的里有多少真的对
recall = 真正对的里有多少你说出来了
f1_score = (2 * precision * recall) / (precision + recall)



标准化 vs 正则化（完全不同的东西！）
特点	标准化 (Standardization)	正则化 (Regularization)
是什么	数据预处理方法	防止过拟合的技术
目的	改变数据范围	限制模型复杂度
公式	x_scaled = (x - 均值) / 标准差	Loss + λ·∑w²
结果	可能有负数	参数变小
阶段	训练前	训练时

''' 

print(f"\n性能评估:")
print(f"  - 训练集准确率: {train_acc:.4f}")
print(f"  - 测试集准确率: {test_acc:.4f}")

# 获取预测概率
y_pred_proba = model.predict_proba(X_test)
print(f"\n预测概率示例 (前5个样本):")
for i in range(min(5, len(y_test))):
    print(f"  样本{i}: P(Setosa)={y_pred_proba[i, 0]:.4f}, P(非Setosa)={y_pred_proba[i, 1]:.4f}, 真标签={y_test[i]}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n混淆矩阵:")
print(cm)

# 分类报告
print(f"\n分类报告:")
print(classification_report(y_test, y_pred_test, target_names=['Setosa', 'Non-Setosa']))

# ==================== 7. 决策边界可视化 ====================
print("\n生成决策边界图...")

# 创建网格
h = 0.02  # 网格步长
x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 预测网格上所有点的标签
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘图  axes = 坐标轴/子图（绘图对象）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：决策边界
ax = axes[0]
ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 3), colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)  # 决策边界

# 绘制训练和测试点
ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
          c='blue', marker='o', label='Setosa (Train)', s=50, edgecolors='k', alpha=0.7)
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
          c='red', marker='o', label='Non-Setosa (Train)', s=50, edgecolors='k', alpha=0.7)

ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], 
          c='blue', marker='s', label='Setosa (Test)', s=100, edgecolors='darkblue')
ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
          c='red', marker='s', label='Non-Setosa (Test)', s=100, edgecolors='darkred')

ax.set_xlabel(f"{iris.feature_names[feature_indices[0]]}", fontsize=11)
ax.set_ylabel(f"{iris.feature_names[feature_indices[1]]}", fontsize=11)
ax.set_title('Logistic Regression Decision Boundary\n逻辑回归决策边界 (二分类)', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

# 右图：混淆矩阵热力图
ax = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
           xticklabels=['Setosa', 'Non-Setosa'],
           yticklabels=['Setosa', 'Non-Setosa'])
ax.set_ylabel('True Label / 真标签')
ax.set_xlabel('Predicted Label / 预测标签')
ax.set_title(f'Confusion Matrix (Accuracy: {test_acc:.2%})\n混淆矩阵', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('iris_binary_classification.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：iris_binary_classification.png")
plt.show()

# ==================== 8. 预测概率的可视化 ====================
fig, ax = plt.subplots(figsize=(10, 6))

# 获取预测概率
y_pred_proba_all = model.predict_proba(X_2d_scaled)
proba_setosa = y_pred_proba_all[:, 0]

# 按真标签分组绘制
scatter = ax.scatter(X_2d_scaled[y_binary == 0, 0], 
                     X_2d_scaled[y_binary == 0, 1],
                     c=proba_setosa[y_binary == 0],
                     cmap='Blues', s=80, alpha=0.7, label='Setosa')

scatter = ax.scatter(X_2d_scaled[y_binary == 1, 0], 
                     X_2d_scaled[y_binary == 1, 1],
                     c=proba_setosa[y_binary == 1],
                     cmap='Reds', s=80, alpha=0.7, label='Non-Setosa')

ax.set_xlabel(iris.feature_names[feature_indices[0]], fontsize=11)
ax.set_ylabel(iris.feature_names[feature_indices[1]], fontsize=11)
ax.set_title('Prediction Probability P(Setosa|x)\n预测概率（颜色深度表示概率大小）', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('P(Setosa) / 概率', fontsize=11)
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('iris_probability_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：iris_probability_heatmap.png")
plt.show()

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
