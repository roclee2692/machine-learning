"""
快速参考：逻辑回归 & 朴素贝叶斯 - 核心代码片段
========================================================
复制这些代码片段快速开始你的项目
"""

# ======================== 1. 基础导入 ========================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# ======================== 2. 数据加载与准备 ========================

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data        # 特征
y = iris.target      # 标签
print(f"数据形状: {X.shape}")  # (150, 4)

# 二分类示例
y_binary = (y != 0).astype(int)  # Setosa(0) vs 非Setosa(1)

# 特征标准化（重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练/测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ======================== 3. 逻辑回归 - 区分性模型 ========================

# 3.1 二分类
print("\n【逻辑回归 - 二分类】")
lr_binary = LogisticRegression(random_state=42, max_iter=1000)
lr_binary.fit(X_train, y_binary[:len(X_train)])
acc = accuracy_score(y_binary[len(X_train):], lr_binary.predict(X_test))
print(f"准确率: {acc:.4f}")

# 3.2 多分类（one-vs-rest）
print("\n【逻辑回归 - 多分类】")
lr_multi = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
lr_multi.fit(X_train, y_train)
acc = accuracy_score(y_test, lr_multi.predict(X_test))
print(f"准确率: {acc:.4f}")

# 3.3 多分类（softmax/multinomial）
print("\n【逻辑回归 - Softmax】")
lr_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=1000)
lr_softmax.fit(X_train, y_train)
acc = accuracy_score(y_test, lr_softmax.predict(X_test))
print(f"准确率: {acc:.4f}")

# 3.4 获取预测概率
y_proba = lr_multi.predict_proba(X_test)  # (n_samples, n_classes)
print(f"预测概率形状: {y_proba.shape}")
print(f"第一个样本的概率: {y_proba[0]}")

# 3.5 获取模型参数
print(f"权重矩阵形状: {lr_multi.coef_.shape}")  # (n_classes, n_features)
print(f"截距: {lr_multi.intercept_}")

# 3.6 交叉验证
cv_scores = cross_val_score(lr_multi, X_scaled, y, cv=5, scoring='accuracy')
print(f"5折交叉验证: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ======================== 4. 朴素贝叶斯 - 生成性模型 ========================

print("\n【朴素贝叶斯】")
nb = GaussianNB()
nb.fit(X_train, y_train)
acc = accuracy_score(y_test, nb.predict(X_test))
print(f"准确率: {acc:.4f}")

# 4.1 获取模型参数
print(f"类别先验: {nb.class_prior_}")          # P(y)
print(f"特征均值: {nb.theta_}")                # μ
print(f"特征方差: {nb.var_}")                  # σ²

# 4.2 获取预测概率
y_proba_nb = nb.predict_proba(X_test)
print(f"预测概率形状: {y_proba_nb.shape}")

# 4.3 交叉验证
cv_scores_nb = cross_val_score(nb, X_scaled, y, cv=5, scoring='accuracy')
print(f"5折交叉验证: {cv_scores_nb.mean():.4f} ± {cv_scores_nb.std():.4f}")

# ======================== 5. 模型评估 ========================

print("\n【模型评估指标】")

y_pred = lr_multi.predict(X_test)

# 5.1 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n{cm}")

# 5.2 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5.3 按类别计算指标
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n宏平均指标:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")

# ======================== 6. 模型对比 ========================

print("\n【模型对比】")
models = {
    '逻辑回归 (OvR)': LogisticRegression(multi_class='ovr', random_state=42),
    '逻辑回归 (Softmax)': LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42),
    '朴素贝叶斯': GaussianNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:20s} | 训练: {train_acc:.4f} | 测试: {test_acc:.4f}")

# ======================== 7. 实际预测 ========================

print("\n【实际预测示例】")

# 对新数据进行预测
new_sample = X_test[0].reshape(1, -1)  # 取第一个测试样本
pred_class = lr_multi.predict(new_sample)[0]
pred_proba = lr_multi.predict_proba(new_sample)[0]
true_class = y_test[0]

print(f"样本特征: {new_sample}")
print(f"预测类别: {iris.target_names[pred_class]}")
print(f"真实类别: {iris.target_names[true_class]}")
print(f"预测概率: {pred_proba}")

# ======================== 8. 参数调优 ========================

print("\n【参数调优示例】")

# 8.1 逻辑回归的正则化强度
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
for C in C_values:
    lr = LogisticRegression(C=C, random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    acc = accuracy_score(y_test, lr.predict(X_test))
    print(f"C={C:6.3f} | 准确率: {acc:.4f}")

# 8.2 网格搜索（更全面的方法）
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳CV分数: {grid_search.best_score_:.4f}")
print(f"测试集准确率: {accuracy_score(y_test, grid_search.predict(X_test)):.4f}")

# ======================== 9. 关键代码片段速查 ========================

"""
【逻辑回归】
lr = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)

【朴素贝叶斯】
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_proba = nb.predict_proba(X_test)

【评估】
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
cross_val_score(model, X, y, cv=5)

【参数】
- LogisticRegression
  C: 正则化强度(default=1.0)，越小越强
  multi_class: 'ovr'(默认) 或 'multinomial'
  solver: 'lbfgs'、'liblinear'、'saga'等
  
- GaussianNB
  priors: 先验概率(default=None，自动计算)
  var_smoothing: 平滑参数(default=1e-9)
"""

print("\n========== 参考指南完成 ==========")
print("更多示例见: iris_binary_classification.py")
print("           iris_multiclass_classification.py")
print("           discriminative_vs_generative.py")
print("           iris_full_features_multiclass.py")
