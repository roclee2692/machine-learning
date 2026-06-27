# 鸢尾花（Iris）分类完整指南

## 概述

本文件夹包含了使用 **逻辑回归（Logistic Regression）** 和 **朴素贝叶斯（Naive Bayes）** 进行鸢尾花数据集分类的完整示例。通过这些代码，你可以：

1. ✓ 掌握二分类和多分类的实现方法
2. ✓ 理解区分性模型 vs 生成性模型的核心差异
3. ✓ 学会如何评估和对比不同的分类模型
4. ✓ 掌握交叉验证和性能指标

---

## 文件说明

### 1. `iris_binary_classification.py` - 二分类示例
**目标**: 区分 Setosa 和非 Setosa（二分类）

**特点**:
- 使用2个特征（Sepal Length, Petal Length）便于可视化
- 逻辑回归模型
- 详细的决策边界可视化
- 预测概率的热力图

**关键代码**:
```python
# 转成二分类
y_binary = (y != 0).astype(int)  # Setosa(0) vs 非Setosa(1)

# 训练逻辑回归
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 模型性质
print(f"决策函数: P(Setosa|x) = σ(w·x + b)")
```

**输出**:
- `iris_binary_classification.png` - 决策边界和混淆矩阵
- `iris_probability_heatmap.png` - 预测概率的分布

---

### 2. `iris_multiclass_classification.py` - 多分类示例
**目标**: 分类3个类别（Setosa, Versicolor, Virginica）

**特点**:
- 逻辑回归的 **one-vs-rest (OvR)** 策略
- 为每个类别训练一个二分类器
- 3个预测概率的热力图（每类一个）
- 详细的决策区域可视化

**关键代码**:
```python
# 多分类逻辑回归（one-vs-rest）
model = LogisticRegression(multi_class='ovr', random_state=42)
model.fit(X_train, y_train)

# OvR原理：
# 模型1: P(Setosa|x) vs 非Setosa
# 模型2: P(Versicolor|x) vs 非Versicolor
# 模型3: P(Virginica|x) vs 非Virginica
# 预测：选择概率最高的类别
```

**输出**:
- `iris_multiclass_classification.png` - 决策边界和混淆矩阵
- `iris_probability_distribution.png` - 各类别预测概率的分布
- `iris_probability_heatmap_multiclass.png` - 每个类别的概率热力图

---

### 3. `discriminative_vs_generative.py` - 模型对比
**目标**: 深入理解**区分性模型**和**生成性模型**的差异

**对比的两个模型**:
- **逻辑回归** - 区分性模型（Discriminative）
- **朴素贝叶斯** - 生成性模型（Generative）

**关键概念**:

#### 区分性模型（逻辑回归）
```
直接学习 P(y|x)
数学形式：P(C₁|x) = σ(w·x + b)
特点：只学 "怎么分"，不学 "数据长什么样"
```

#### 生成性模型（朴素贝叶斯）
```
先学 P(x|y) 和 P(y)，再用贝叶斯公式：
P(y|x) = P(x|y)·P(y) / P(x)

特点：先学 "数据长什么样"，再看 "怎么分"
```

**输出**:
- `discriminative_vs_generative.png` - 决策边界对比
- `confusion_matrix_comparison.png` - 混淆矩阵对比
- `probability_distribution_comparison.png` - 概率分布对比
- `probability_heatmap_comparison.png` - 概率热力图对比

---

### 4. `iris_full_features_multiclass.py` - 完整示例
**目标**: 使用全部4个特征的多分类完整示例

**特点**:
- 使用全部4个特征（Sepal Length, Sepal Width, Petal Length, Petal Width）
- 同时演示逻辑回归和朴素贝叶斯
- 详细的性能评估（准确率、F1、Precision、Recall）
- 5折交叉验证
- 逐个类别的性能分析

**关键指标**:
```python
# 准确率、F1 Score、Precision、Recall
# 交叉验证（Cross-Validation）
# 混淆矩阵
# 分类报告
```

**输出**:
- `full_features_confusion_matrix.png` - 混淆矩阵
- `cross_validation_comparison.png` - 交叉验证对比
- `per_class_performance.png` - 各类别性能对比
- `probability_distributions_full.png` - 所有类别的概率分布

---

## 核心概念总结

### 区分性 vs 生成性模型对比表

| 对比维度 | 区分性模型（Discriminative） | 生成性模型（Generative） |
|---------|--------------------------|------------------------|
| **建模目标** | 直接学习 P(y\|x) | 先学 P(x\|y)、P(y)，再用贝叶斯求 P(y\|x) |
| **数学形式** | P(C₁\|x) = σ(w·x + b) | P(y\|x) = P(x\|y)·P(y) / P(x) |
| **核心思想** | 只找 "怎么区分" 的规则 | 先学 "数据长什么样"，再看 "怎么分" |
| **代表模型** | 逻辑回归、SVM、决策树、神经网络 | 朴素贝叶斯、高斯判别分析、HMM |
| **关键特点** | 1. 边界区分能力强 2. 不需要分布假设 3. 不能生成新数据 | 1. 对分布有明确假设 2. 抗噪声、数据少时稳定 3. 能生成新数据 |
| **优点** | 准确率通常较高，专注分类边界 | 数据不足、噪声多时更稳定；可生成样本 |
| **缺点** | 边界外泛化能力弱，容易过拟合 | 分布假设错误时性能下降；计算量大 |
| **适用场景** | 分类预测任务，追求准确率优先 | 数据不足、噪声多；需要数据生成 |

---

## 逻辑回归的数学原理

### 1. 二分类情况
对于两个类别 $C_0$ 和 $C_1$，逻辑回归通过 **sigmoid 函数**将线性输出映射到概率：

$$P(C_1|x) = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

其中：
- $w$ 是权重向量
- $b$ 是截距（偏置）
- $\sigma$ 是 sigmoid 函数

**决策规则**: 
- 如果 $P(C_1|x) > 0.5$，预测为 $C_1$
- 否则预测为 $C_0$
- 决策边界在 $w \cdot x + b = 0$

### 2. 多分类情况（One-vs-Rest）
对于 $K$ 个类别，为每一个类别 $k$ 训练一个二分类器：

$$P(C_k|x) = \sigma(w_k \cdot x + b_k)$$

**预测**：选择概率最高的类别
$$\hat{y} = \arg\max_k P(C_k|x)$$

### 3. 成本函数（交叉熵损失）
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

训练通过梯度下降最小化这个成本函数。

---

## 朴素贝叶斯的数学原理

### 1. 贝叶斯定理
$$P(C_k|x) = \frac{P(x|C_k) \cdot P(C_k)}{P(x)}$$

其中：
- $P(C_k|x)$ - 后验概率（我们要求的）
- $P(x|C_k)$ - 似然度（给定类别下的数据分布）
- $P(C_k)$ - 先验概率（类别出现的概率）
- $P(x)$ - 证据（所有类别的归一化）

### 2. 朴素假设
假设给定类别时，特征之间**相互独立**：

$$P(x|C_k) = P(x_1|C_k) \cdot P(x_2|C_k) \cdot ... \cdot P(x_n|C_k) = \prod_{j=1}^{n} P(x_j|C_k)$$

### 3. 高斯分布假设
对于连续特征，假设满足高斯分布：

$$P(x_j|C_k) = \frac{1}{\sqrt{2\pi\sigma_{j,k}^2}} \exp\left(-\frac{(x_j - \mu_{j,k})^2}{2\sigma_{j,k}^2}\right)$$

其中 $\mu_{j,k}$ 和 $\sigma_{j,k}^2$ 是类别 $k$ 中特征 $j$ 的均值和方差。

### 4. 预测
$$\hat{C} = \arg\max_k P(C_k) \prod_{j=1}^{n} P(x_j|C_k)$$

---

## 如何运行这些代码

### 安装依赖
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 运行示例
```bash
# 二分类示例
python iris_binary_classification.py

# 多分类示例
python iris_multiclass_classification.py

# 区分性 vs 生成性对比
python discriminative_vs_generative.py

# 完整示例（全特征）
python iris_full_features_multiclass.py
```

每个脚本都会生成相应的可视化图表和详细的控制台输出。

---

## 预期结果

### 性能指标
- **逻辑回归**: 在鸢尾花数据集上通常能达到 95%+ 的准确率
- **朴素贝叶斯**: 通常能达到 90%+ 的准确率

### 关键观察
1. **逻辑回归**的决策边界通常是**线性的**
2. **朴素贝叶斯**的决策边界通常是**二次曲线**
3. 在鸢尾花数据集上，**逻辑回归**表现略好（因为数据大致线性可分）
4. 如果数据有**高斯分布特征**，朴素贝叶斯会表现更好

---

## 进阶练习

### 1. 特征选择
修改 `feature_indices` 来尝试不同的特征组合，看对模型的影响。

### 2. 参数调优
```python
# 调整正则化强度
model = LogisticRegression(C=0.1, max_iter=1000)  # C越小，正则化越强

# 尝试不同的核函数
# model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
```

### 3. 特征工程
```python
# 添加多项式特征
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)
```

### 4. 模型对比
尝试其他分类模型（SVM、KNN、决策树）并与逻辑回归比较。

---

## 常见问题

### Q1: 为什么要标准化特征？
**A**: 因为逻辑回归使用梯度下降，特征的尺度会影响收敛速度。标准化使所有特征有相同的重要性。

### Q2: One-vs-Rest 和 Softmax 有什么区别？
**A**: 
- **One-vs-Rest**: 为每个类别训练一个独立的二分类器，可能输出的概率总和 > 1
- **Softmax**: 直接优化多分类，输出的概率总和 = 1（更常用）

### Q3: 朴素贝叶斯为什么叫 "朴素"？
**A**: 因为它假设特征之间相互独立，这个假设通常不成立，所以被称为 "朴素"。

### Q4: 如何选择区分性还是生成性模型？
**A**: 
- 数据充足、追求最高准确率 → **区分性**（逻辑回归、SVM）
- 数据不足、需要生成新数据 → **生成性**（朴素贝叶斯、GDA）

---

## 参考资源

- [Scikit-learn 逻辑回归文档](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [鸢尾花数据集说明](https://archive.ics.uci.edu/ml/datasets/iris)

---

## 许可

所有代码示例均为教育目的。可自由修改和使用。

