# Machine Learning 机器学习核心算法学习仓库

> A hands-on, tutorial-style repository covering core machine learning algorithms — from linear regression to Transformers — with theory notes, runnable code, exercises, solutions, result figures, and datasets.
>
> 一个实战式、教程风格的机器学习学习仓库，覆盖从线性回归到 Transformer 的核心算法，每章包含理论讲解、可运行代码、练习题、答案详解、结果图表与数据集。

**Language / 语言**: [English](#english) | [中文](#中文)

---

## English

### 📖 Overview

This repository documents a complete learning path through the core algorithms of machine learning and deep learning. Each chapter is **self-contained**: a folder holds its theory notes (`.md`), code implementation (`.py`), exercises, answer keys, and the result figures that the code produces — so you can read, run, and verify in one place.

It is suitable for learners following Andrew Ng's ML course (or similar) who want concrete, executable examples.

### 📂 Repository Structure

| Chapter | Topic | Highlights |
|---|---|---|
| `00-基础知识` | Fundamentals | Feature engineering guide & practice |
| `01-线性回归` | Linear Regression | From-scratch + sklearn, gradient-descent viz, California-housing mini-project |
| `02-逻辑回归` | Logistic Regression | Binary/multiclass, discriminative vs. generative, Iris case studies |
| `03-决策树与随机森林` | Decision Trees & Random Forest | Information gain, Gini, ensembles |
| `04-KNN与SVM` | KNN & SVM | Distance metrics, kernel trick |
| `05-聚类算法` | Clustering | K-Means, hierarchical, DBSCAN |
| `06-神经网络与深度学习基础` | Neural Networks & DL Basics | NumPy MLP + PyTorch counterpart |
| `07-CNN与RNN` | CNN & RNN | CNN vs RNN vs LSTM on FashionMNIST |
| `08-复杂非线性回归` | Complex Nonlinear Regression | LSTM time-series (ETTh1) + ensemble methods |
| `09-Transform` | Transformer | Transformer fundamentals & implementation |
| `数据集` | Datasets | `iris.csv`, `ETTh1.csv`, FashionMNIST, AG News |

Most chapters follow this layout:

```
NN-章节名称/
├── 01-理论讲解.md      # Theory notes
├── 02-代码实现.py      # Code implementation
├── 03-练习题.md        # Exercises
└── 04-答案详解.py      # Solutions
```

### 🗃️ Datasets

| File | Used by | Notes |
|---|---|---|
| `数据集/iris.csv` | Linear/Logistic Regression | Classic Iris dataset |
| `数据集/FashionMNIST/` | CNN/RNN (Ch. 07) | Auto-downloadable via torchvision; included for convenience |
| `数据集/ag_news/` | Transformer / NLP | Text classification dataset |
| `08-复杂非线性回归/LSTM/data/ETTh1.csv` | LSTM (Ch. 08) | Electricity Transformer oil-temperature time series |

### 🚀 Getting Started

```bash
# Clone
git clone https://github.com/roclee2692/machine-learning.git
cd machine-learning

# Core dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Optional (deep learning chapters)
pip install torch torchvision
```

Then open any chapter, read the `.md` theory file, and run the `.py` code, e.g.:

```bash
python 01-线性回归/02-代码实现.py
```

### 🎓 Suggested Learning Flow

1. Read the theory notes (`01-理论讲解.md`).
2. Run and study the code (`02-代码实现.py`).
3. Try the exercises (`03-练习题.md`) on your own.
4. Compare against the solutions (`04-答案详解.py`).

### 📜 License

See [LICENSE](LICENSE) if present. This repository is for educational use.

---

## 中文

### 📖 简介

本仓库记录了一条完整的机器学习与深度学习核心算法学习路线。每一章都是**自包含**的：一个文件夹里同时放着理论讲解（`.md`）、代码实现（`.py`）、练习题、答案详解，以及代码产出的结果图表——可以在一处完成阅读、运行与验证。

适合跟随吴恩达机器学习课程（或同类课程）学习、希望获得可直接运行示例的同学。

### 📂 仓库结构

| 章节 | 主题 | 重点内容 |
|---|---|---|
| `00-基础知识` | 基础知识 | 特征工程完整指南与实战 |
| `01-线性回归` | 线性回归 | 从零实现 + sklearn、梯度下降可视化、加州房价小项目 |
| `02-逻辑回归` | 逻辑回归 | 二分类/多分类、判别式 vs 生成式、鸢尾花案例 |
| `03-决策树与随机森林` | 决策树与随机森林 | 信息增益、基尼指数、集成学习 |
| `04-KNN与SVM` | KNN 与 SVM | 距离度量、核技巧 |
| `05-聚类算法` | 聚类算法 | K-Means、层次聚类、DBSCAN |
| `06-神经网络与深度学习基础` | 神经网络基础 | NumPy 手写 MLP + PyTorch 对照实现 |
| `07-CNN与RNN` | CNN 与 RNN | 在 FashionMNIST 上对比 CNN/RNN/LSTM |
| `08-复杂非线性回归` | 复杂非线性回归 | LSTM 时间序列（ETTh1）+ 集成算法 |
| `09-Transform` | Transformer | Transformer 原理与实现 |
| `数据集` | 数据集 | `iris.csv`、`ETTh1.csv`、FashionMNIST、AG News |

多数章节采用如下结构：

```
NN-章节名称/
├── 01-理论讲解.md      # 详细理论知识
├── 02-代码实现.py      # 完整代码示例
├── 03-练习题.md        # 配套练习题
└── 04-答案详解.py      # 练习题答案
```

### 🗃️ 数据集说明

| 文件 | 使用章节 | 说明 |
|---|---|---|
| `数据集/iris.csv` | 线性/逻辑回归 | 经典鸢尾花数据集 |
| `数据集/FashionMNIST/` | CNN/RNN（第 07 章） | 可由 torchvision 自动下载，为方便已一并收录 |
| `数据集/ag_news/` | Transformer / NLP | 文本分类数据集 |
| `08-复杂非线性回归/LSTM/data/ETTh1.csv` | LSTM（第 08 章） | 电力变压器油温时间序列数据 |

### 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/roclee2692/machine-learning.git
cd machine-learning

# 核心依赖
pip install numpy pandas matplotlib seaborn scikit-learn

# 可选（深度学习章节）
pip install torch torchvision
```

然后进入任意章节，先看 `.md` 理论文件，再运行 `.py` 代码，例如：

```bash
python 01-线性回归/02-代码实现.py
```

### 🎓 建议学习流程

1. 阅读理论讲解（`01-理论讲解.md`）。
2. 运行并研读代码（`02-代码实现.py`）。
3. 独立完成练习题（`03-练习题.md`）。
4. 对照答案查漏补缺（`04-答案详解.py`）。

### 💡 学习建议

- **理论先行**：先理解数学原理，再看代码。
- **动手实践**：边看边运行，独立完成练习远比只看更重要。
- **对比答案**：找出自己与参考实现的差距。

### 📜 许可

如有 [LICENSE](LICENSE) 文件请以其为准。本仓库仅用于学习用途。

---

**记住：机器学习是实践的学问，动手做远比看教程重要！🚀**
**Remember: machine learning is learned by doing — running code beats reading about it. 🚀**
