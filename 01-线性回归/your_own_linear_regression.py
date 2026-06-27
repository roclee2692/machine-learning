"""
【任务】手写线性回归，用Iris花的数据预测花瓣长度
使用特征：SepalLengthCm, SepalWidthCm （萼片长、宽）
预测目标：PetalLengthCm （花瓣长度）
"""
import os

import numpy as np
import pandas as pd

# ==================== 第1步：数据加载与准备 ====================
# 【TODO】：请你写代码
# 1. 用pandas读取 iris.csv 文件
# 2. 提取前两列特征（X）：SepalLengthCm, SepalWidthCm
# 3. 提取目标列（y）：PetalLengthCm
# 4. 数据归一化（特征 X 缩放到 [0,1]）
# 用相对于脚本位置的路径加载数据集，便于在任意机器上运行
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "数据集", "iris.csv")
data = pd.read_csv(DATA_PATH)  # 你的代码：pd.read_csv(...)
X = data[['SepalLengthCm','SepalWidthCm']].values    # 你的代码：data[['SepalLengthCm', 'SepalWidthCm']].values
y = data['PetalLengthCm'].values     # 你的代码：data['PetalLengthCm'].values

# 数据归一化
# X_normalized = (X - X.min()) / (X.max() - X.min())
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
'''X.min(axis=0)  →  对每一列分别求最小值  → 返回 [最小值1, 最小值2, ...]
X.max(axis=0)  →  对每一列分别求最大值  → 返回 [最大值1, 最大值2, ...]

广播机制会自动把这些向量扩展到所有行，
所以每列都用自己的最小值和最大值来归一化！'''

print("数据加载完成")
print(f"X shape: {X.shape}")  # 应该是 (150, 2)
print(f"y shape: {y.shape}")  # 应该是 (150,)
print(f"前5个样本X:\n{X[:5]}")
print(f"前5个样本y: {y[:5]}")


# ==================== 第2步：初始化参数 ====================
def init_parameters(n_features):
    """
    【TODO】初始化权重和偏置
    参数个数应该与特征数相同
    """
    # w = ...
    # b = ...
    # return w, b
    w = np.zeros(n_features)#初始化为 全0数组，不是数据
    b = 0
    return w, b

# ==================== 第3步：前向传播 ====================
def forward_propagation(X, w, b):
    """
    【TODO】计算预测值
    公式：ŷ = X·w + b
    
    提示：用 np.dot() 进行矩阵乘法
    """
    # y_pred = ...
    # return y_pred
    y_pred = np.dot(X, w) + b
    return y_pred
    

# ==================== 第4步：损失函数 ====================
def compute_loss(y_pred, y_true):
    """
    【TODO】计算均方误差 MSE
    公式：L = (1/m) * Σ(y_pred - y_true)²
    
    提示：
    - m = 样本数量 = len(y_true)
    - 用 np.sum() 求和
    - 误差的平方用 ** 2
    """
    m=len(y_true)
    loss = (1/m)*np.sum(pow(y_pred-y_true,2))
    return loss
    


# ==================== 第5步：计算梯度 ====================
def compute_gradients(X, y_pred, y_true):
    """
    【TODO】计算权重和偏置的梯度
    公式：
      dw = (2/m) * X^T · (y_pred - y_true)
      db = (2/m) * sum(y_pred - y_true)
    
    提示：
    - m = 样本数量
    - 用 np.dot(X.T, error) 计算 dw
    - 用 np.sum(error) 计算 db
    """
    m = len(y_true)
    error = None  # 计算预测误差
    
    # dw = ...
    # db = ...
    # return dw, db
    error=y_pred - y_true
    dw=(2/m)*np.dot(X.T,error)
    db=(2/m)*np.sum(error)
    return dw,db


# ==================== 第6步：参数更新 ====================
def update_parameters(w, b, dw, db, learning_rate):
    """
    【TODO】使用梯度下降更新参数
    公式：
      w_new = w_old - learning_rate * dw
      b_new = b_old - learning_rate * db
    """
    # w = ...
    # b = ...
    # return w, b
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


# ==================== 第7步：训练循环 ====================
def train(X, y, learning_rate=0.01, iterations=100):
    """
    【TODO】完整的训练循环
    
    步骤：
    1. 初始化参数（w, b）
    2. for 循环 iterations 次：
       - 前向传播：y_pred = forward_propagation(X, w, b)
       - 计算损失：loss = compute_loss(y_pred, y)
       - 计算梯度：dw, db = compute_gradients(X, y_pred, y)
       - 更新参数：w, b = update_parameters(w, b, dw, db, learning_rate)
       - 每隔一段时间打印损失值
    
    3. 返回 (w, b, loss_history)
    """
    w, b = init_parameters(X.shape[1])
    loss_history = []
    for i in range(iterations):
        y_pred=forward_propagation(X,w,b)
        loss=compute_loss(y_pred,y)
        loss_history.append(loss)
        dw,db=compute_gradients(X,y_pred,y)
        w,b=update_parameters(w,b,dw,db,learning_rate)
        if i % max(1,iterations//10)==0:
            print(f'迭代{i:3d}')
    return w, b, loss_history

# ==================== 第8步：预测和评估 ====================
def predict(X, w, b):
    """预测函数"""
    return forward_propagation(X, w, b)


def evaluate(y_true, y_pred):
    """计算R²分数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n【开始：手写线性回归使用Iris数据】\n")
    
    # 步骤1：加载数据（你需要实现）
    print("=" * 60)
    print("步骤1：数据加载")
    print("=" * 60)
    # 补充上面的代码
    
    # 步骤2-7：训练模型
    print("\n" + "=" * 60)
    print("步骤2-7：模型训练")
    print("=" * 60)
    w, b, loss_history = train(X_normalized, y, learning_rate=0.01, iterations=200)
    
    # 步骤8：评估
    print("\n" + "=" * 60)
    print("步骤8：模型评估")
    print("=" * 60)
    y_pred = predict(X_normalized, w, b)
    r2 = evaluate(y, y_pred)
    print(f"R² 分数: {r2:.6f}")
    print(f"最终参数 w: {w}")
    print(f"最终参数 b: {b:.6f}")
    
    print("\n✅ 完成！")
