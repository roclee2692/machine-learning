"""
================================================================================
03-PyTorch对照实现.py — 同样的网络，用 PyTorch 重写
================================================================================

⚠️ 必须先把 02-代码实现.py 跑通，再来看这个文件！

目的：让你**亲眼看见** PyTorch 替你做了什么。
同样的事情：
    NumPy 版要写 ~50 行（forward + backward + update）
    PyTorch 版只要 ~5 行
但 PyTorch 没有"魔法"，它做的就是你在 02 文件里写的那些事。
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 第 1 部分：用 nn.Module 定义网络
# ==============================================================================
class TinyMLP(nn.Module):
    """
    和 02-代码实现.py 里的 TinyMLP 完全一样的结构

    对照关系：
        NumPy 版 self.W1 / self.b1   ←→   PyTorch 的 nn.Linear (自动管理)
        NumPy 版 forward()            ←→   forward() 方法
        NumPy 版 backward()           ←→   loss.backward()  ← 一行解决！
        NumPy 版 update()             ←→   optimizer.step() ← 一行解决！
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # 等价于 W1, b1
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 等价于 W2, b2
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # 这就是你在 NumPy 版里手写的前向传播
        # 但 PyTorch 在背后还构建了"计算图"，记录每一步操作
        # 调用 loss.backward() 时它才能反向遍历这张图
        h = self.activation(self.fc1(x))   # = σ(X @ W1.T + b1)
        y_hat = self.fc2(h)                 # = H @ W2.T + b2
        return y_hat


# ==============================================================================
# 第 2 部分：训练
# ==============================================================================
def main():
    # ----- 数据 -----
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 转成 PyTorch tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # ----- 模型 / 损失 / 优化器 -----
    model = TinyMLP(input_dim=2, hidden_dim=8, output_dim=1)
    loss_fn = nn.MSELoss()                                    # = 你写的 mse_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)   # = 你写的 update()

    # ----- 训练循环 -----
    n_epochs = 2000
    losses = []
    for epoch in range(n_epochs):

        # ↓↓↓ 这 5 行就是 02 文件里你手写的全部 ↓↓↓
        y_pred = model(X_train_t)              # forward
        loss = loss_fn(y_pred, y_train_t)      # 损失
        optimizer.zero_grad()                  # 清空上一轮的梯度（如果不清，会累加）
        loss.backward()                        # 反向传播，自动算所有参数的梯度
        optimizer.step()                       # W ← W - lr * dW
        # ↑↑↑ 5 行 = NumPy 版 50+ 行 ↑↑↑

        losses.append(loss.item())
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:4d} | loss = {loss.item():.4f}")

    # ----- 评估 -----
    with torch.no_grad():  # 评估时关掉自动求导（省内存）
        train_pred = (model(X_train_t) > 0.5).int()
        test_pred = (model(X_test_t) > 0.5).int()
        acc_train = (train_pred == y_train_t.int()).float().mean().item()
        acc_test = (test_pred == y_test_t.int()).float().mean().item()
    print(f"\n训练集准确率: {acc_train:.4f}")
    print(f"测试集准确率: {acc_test:.4f}")

    # ----- 可视化 -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('PyTorch 版本训练曲线')
    axes[0].grid(True)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3,
                     colors=['#FF6B6B', '#4ECDC4'])
    axes[1].scatter(X[y == 0, 0], X[y == 0, 1],
                    c='#C0392B', edgecolor='k', label='类别 0')
    axes[1].scatter(X[y == 1, 0], X[y == 1, 1],
                    c='#16A085', edgecolor='k', label='类别 1')
    axes[1].set_title('PyTorch 版决策边界')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('06_pytorch_mlp_result.png', dpi=120)
    print("图已保存为 06_pytorch_mlp_result.png")
    plt.show()


# ==============================================================================
# 第 3 部分：彩蛋 — 看一眼"计算图"
# ==============================================================================
def peek_at_autograd():
    """
    PyTorch 的 autograd 就是个"自动 backward"。它怎么做到的？
    答：每次 forward 它都在背后偷偷构建一张 DAG（有向无环图），
       记录每个 tensor 是怎么算出来的、对哪些 tensor 求了梯度。
       当你调 .backward() 时，它沿着图反向遍历，自动应用链式法则。

    这里我们手动看一眼这张图。
    """
    print("\n" + "=" * 60)
    print("【彩蛋】窥探 PyTorch 的计算图")
    print("=" * 60)

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    w = torch.tensor([0.5, 0.3], requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)

    z = (x * w).sum() + b       # z = x1*w1 + x2*w2 + b
    y = torch.sigmoid(z)         # y = σ(z)
    L = (y - 0.0) ** 2           # 假设真实值是 0，损失 = (y - 0)^2

    print(f"x = {x}")
    print(f"前向计算: z = {z.item():.4f}, y = {y.item():.4f}, L = {L.item():.4f}")

    # 这里 PyTorch 自动构建了图：
    #   L ← y ← z ← (w, x, b)
    # 调 backward 后，每个 requires_grad=True 的 tensor 都会拿到自己的梯度
    L.backward()

    print(f"\n反向传播之后:")
    print(f"  ∂L/∂w = {w.grad}")
    print(f"  ∂L/∂x = {x.grad}")
    print(f"  ∂L/∂b = {b.grad.item():.4f}")
    print("\n这些梯度就是用链式法则算出来的，和你在 02 文件里手写的逻辑一模一样。")
    print("PyTorch 的全部功劳就是把这件事自动化了。")


# ==============================================================================
if __name__ == '__main__':
    main()
    peek_at_autograd()
