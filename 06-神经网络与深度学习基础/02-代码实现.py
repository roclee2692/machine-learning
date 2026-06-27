"""
================================================================================
02-代码实现.py — 用 NumPy 从零手写一个神经网络
================================================================================

⚠️ 重要规则 ⚠️
1. 这个文件里有 5 处 TODO，标记为 [TODO-1] 到 [TODO-5]
2. **不准看 05-答案详解.py！** 答案就摆在那里，但看了你就废了
3. 推荐做法：
   - 先把 TODO 全部写完
   - 直接运行整个文件
   - 如果训练损失能稳步下降，恭喜你做对了
   - 如果损失不变 / NaN / 上升，回去 debug
4. 每个 TODO 上面都有提示和形状要求，但**不会给你代码**

为什么不用 PyTorch？
   PyTorch 的 `loss.backward()` 一行就把反向传播做完了。
   但你不知道这一行后面发生了什么。
   写这个 NumPy 版，就是要让你看清楚每一个矩阵乘法、每一次梯度计算。
   写完之后再去看 PyTorch，你会瞬间理解它在替你做什么。

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 让随机数可复现，方便对照
np.random.seed(42)

# 中文字体（Windows 用黑体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 第 1 部分：激活函数
# ==============================================================================
def sigmoid(z):
    """Sigmoid 激活函数：σ(z) = 1 / (1 + e^-z)"""
    # 防止 e^z 溢出：把 z 限制在 [-500, 500]
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """
    [TODO-1] 实现 sigmoid 的导数

    数学公式: σ'(z) = σ(z) * (1 - σ(z))

    注意：这个函数的输入是已经过 sigmoid 的值 a (即 a = σ(z))
    所以你不需要再算一次 sigmoid，直接用 a 就好
    返回值的形状应该和输入 a 相同
    """
    return a * (1 - a)


def relu(z):
    """ReLU 激活函数"""
    return np.maximum(0, z)


def relu_derivative(z):
    """ReLU 的导数：z > 0 时为 1，否则为 0"""
    return (z > 0).astype(np.float64)


# ==============================================================================
# 第 2 部分：损失函数
# ==============================================================================
def mse_loss(y_pred, y_true):
    """均方误差损失（用于回归 / 这里也用于二分类的演示）

    L = (1 / 2B) * Σ (y_pred - y_true)^2
    """
    return 0.5 * np.mean((y_pred - y_true) ** 2)


# ==============================================================================
# 第 3 部分：神经网络主体（[输入维度, 隐藏神经元数, 输出维度]）
# ==============================================================================
class TinyMLP:
    """
    一个最简单的两层 MLP（一个隐藏层）

    架构: input_dim → hidden_dim (sigmoid) → output_dim (无激活)

    形状约定（行优先 = 每行一个样本）:
        X       : (B, input_dim)
        W1      : (hidden_dim, input_dim)
        b1      : (hidden_dim,)
        Z1 = X @ W1.T + b1       (B, hidden_dim)
        H  = σ(Z1)                (B, hidden_dim)
        W2      : (output_dim, hidden_dim)
        b2      : (output_dim,)
        Y_hat = H @ W2.T + b2    (B, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr

        # 用 He 初始化（适合 ReLU）/ 这里 sigmoid 也凑合用
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        # 缓存中间变量（反向传播会用到）
        self.cache = {}

    # --------------------------------------------------------------------------
    def forward(self, X):
        """
        前向传播

        [TODO-2] 实现下面 4 行计算（按理论讲解 4.x 节的公式）：
            Z1 = ?
            H  = ?    (用 sigmoid)
            Z2 = ?    (输出层不加激活，直接线性输出)
            Y_hat = Z2

        提示：
            X    : (B, input_dim)
            W1   : (hidden_dim, input_dim)   → 用的时候要 W1.T
            b1   : (hidden_dim,)              → numpy 自动广播
            W2   : (output_dim, hidden_dim)  → 用的时候要 W2.T
            b2   : (output_dim,)
        """
        # ========== 你的代码开始 ==========
        Z1 = X @ self.W1.T + self.b1   # (B, hidden_dim)
        H = sigmoid(Z1)                 # (B, hidden_dim)
        Z2 = H @ self.W2.T + self.b2   # (B, output_dim)
        # ========== 你的代码结束 ==========

        Y_hat = Z2

        # 缓存中间变量给 backward 用
        self.cache['X'] = X
        self.cache['Z1'] = Z1
        self.cache['H'] = H
        self.cache['Z2'] = Z2

        return Y_hat

    # --------------------------------------------------------------------------
    def backward(self, Y_true):
        """
        反向传播：算出每个参数的梯度

        参考理论讲解 6.4 节的批量形式公式
        """
        X = self.cache['X']
        Z1 = self.cache['Z1']
        H = self.cache['H']
        Y_hat = self.cache['Z2']

        B = X.shape[0]

        # ----------------------------------------------------------------------
        # Step 1: 输出层梯度
        # ----------------------------------------------------------------------
        # ∂L/∂Y_hat = (1/B) * (Y_hat - Y_true), shape: (B, output_dim)
        dY_hat = (Y_hat - Y_true) / B

        # ----------------------------------------------------------------------
        # Step 2: 输出层参数梯度
        # ----------------------------------------------------------------------
        # ∂L/∂W2 = dY_hat.T @ H, shape: (output_dim, hidden_dim) ← 应该和 W2 同形
        # ∂L/∂b2 = dY_hat 沿 batch 维求和, shape: (output_dim,)
        dW2 = dY_hat.T @ H
        db2 = np.sum(dY_hat, axis=0)

        # ----------------------------------------------------------------------
        # Step 3: 把梯度传回隐藏层
        # ----------------------------------------------------------------------
        # ∂L/∂H = dY_hat @ W2, shape: (B, hidden_dim)
        dH = dY_hat @ self.W2

        # ----------------------------------------------------------------------
        # [TODO-3] 穿过 sigmoid 激活函数
        # ----------------------------------------------------------------------
        # ∂L/∂Z1 = dH ⊙ σ'(Z1), 其中 ⊙ 是逐元素相乘
        # 注意：sigmoid_derivative 的输入是 H（已经过 sigmoid 的值），不是 Z1
        #
        # 你应该写出形如:  dZ1 = dH * sigmoid_derivative(?)
        # ========== 你的代码开始 ==========
        dZ1 = dH * sigmoid_derivative(H)  # (B, hidden_dim)
        # ========== 你的代码结束 ==========

        # ----------------------------------------------------------------------
        # [TODO-4] 隐藏层参数梯度
        # ----------------------------------------------------------------------
        # ∂L/∂W1 = dZ1.T @ X, shape 应该是 (hidden_dim, input_dim) ← 和 W1 同形
        # ∂L/∂b1 = dZ1 沿 batch 维求和, shape: (hidden_dim,)
        # ========== 你的代码开始 ==========
        dW1 = dZ1.T @ X                   # (hidden_dim, input_dim)
        db1 = np.sum(dZ1, axis=0)          # (hidden_dim,)
        # ========== 你的代码结束 ==========

        # 把所有梯度打包返回
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return grads

    # --------------------------------------------------------------------------
    def update(self, grads):
        """
        [TODO-5] 用梯度下降更新参数

        公式: W ← W - lr * dW
        需要更新 self.W1, self.b1, self.W2, self.b2
        """
        # ========== 你的代码开始 ==========
        self.W1 -= self.lr * grads['W1']
        self.b1 -= self.lr * grads['b1']
        self.W2 -= self.lr * grads['W2']
        self.b2 -= self.lr * grads['b2']
        # ========== 你的代码结束 ==========

    # --------------------------------------------------------------------------
    def train_step(self, X, Y):
        """一次训练 = 前向 + 反向 + 更新"""
        Y_hat = self.forward(X)
        loss = mse_loss(Y_hat, Y)
        grads = self.backward(Y)
        self.update(grads)
        return loss


# ==============================================================================
# 第 4 部分：在 moons 数据集上训练（典型的非线性二分类）
# ==============================================================================
def main():
    # ----- 数据 -----
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    y = y.reshape(-1, 1).astype(np.float64)  # (500, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----- 模型 -----
    model = TinyMLP(input_dim=2, hidden_dim=8, output_dim=1, lr=0.5)

    # ----- 训练循环 -----
    n_epochs = 2000
    losses = []
    for epoch in range(n_epochs):
        loss = model.train_step(X_train, y_train)
        losses.append(loss)
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:4d} | loss = {loss:.4f}")

    # ----- 评估 -----
    y_pred_train = (model.forward(X_train) > 0.5).astype(int)
    y_pred_test = (model.forward(X_test) > 0.5).astype(int)
    acc_train = np.mean(y_pred_train == y_train)
    acc_test = np.mean(y_pred_test == y_test)
    print(f"\n训练集准确率: {acc_train:.4f}")
    print(f"测试集准确率: {acc_test:.4f}")

    # ----- 可视化 -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：损失曲线
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练损失曲线')
    axes[0].grid(True)

    # 右：决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3,
                     colors=['#FF6B6B', '#4ECDC4'])
    axes[1].scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1],
                    c='#C0392B', edgecolor='k', label='类别 0')
    axes[1].scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1],
                    c='#16A085', edgecolor='k', label='类别 1')
    axes[1].set_title('决策边界（你写的网络学出来的）')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('06_numpy_mlp_result.png', dpi=120)
    print("\n图已保存为 06_numpy_mlp_result.png")
    plt.show()


# ==============================================================================
# 自检函数：跑之前先 sanity check 你的 forward/backward
# ==============================================================================
def gradient_check():
    """
    数值梯度检验：验证你的反向传播是不是真的对
    （如果你 TODO 写错了，损失也可能在下降，但梯度方向不对，这个测试能抓出来）

    原理：梯度的定义是 (L(W + ε) - L(W - ε)) / (2ε)
    我们对每个参数算这个数值梯度，和你 backward 算出的梯度对比
    """
    np.random.seed(0)
    model = TinyMLP(input_dim=2, hidden_dim=3, output_dim=1)
    X = np.random.randn(5, 2)
    Y = np.random.randn(5, 1)

    # 用 backward 算出的梯度
    model.forward(X)
    grads = model.backward(Y)

    # 数值梯度（这里只检验 W1）
    eps = 1e-6
    num_grad_W1 = np.zeros_like(model.W1)
    for i in range(model.W1.shape[0]):
        for j in range(model.W1.shape[1]):
            model.W1[i, j] += eps
            loss_plus = mse_loss(model.forward(X), Y)
            model.W1[i, j] -= 2 * eps
            loss_minus = mse_loss(model.forward(X), Y)
            model.W1[i, j] += eps  # 还原
            num_grad_W1[i, j] = (loss_plus - loss_minus) / (2 * eps)

    diff = np.abs(grads['W1'] - num_grad_W1).max()
    print(f"\n[梯度检验] W1 的解析梯度 vs 数值梯度，最大误差 = {diff:.2e}")
    if diff < 1e-5:
        print("✅ 通过！你的 backward 实现是正确的。")
    else:
        print("❌ 失败。你的 backward 实现有问题，去检查 TODO-3 / TODO-4。")


# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("先做梯度检验（验证 TODO 写对了没）")
    print("=" * 60)
    try:
        gradient_check()
    except Exception as e:
        print(f"❌ TODO 还没写，或者写错了，跑不通。错误：{e}")
        print("先把 TODO-1 ~ TODO-5 写完再来。")
        raise SystemExit

    print()
    print("=" * 60)
    print("梯度检验通过，开始正式训练")
    print("=" * 60)
    main()
