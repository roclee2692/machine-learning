"""
================================================================================
05-答案详解.py — 练习题答案

⚠️ 看之前先尝试自己做！至少 30 分钟。
   答案是用来对照、查漏补缺的，不是用来抄的。
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')   # Windows GBK 控制台兼容 ∂ 等数学符号

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# ============== 02-代码实现.py 的 5 个 TODO 答案（先放这里） ==================
# ==============================================================================
"""
[TODO-1] sigmoid_derivative(a):
    return a * (1 - a)

[TODO-2] forward 中:
    Z1 = X @ self.W1.T + self.b1       # (B, hidden_dim)
    H = sigmoid(Z1)                      # (B, hidden_dim)
    Z2 = H @ self.W2.T + self.b2         # (B, output_dim)

[TODO-3] dZ1 = dH * sigmoid_derivative(H)
        # 注意是 H 不是 Z1，因为 sigmoid_derivative 设计成接收 σ(z) 的值

[TODO-4] dW1 = dZ1.T @ X
        db1 = np.sum(dZ1, axis=0)

[TODO-5] update 中:
    self.W1 -= self.lr * grads['W1']
    self.b1 -= self.lr * grads['b1']
    self.W2 -= self.lr * grads['W2']
    self.b2 -= self.lr * grads['b2']
"""


# ==============================================================================
# 🖊️ 手 1. 完整手算前向 + 反向 + 更新
# ==============================================================================
def answer_hand_1():
    """
    用代码"模拟"手算过程，每一步都打印出来。
    你应该用纸笔得到一样的数值。
    """
    print("=" * 70)
    print("手 1. [2,2,1] 网络的完整一次训练（手算 vs 代码验证）")
    print("=" * 70)

    # 参数
    W1 = np.array([[0.15, 0.20],
                   [0.25, 0.30]])
    b1 = np.array([0.35, 0.35])
    W2 = np.array([[0.40, 0.45]])
    b2 = np.array([0.60])

    x = np.array([0.05, 0.10])
    y = np.array([0.01])
    lr = 0.1

    # ---- 前向 ----
    z1 = W1 @ x + b1
    h = 1 / (1 + np.exp(-z1))
    z2 = W2 @ h + b2
    y_hat = z2
    L = 0.5 * (y_hat - y) ** 2

    print(f"\n【前向】")
    print(f"z1 = W1 @ x + b1 = {z1}")
    print(f"   z1[0] = 0.15*0.05 + 0.20*0.10 + 0.35 = {0.15*0.05 + 0.20*0.10 + 0.35}")
    print(f"   z1[1] = 0.25*0.05 + 0.30*0.10 + 0.35 = {0.25*0.05 + 0.30*0.10 + 0.35}")
    print(f"h  = σ(z1)        = {h}")
    print(f"z2 = W2 @ h + b2  = {z2}")
    print(f"y_hat = z2         = {y_hat}")
    print(f"L = 0.5*(y_hat - y)^2 = {L}")

    # ---- 反向 ----
    dy_hat = y_hat - y                   # ∂L/∂y_hat
    dW2 = dy_hat[:, None] * h[None, :]   # (1,1) * (1,2) = (1,2)
    db2 = dy_hat
    dh = W2.T @ dy_hat                   # (2,1) @ (1,) = (2,)
    dz1 = dh * h * (1 - h)               # σ'(z1) = h*(1-h)
    dW1 = dz1[:, None] * x[None, :]      # (2,1) * (1,2) = (2,2)
    db1 = dz1

    print(f"\n【反向】")
    print(f"∂L/∂y_hat = y_hat - y = {dy_hat}")
    print(f"∂L/∂W2     = {dW2}")
    print(f"∂L/∂b2     = {db2}")
    print(f"∂L/∂h      = W2.T @ dy_hat = {dh}")
    print(f"∂L/∂z1     = dh * h * (1-h) = {dz1}")
    print(f"∂L/∂W1     = \n{dW1}")
    print(f"∂L/∂b1     = {db1}")

    # ---- 更新 ----
    W1_new = W1 - lr * dW1
    W2_new = W2 - lr * dW2
    print(f"\n【更新（lr={lr}）】")
    print(f"新 W1 = \n{W1_new}")
    print(f"新 W2 = {W2_new}")

    print("\n你的手算如果和上面一致，说明前向/反向/更新都搞懂了。")


# ==============================================================================
# 🖊️ 手 2. 去掉激活函数 → 退化成线性模型
# ==============================================================================
HAND_2_DISCUSSION = """
【手 2 答案】

设网络:
    h = W1 @ x + b1                    # h ∈ R^2
    y_hat = W2 @ h + b2                # y_hat ∈ R

代入展开:
    y_hat = W2 @ (W1 @ x + b1) + b2
          = (W2 @ W1) @ x + (W2 @ b1 + b2)
          = w' @ x + b'

其中:
    w' = W2 @ W1   (1×2)
    b' = W2 @ b1 + b2  (标量)

结论:
    无论叠多少层，只要中间没有非线性，整个网络永远等价于
    "一个线性变换"，也就是单层线性模型 ŷ = w'.x + b'。

为什么 XOR 学不会？
    线性模型的决策边界永远是一条直线（高维下是超平面）。
    XOR 的两类样本无法用一条直线分开（参考理论讲解 0 节）。
    所以 "无激活函数的深度网络" = 线性模型 = 学不会 XOR。

→ 这就是激活函数为什么是神经元的灵魂。
"""


# ==============================================================================
# 🖊️ 手 3. 梯度消失
# ==============================================================================
def answer_hand_3():
    print("=" * 70)
    print("手 3. 梯度消失到底有多严重")
    print("=" * 70)

    # 1. 反向传播经过 10 层 sigmoid，会乘 10 次 sigmoid 导数
    n_layers = 10
    sigmoid_max_deriv = 0.25
    factor = sigmoid_max_deriv ** n_layers

    print(f"\n经过 {n_layers} 层 sigmoid，梯度被乘以 {sigmoid_max_deriv} 共 {n_layers} 次")
    print(f"连乘结果 = {sigmoid_max_deriv}^{n_layers} = {factor:.2e}")
    print(f"\n如果最后一层梯度是 1.0，第 1 层收到的梯度大约是: {factor:.2e}")
    print(f"= {factor*1e6:.4f} × 10⁻⁶  (几乎为 0)")

    print(f"""
后果:
  第 1 层收到的梯度 ≈ 10⁻⁷，参数几乎不更新 → 浅层不学习
  → 深度网络的"深"反而变成了诅咒

ReLU 能缓解，因为:
  ReLU 的导数在正区间 = 1，负区间 = 0
  正向激活的神经元，梯度直接 1×1×1×... 传回去，不衰减
  代价：负值会被"杀死"（dead ReLU），但整体效果远好于 sigmoid
""")


# ==============================================================================
# 🖊️ 手 4. 深 vs 宽
# ==============================================================================
HAND_4_DISCUSSION = """
【手 4 答案 — 直觉理解】

单层网络要做"从像素直接到 100 类物体":
  - 它必须一步到位，把所有概念同时学会
  - 神经元数量大致需要 4 × 10 × 20 × 100 = 80,000（指数级）
    （每种像素组合 → 每种物体类别都要单独的"模板"）

4 层网络分层学习:
  - 第 1 层：4 个神经元（4 种边缘）
  - 第 2 层：10 个神经元（基于 4 种边缘的组合）
  - 第 3 层：20 个神经元
  - 第 4 层：100 个神经元
  - 总计: 4 + 10 + 20 + 100 = 134（线性增长）

差距:
  浅而宽 ≈ 80,000 个神经元
  深而窄 ≈ 134 个神经元
  ——————————————————————
  深度比宽度的优势是【指数 vs 线性】

核心原因:
  深度的本质是"函数复合"，复合的表达能力以指数级增长。
  浅层的本质是"函数加和"，表达能力只能线性增长。

  用复合学到的特征是【可重用的】（边缘可以组成无数种纹理），
  用加和学到的特征是【独立的】（每种组合都要单独的神经元）。
"""


# ==============================================================================
# 💻 编 1. 三层 MLP
# ==============================================================================
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class DeeperMLP:
    """[input_dim, h1, h2, output_dim] 两层隐藏，sigmoid 激活"""

    def __init__(self, input_dim, h1, h2, output_dim, lr=0.5):
        self.lr = lr
        self.W1 = np.random.randn(h1, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h2, h1) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(output_dim, h2) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(output_dim)

    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        H1 = sigmoid(Z1)
        Z2 = H1 @ self.W2.T + self.b2
        H2 = sigmoid(Z2)
        Z3 = H2 @ self.W3.T + self.b3
        Y_hat = Z3
        self.cache = dict(X=X, Z1=Z1, H1=H1, Z2=Z2, H2=H2, Z3=Z3)
        return Y_hat

    def backward(self, Y_true):
        c = self.cache
        B = c['X'].shape[0]
        dY = (c['Z3'] - Y_true) / B           # (B, 1)
        # 第 3 层
        dW3 = dY.T @ c['H2']
        db3 = dY.sum(axis=0)
        # 传回第 2 层
        dH2 = dY @ self.W3
        dZ2 = dH2 * c['H2'] * (1 - c['H2'])
        dW2 = dZ2.T @ c['H1']
        db2 = dZ2.sum(axis=0)
        # 传回第 1 层
        dH1 = dZ2 @ self.W2
        dZ1 = dH1 * c['H1'] * (1 - c['H1'])
        dW1 = dZ1.T @ c['X']
        db1 = dZ1.sum(axis=0)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)

    def update(self, g):
        for k in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            setattr(self, k, getattr(self, k) - self.lr * g[k])


def answer_code_1():
    print("=" * 70)
    print("编 1. 三层 MLP [2, 8, 4, 1] 训练 moons")
    print("=" * 70)
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    y = y.reshape(-1, 1).astype(float)
    model = DeeperMLP(2, 8, 4, 1, lr=0.5)
    losses = []
    for ep in range(2000):
        Y_hat = model.forward(X)
        loss = 0.5 * np.mean((Y_hat - y) ** 2)
        model.update(model.backward(y))
        losses.append(loss)
    acc = np.mean((model.forward(X) > 0.5) == y)
    print(f"训练 2000 epochs, 最终 loss = {losses[-1]:.4f}, 准确率 = {acc:.4f}")
    return losses


# ==============================================================================
# 💻 编 2. softmax + cross-entropy
# ==============================================================================
def softmax(Z):
    """数值稳定的 softmax: 减去每行最大值再 exp"""
    Z = Z - Z.max(axis=1, keepdims=True)
    eZ = np.exp(Z)
    return eZ / eZ.sum(axis=1, keepdims=True)


def cross_entropy_loss(Y_hat, Y_onehot):
    """L = -1/B * Σ y_ik * log(y_hat_ik)"""
    eps = 1e-12
    return -np.mean(np.sum(Y_onehot * np.log(Y_hat + eps), axis=1))


class SoftmaxMLP:
    """
    [input_dim, hidden, num_classes] + softmax + cross-entropy

    关键推导（写在这里）:
      Y_hat = softmax(Z_out)
      L = -Σ Y * log(Y_hat)

      可以证明: ∂L/∂Z_out = (Y_hat - Y) / B  ← 极致简洁

      为什么这么简洁？因为 softmax 的 Jacobian 和 cross-entropy 的 1/y_hat
      在数学上恰好抵消了 y_hat。这是工程上 softmax+CE 几乎绑定使用的原因。
    """

    def __init__(self, input_dim, hidden, num_classes, lr=0.1):
        self.lr = lr
        self.W1 = np.random.randn(hidden, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(num_classes, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(num_classes)

    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        H = np.maximum(0, Z1)               # 这里换 ReLU 试试看
        Z2 = H @ self.W2.T + self.b2
        Y_hat = softmax(Z2)
        self.cache = dict(X=X, Z1=Z1, H=H, Z2=Z2, Y_hat=Y_hat)
        return Y_hat

    def backward(self, Y_onehot):
        c = self.cache
        B = c['X'].shape[0]
        # 极致简洁的输出层梯度
        dZ2 = (c['Y_hat'] - Y_onehot) / B
        dW2 = dZ2.T @ c['H']
        db2 = dZ2.sum(axis=0)
        dH = dZ2 @ self.W2
        dZ1 = dH * (c['Z1'] > 0)            # ReLU 导数
        dW1 = dZ1.T @ c['X']
        db1 = dZ1.sum(axis=0)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2)

    def update(self, g):
        for k in ['W1', 'b1', 'W2', 'b2']:
            setattr(self, k, getattr(self, k) - self.lr * g[k])


def answer_code_2():
    print("=" * 70)
    print("编 2. Softmax + Cross-Entropy 在 digits 数据集上训练")
    print("=" * 70)
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target  # 归一化到 [0, 1]
    n_classes = 10
    Y_onehot = np.eye(n_classes)[y]
    X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
        X, Y_onehot, y, test_size=0.2, random_state=42
    )
    model = SoftmaxMLP(64, 32, 10, lr=0.5)
    for ep in range(500):
        Y_hat = model.forward(X_train)
        loss = cross_entropy_loss(Y_hat, Y_train)
        model.update(model.backward(Y_train))
        if (ep + 1) % 100 == 0:
            acc = np.mean(np.argmax(model.forward(X_test), axis=1) == y_test)
            print(f"  Epoch {ep+1:3d} | loss={loss:.4f} | test_acc={acc:.4f}")


# ==============================================================================
# 💻 编 3. Mini-batch SGD 对比
# ==============================================================================
def answer_code_3():
    print("=" * 70)
    print("编 3. Full-batch vs Mini-batch vs SGD 对比")
    print("=" * 70)
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    y = y.reshape(-1, 1).astype(float)

    def train(batch_size, n_epochs=200):
        np.random.seed(42)
        model = DeeperMLP(2, 8, 4, 1, lr=0.5)
        losses = []
        for ep in range(n_epochs):
            idx = np.random.permutation(len(X))
            X_shuf, y_shuf = X[idx], y[idx]
            ep_loss = []
            for i in range(0, len(X), batch_size):
                Xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]
                Y_hat = model.forward(Xb)
                ep_loss.append(0.5 * np.mean((Y_hat - yb) ** 2))
                model.update(model.backward(yb))
            losses.append(np.mean(ep_loss))
        return losses

    losses_full = train(batch_size=400)
    losses_mini = train(batch_size=32)
    losses_sgd = train(batch_size=1)

    plt.figure(figsize=(10, 5))
    plt.plot(losses_full, label='Full-batch (B=400)', alpha=0.8)
    plt.plot(losses_mini, label='Mini-batch (B=32)', alpha=0.8)
    plt.plot(losses_sgd, label='SGD (B=1)', alpha=0.6)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('三种 batch 策略对比')
    plt.legend(); plt.grid(True)
    plt.savefig('06_batch_comparison.png', dpi=120)
    print("观察:")
    print("  Full-batch  : 曲线最平滑，但每个 epoch 只更新 1 次，慢")
    print("  Mini-batch  : 抖动适中，每个 epoch 多次更新，最快收敛 ← 实战首选")
    print("  SGD (B=1)   : 抖动剧烈，但能跳出局部极小，需要小学习率")
    print("\n图已保存为 06_batch_comparison.png")


# ==============================================================================
# 💻 编 4. 真实数据集 — 乳腺癌
# ==============================================================================
def answer_code_4():
    print("=" * 70)
    print("编 4. 乳腺癌分类：神经网络 vs 逻辑回归")
    print("=" * 70)
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 必须标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_col = y_train.reshape(-1, 1).astype(float)

    # 神经网络
    model = DeeperMLP(30, 16, 8, 1, lr=0.05)
    batch_size = 32
    for ep in range(200):
        idx = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            Xb = X_train[idx[i:i+batch_size]]
            yb = y_train_col[idx[i:i+batch_size]]
            model.forward(Xb)
            model.update(model.backward(yb))
    nn_pred = (model.forward(X_test) > 0.5).astype(int).ravel()
    nn_acc = np.mean(nn_pred == y_test)
    nn_f1 = f1_score(y_test, nn_pred)

    # 逻辑回归基线
    lr_clf = LogisticRegression(max_iter=2000)
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    lr_acc = np.mean(lr_pred == y_test)
    lr_f1 = f1_score(y_test, lr_pred)

    print(f"\n神经网络  : acc={nn_acc:.4f}, F1={nn_f1:.4f}")
    print(f"逻辑回归  : acc={lr_acc:.4f}, F1={lr_f1:.4f}")
    print("\n混淆矩阵 (神经网络):")
    print(confusion_matrix(y_test, nn_pred))

    print("""
思考题答案:
  这个数据集只有 569 个样本、30 个特征，且类别相对线性可分。
  这种小、简单、低维的数据上，神经网络的优势体现不出来，
  甚至可能因为参数多反而更容易过拟合。

  神经网络真正的舞台是：
  - 高维 / 非结构化数据（图像、文本、语音）
  - 大数据量（10万 → 千万级样本）
  - 复杂的非线性关系

  在表格类小数据集上，逻辑回归 / 树模型 / XGBoost 通常是更优解。
  这不是神经网络没用，而是"杀鸡焉用牛刀"。
""")


# ==============================================================================
if __name__ == '__main__':
    answer_hand_1()
    print(HAND_2_DISCUSSION)
    answer_hand_3()
    print(HAND_4_DISCUSSION)
    answer_code_1()
    print()
    answer_code_2()
    print()
    answer_code_3()
    print()
    answer_code_4()
