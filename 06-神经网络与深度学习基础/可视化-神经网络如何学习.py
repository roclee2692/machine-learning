"""
================================================================================
可视化-神经网络如何学习.py

🎬 这是本章的"明星脚本"。
   它把神经网络的学习过程做成实时动画，让你看见三件事：
     1. 决策边界 是怎么从一团糟逐渐"长"成 moons 的形状的
     2. 损失曲线 是怎么下降的
     3. 每个隐藏神经元 各自在画一条什么样的"线"，最后怎么组合出复杂边界

🎯 看完之后，请回答以下问题（写在纸上）：
   Q1. 训练初期，4 条隐藏神经元的线大致随机指向哪里？
   Q2. 训练后期，这 4 条线最终各自在画什么？(它们各自识别什么"局部特征"？)
   Q3. 为什么单个神经元都画不出 moons，但 4 个一组合就能？

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 简化版的网络（和 02 文件结构一致，方便对照）
# ==============================================================================
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class TinyMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.5):
        self.lr = lr
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        H = sigmoid(Z1)
        Z2 = H @ self.W2.T + self.b2
        self.cache = dict(X=X, Z1=Z1, H=H, Z2=Z2)
        return Z2

    def backward(self, Y_true):
        c = self.cache
        B = c['X'].shape[0]
        dY = (c['Z2'] - Y_true) / B
        dW2 = dY.T @ c['H']
        db2 = dY.sum(axis=0)
        dH = dY @ self.W2
        dZ1 = dH * c['H'] * (1 - c['H'])
        dW1 = dZ1.T @ c['X']
        db1 = dZ1.sum(axis=0)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2)

    def step(self, X, Y):
        Y_hat = self.forward(X)
        loss = 0.5 * np.mean((Y_hat - Y) ** 2)
        g = self.backward(Y)
        for k in ['W1', 'b1', 'W2', 'b2']:
            setattr(self, k, getattr(self, k) - self.lr * g[k])
        return loss


# ==============================================================================
# 数据 + 网络
# ==============================================================================
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
y_col = y.reshape(-1, 1).astype(float)

HIDDEN = 4   # 4 个隐藏神经元，方便逐个画出每个神经元的"决策线"
model = TinyMLP(input_dim=2, hidden_dim=HIDDEN, output_dim=1, lr=0.8)

# 决策边界采样网格
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]


# ==============================================================================
# 设置画布（3 个子图）
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle('神经网络的学习过程（左：决策边界 | 中：损失曲线 | 右：每个隐藏神经元在画什么）',
             fontsize=13)

# 左：决策边界 + 数据点
ax_db = axes[0]
contour = [None]  # 用 list 包住，方便在闭包中重新赋值

def draw_data():
    ax_db.scatter(X[y == 0, 0], X[y == 0, 1], c='#C0392B',
                  edgecolor='k', s=30, label='类别 0', zorder=3)
    ax_db.scatter(X[y == 1, 0], X[y == 1, 1], c='#16A085',
                  edgecolor='k', s=30, label='类别 1', zorder=3)
    ax_db.set_xlim(x_min, x_max)
    ax_db.set_ylim(y_min, y_max)
    ax_db.set_title('决策边界（实时更新）')
    ax_db.legend(loc='upper right')

draw_data()

# 中：损失曲线
ax_loss = axes[1]
loss_line, = ax_loss.plot([], [], color='#2E86AB', lw=2)
ax_loss.set_xlim(0, 1)
ax_loss.set_ylim(0, 0.3)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('损失曲线')
ax_loss.grid(True, alpha=0.3)

# 右：每个隐藏神经元的"决策线"
# 一个隐藏神经元 i 的"线"由 W1[i] @ x + b1[i] = 0 定义
# （即 sigmoid 输出 0.5 的位置 — 这条线把空间一分为二）
ax_hidden = axes[2]
COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
neuron_lines = [ax_hidden.plot([], [], color=COLORS[i], lw=2.5,
                                label=f'神经元 {i+1}')[0]
                for i in range(HIDDEN)]
ax_hidden.scatter(X[y == 0, 0], X[y == 0, 1], c='lightgray',
                  edgecolor='gray', s=20, zorder=2)
ax_hidden.scatter(X[y == 1, 0], X[y == 1, 1], c='dimgray',
                  edgecolor='black', s=20, zorder=2)
ax_hidden.set_xlim(x_min, x_max)
ax_hidden.set_ylim(y_min, y_max)
ax_hidden.set_title(f'{HIDDEN} 个隐藏神经元各自画的"线"')
ax_hidden.legend(loc='upper right', fontsize=9)


# ==============================================================================
# 动画核心：每帧训练 N 步，然后刷新
# ==============================================================================
losses = []
N_STEPS_PER_FRAME = 10
TOTAL_FRAMES = 200    # 总共 200 帧 = 训练 2000 步

def update(frame):
    # 训练 N 步
    for _ in range(N_STEPS_PER_FRAME):
        loss = model.step(X, y_col)
        losses.append(loss)

    # ----- 左：刷新决策边界 -----
    Z = model.forward(grid).reshape(xx.shape)
    if contour[0] is not None:
        for c in contour[0].collections:
            c.remove()
    contour[0] = ax_db.contourf(xx, yy, Z, levels=[Z.min(), 0.5, Z.max()],
                                 alpha=0.35, colors=['#FF6B6B', '#4ECDC4'],
                                 zorder=1)
    ax_db.set_title(f'决策边界 | epoch {len(losses)} | loss={loss:.4f}')

    # ----- 中：刷新损失曲线 -----
    loss_line.set_data(range(len(losses)), losses)
    ax_loss.set_xlim(0, max(100, len(losses)))
    ax_loss.set_ylim(0, max(0.3, max(losses[:50]) * 1.1) if losses else 0.3)

    # ----- 右：刷新每个隐藏神经元的"线" -----
    # 神经元 i 的线: W1[i,0]*x + W1[i,1]*y + b1[i] = 0
    # 解出 y = -(W1[i,0]*x + b1[i]) / W1[i,1]
    x_pts = np.linspace(x_min, x_max, 50)
    for i in range(HIDDEN):
        w0, w1 = model.W1[i, 0], model.W1[i, 1]
        b = model.b1[i]
        if abs(w1) < 1e-6:    # 竖线，特殊处理
            x_v = -b / (w0 + 1e-9)
            neuron_lines[i].set_data([x_v, x_v], [y_min, y_max])
        else:
            y_pts = -(w0 * x_pts + b) / w1
            neuron_lines[i].set_data(x_pts, y_pts)

    return [loss_line, *neuron_lines]


# ==============================================================================
ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=50, blit=False)
plt.tight_layout()


# 想保存为 gif/mp4 就取消下面注释（需要 pillow 或 ffmpeg）
# ani.save('06_神经网络学习过程.gif', writer='pillow', fps=20)

if __name__ == '__main__':
    print("=" * 60)
    print("🎬 动画启动中... 关闭窗口结束")
    print("=" * 60)
    print("""
观察重点:
  1. 看左图：边界从随机的混乱状态，逐渐"挤"出 moons 的形状
  2. 看中图：损失先快速下降，再逐渐平稳
  3. 看右图（最重要！）：
     - 训练初期，4 条线几乎随机指向各处
     - 训练后期，每条线会找到一个"有意义"的位置：
       * 有的横在两月之间
       * 有的斜着切下半月
       * 有的可能从外围"夹"住
     - 4 条线各自只是直线，但它们的【加权组合】（由 W2 决定）
       拼出了你看到的那个弯曲的最终边界

这就是神经网络的"分而治之"：
  → 每个神经元只负责画一条简单的直线
  → 输出层把它们组合成复杂的非线性边界
  → 这才是 "深度" 的真正威力 (这里只是宽度演示)
""")
    plt.show()
