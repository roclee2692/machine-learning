"""
================================================================================
02-代码实现.py — RNN vs LSTM vs CNN 三方对比实验（Fashion-MNIST 数据集）
================================================================================

注意：如果遇到 OMP 多实例错误，在终端运行前设置：
    set KMP_DUPLICATE_LIB_OK=TRUE
或在代码中已使用 Agg 后端避免此问题。

实验目的：
    在同一个数据集（Fashion-MNIST）上分别训练 RNN、LSTM 和 CNN，
    对比三者的准确率、训练速度和损失曲线，直观感受三种架构的差异。

    重点理解：
    - RNN 有什么问题？（梯度消失 → 学不到长距离依赖）
    - LSTM 如何解决？（门控机制 → 能记住长期信息）
    - CNN 为什么在图像上更强？（空间特征提取 → 纹理/形状）

数据集：
    Fashion-MNIST — 28×28 灰度图，10 类（T恤、裤子、套衫、裙子、外套、
    凉鞋、衬衫、运动鞋、包、短靴），60k 训练 + 10k 测试。

    三种模型怎么看同一张图：
    - RNN:   按行从上往下"读"，但读到下半身时已经忘了上半身长什么样
    - LSTM:  同样按行读，但有选择地记住"领口是V领"、"有两条裤腿"等关键特征
    - CNN:   用卷积核扫描全图，直接提取"领口形状"、"裤腿轮廓"等空间纹理

Fashion-MNIST 数据集（Zalando Research, 2017）：
    28×28 灰度图，10 类衣服/鞋/包，60k 训练 + 10k 测试
    是 MNIST 的"升级版替代品"——格式相同但难度更高，更能区分模型优劣

依赖：
    pip install torch torchvision matplotlib numpy

================================================================================
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免 OMP 多实例冲突
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==============================================================================
# 第 1 部分：数据加载
# ==============================================================================
def get_dataloaders(batch_size=128):
    """
    加载 Fashion-MNIST 数据集，返回训练和测试 DataLoader

    Fashion-MNIST 类别：
        0: T恤/上衣   1: 裤子      2: 套衫      3: 裙子      4: 外套
        5: 凉鞋       6: 衬衫      7: 运动鞋    8: 包        9: 短靴
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                        # 转为 [0, 1] 的张量
        transforms.Normalize((0.2860,), (0.3530,))    # Fashion-MNIST 的均值和标准差
    ])

    train_dataset = datasets.FashionMNIST(
        root='../数据集', train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='../数据集', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 类别名称（用于可视化）
CLASS_NAMES = [
    'T恤/上衣', '裤子', '套衫', '裙子', '外套',
    '凉鞋', '衬衫', '运动鞋', '包', '短靴'
]


# ==============================================================================
# 第 2 部分：CNN 模型
# ==============================================================================
class SimpleCNN(nn.Module):
    """
    一个简单的 CNN，用于 Fashion-MNIST 分类

    架构：
        输入 (1, 28, 28)
        → Conv2d(1→16, 3×3) → ReLU → MaxPool(2×2)   → (16, 14, 14)
        → Conv2d(16→32, 3×3) → ReLU → MaxPool(2×2)   → (32, 7, 7)
        → Flatten                                     → (1568)
        → Linear(1568→128) → ReLU                     → (128)
        → Linear(128→10)                              → (10)

    参数量：~207k
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (1,28,28) → (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (16,28,28) → (16,14,14)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # (16,14,14) → (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # (32,14,14) → (32,7,7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),              # (32,7,7) → (1568)
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==============================================================================
# 第 3 部分：Vanilla RNN 模型
# ==============================================================================
class SimpleRNN(nn.Module):
    """
    一个简单的 Vanilla RNN（Elman RNN），用于 Fashion-MNIST 分类

    核心思路：
        把 28×28 的图像当成长度为 28 的序列，每个时间步输入 28 维（一行像素）
        RNN 按行"读"完整张图，用最后时间步的隐藏状态做分类

        输入 (1, 28, 28)
        → reshape → (batch, 28, 28)     # 28 个时间步，每步 28 维
        → RNN(28→64, 2 layers)          # 2 层 Vanilla RNN，隐藏维度 64
        → 取最后时间步输出 → (64)
        → Linear(64→10)                 → (10)

    Vanilla RNN 的问题：
        h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)
        梯度在时间步之间连乘 tanh 和 W_hh，序列一长梯度就消失或爆炸
        → 早期像素的信息传不到最后，分类效果差

    参数量：~17k（比 LSTM 少很多，因为没有门控参数）
    """

    def __init__(self, input_size=28, hidden_size=64, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            nonlinearity='tanh',   # 默认就是 tanh
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28) → 去掉通道维 → (batch, 28, 28)
        x = x.squeeze(1)

        # RNN 前向传播
        # out: (batch, seq_len, hidden_size) — 所有时间步的输出
        # h_n: (num_layers, batch, hidden_size) — 最后时间步的隐藏状态
        out, _ = self.rnn(x)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch, hidden_size)

        # 全连接分类
        out = self.fc(out)   # (batch, 10)
        return out


# ==============================================================================
# 第 4 部分：LSTM 模型
# ==============================================================================
class SimpleLSTM(nn.Module):
    """
    一个简单的 LSTM，用于 Fashion-MNIST 分类

    核心思路：
        和 RNN 一样把图像当序列处理，但引入了门控机制：
        - 遗忘门：决定丢弃哪些旧信息
        - 输入门：决定写入哪些新信息
        - 输出门：决定输出哪些信息

        这让 LSTM 能选择性地记住长期信息，缓解梯度消失问题

        输入 (1, 28, 28)
        → reshape → (batch, 28, 28)     # 28 个时间步，每步 28 维
        → LSTM(28→64, 2 layers)         # 2 层 LSTM，隐藏维度 64
        → 取最后时间步输出 → (64)
        → Linear(64→10)                 → (10)

    参数量：~65k（比 Vanilla RNN 多，因为有 3 个门的参数）
    """

    def __init__(self, input_size=28, hidden_size=64, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28) → 去掉通道维 → (batch, 28, 28)
        x = x.squeeze(1)

        # LSTM 前向传播
        out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch, hidden_size)

        # 全连接分类
        out = self.fc(out)   # (batch, 10)
        return out


# ==============================================================================
# 第 5 部分：通用训练/评估函数
# ==============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer):
    """训练一个 epoch，返回平均 loss"""
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(train_loader.dataset)


def evaluate(model, test_loader):
    """在测试集上评估，返回准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return correct / total


def train_model(model, train_loader, test_loader, n_epochs=5, lr=0.001):
    """
    完整训练流程，返回训练记录

    返回：
        history: dict，包含 train_losses, test_accs, train_time
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_losses': [], 'test_accs': []}

    start_time = time.time()
    for epoch in range(n_epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        acc = evaluate(model, test_loader)
        history['train_losses'].append(loss)
        history['test_accs'].append(acc)
        print(f"  Epoch {epoch+1:2d}/{n_epochs} | loss = {loss:.4f} | 测试准确率 = {acc:.4f}")
    elapsed = time.time() - start_time
    history['train_time'] = elapsed

    return history


# ==============================================================================
# 第 6 部分：可视化
# ==============================================================================
def plot_comparison(histories, save_path='07_rnn_vs_lstm_vs_cnn_result.png'):
    """
    绘制 RNN vs LSTM vs CNN 三方对比图：
        左图：训练损失曲线
        中图：测试准确率曲线
        右图：训练时间柱状图
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(histories['RNN']['train_losses']) + 1)

    styles = {
        'RNN':  {'color': '#2ECC71', 'marker': '^', 'label': 'Vanilla RNN'},
        'LSTM': {'color': '#E74C3C', 'marker': 's', 'label': 'LSTM'},
        'CNN':  {'color': '#3498DB', 'marker': 'o', 'label': 'CNN'},
    }

    # ---- 左图：损失曲线 ----
    for name, h in histories.items():
        s = styles[name]
        axes[0].plot(epochs, h['train_losses'], color=s['color'],
                     marker=s['marker'], label=s['label'], markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('训练损失')
    axes[0].set_title('训练损失对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- 中图：准确率曲线 ----
    for name, h in histories.items():
        s = styles[name]
        axes[1].plot(epochs, h['test_accs'], color=s['color'],
                     marker=s['marker'], label=s['label'], markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('测试准确率')
    axes[1].set_title('测试准确率对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ---- 右图：训练时间 ----
    names = list(histories.keys())
    times = [histories[n]['train_time'] for n in names]
    colors = [styles[n]['color'] for n in names]
    labels = [styles[n]['label'] for n in names]
    bars = axes[2].bar(labels, times, color=colors, width=0.5)
    axes[2].set_ylabel('训练时间（秒）')
    axes[2].set_title('训练时间对比')
    for bar, t in zip(bars, times):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{t:.1f}s', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\n对比图已保存为 {save_path}")
    plt.close()


def show_sample_predictions(models, test_loader, save_path='07_predictions.png'):
    """
    展示 RNN、LSTM、CNN 各自的预测结果，直观对比
    """
    for m in models.values():
        m.eval()

    # 取一个 batch
    X_batch, y_batch = next(iter(test_loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    with torch.no_grad():
        preds = {name: torch.max(m(X_batch[:16]), 1)[1] for name, m in models.items()}

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for i in range(16):
        ax = axes[i // 8][i % 8]
        img = X_batch[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')

        true_label = CLASS_NAMES[y_batch[i]]
        rnn_label = CLASS_NAMES[preds['RNN'][i]]
        lstm_label = CLASS_NAMES[preds['LSTM'][i]]
        cnn_label = CLASS_NAMES[preds['CNN'][i]]

        ax.set_title(f'{true_label}\nR:{rnn_label}\nL:{lstm_label} C:{cnn_label}',
                      fontsize=6, color='black')
        ax.axis('off')

    plt.suptitle('RNN vs LSTM vs CNN 预测对比', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"预测对比图已保存为 {save_path}")
    plt.close()


# ==============================================================================
# 第 7 部分：主函数 — 三方对比实验
# ==============================================================================
def main():
    print("=" * 60)
    print("RNN vs LSTM vs CNN 三方对比实验 — Fashion-MNIST 数据集")
    print("=" * 60)

    # ----- 数据 -----
    print("\n[1/4] 加载数据...")
    train_loader, test_loader = get_dataloaders(batch_size=128)

    # ----- 构建三个模型 -----
    print("\n[2/4] 构建模型...")
    model_dict = {
        'RNN':  SimpleRNN().to(device),
        'LSTM': SimpleLSTM().to(device),
        'CNN':  SimpleCNN().to(device),
    }

    for name, m in model_dict.items():
        params = sum(p.numel() for p in m.parameters())
        print(f"  {name:<6} 参数量: {params:>10,}")

    # ----- 依次训练 -----
    histories = {}
    for i, (name, model) in enumerate(model_dict.items(), 3):
        print(f"\n[{i}/4] 训练 {name}...")
        history = train_model(model, train_loader, test_loader, n_epochs=5, lr=0.001)
        histories[name] = history
        print(f"  {name} 最终测试准确率: {history['test_accs'][-1]:.4f}")
        print(f"  {name} 训练耗时: {history['train_time']:.1f}s")

    # ----- 对比总结 -----
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    header = f"{'指标':<16}" + "".join(f"{n:>10}" for n in histories.keys())
    print(header)
    print("-" * (16 + 10 * len(histories)))

    param_counts = {n: sum(p.numel() for p in m.parameters()) for n, m in model_dict.items()}
    print(f"{'参数量':<16}" + "".join(f"{param_counts[n]:>10,}" for n in histories.keys()))
    print(f"{'测试准确率':<16}" + "".join(f"{histories[n]['test_accs'][-1]:>10.4f}" for n in histories.keys()))
    print(f"{'训练时间(s)':<16}" + "".join(f"{histories[n]['train_time']:>10.1f}" for n in histories.keys()))
    print("=" * 60)

    # 排序
    ranked = sorted(histories.items(), key=lambda x: x[1]['test_accs'][-1], reverse=True)
    rank_str = ' > '.join(
        '{}({:.2%})'.format(n, h['test_accs'][-1]) for n, h in ranked
    )
    print(f"\n准确率排名: {rank_str}")

    print("\n结论：")
    print(f"  RNN:   Vanilla RNN 有梯度消失问题，长序列信息丢失严重")
    print(f"  LSTM:  门控机制缓解了梯度消失，比 RNN 学到更多特征")
    print(f"  CNN:   卷积核提取空间特征（纹理、形状），最适合图像任务")

    # ----- 可视化 -----
    plot_comparison(histories)
    show_sample_predictions(model_dict, test_loader)


# ==============================================================================
if __name__ == '__main__':
    main()
    print("\n所有实验运行完成！")
