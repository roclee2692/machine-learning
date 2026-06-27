"""
================================================================================
02-代码实现.py — 从零实现 Transformer（Encoder-only）做文本分类
================================================================================

结合论文 "Attention Is All You Need" (Vaswani et al., NeurIPS 2017)，
手写 Transformer Encoder 的核心组件，不直接用 nn.TransformerEncoder。

实验任务：AG_NEWS 新闻分类（4 类：世界/体育/商业/科技）

数据来源：AG_NEWS CSV 文件（通过 torchtext 下载，或手动放置）
         如果 torchtext 不可用，会自动从网上下载 CSV

依赖：
    pip install torch matplotlib numpy

================================================================================
"""

import os
import time
import math
import re
import csv
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数
MAX_LEN = 128       # 序列最大长度
VOCAB_SIZE = 20000  # 词表大小
D_MODEL = 128       # 模型维度
N_HEADS = 4         # 注意力头数
D_FF = 256          # FFN 内层维度
N_LAYERS = 2        # Encoder 层数
DROPOUT = 0.1
BATCH_SIZE = 64
N_EPOCHS = 5
LR = 0.001

# AG_NEWS 类别名
CLASS_NAMES = ['世界', '体育', '商业', '科技']


# ==============================================================================
# 第 1 部分：数据处理（不依赖 torchtext）
# ==============================================================================
# AG_NEWS 数据集 URL（torchtext 格式的 CSV）
_AG_NEWS_URLS = {
    'train': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
    'test': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv',
}

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '数据集', 'ag_news')


def _download_ag_news():
    """下载 AG_NEWS CSV 文件到本地"""
    import urllib.request
    os.makedirs(_DATA_DIR, exist_ok=True)

    for split, url in _AG_NEWS_URLS.items():
        filepath = os.path.join(_DATA_DIR, f'{split}.csv')
        if not os.path.exists(filepath):
            print(f"  下载 {split} 数据...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"  {split} 数据已存在，跳过下载")


def _read_ag_news_csv(split):
    """
    读取 AG_NEWS CSV 文件
    格式：label(1-4), title, description
    返回 [(label, text), ...]
    """
    filepath = os.path.join(_DATA_DIR, f'{split}.csv')
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0]) - 1  # 转为 0-3
            text = row[1] + ' ' + row[2]  # title + description
            data.append((label, text))
    return data


def simple_tokenizer(text):
    """
    简单的英文分词器：小写 + 按空格/标点分割
    不依赖 spaCy / torchtext 的 tokenizer
    """
    text = text.lower()
    # 用正则分割：保留字母和数字，其他字符作为分隔符
    tokens = re.findall(r'[a-z0-9]+', text)
    return tokens


class Vocab:
    """
    简单的词表类，替代 torchtext.vocab
    """

    def __init__(self, counter, max_size=20000):
        # 特殊 token
        self.itos = ['<pad>', '<unk>']
        self.stoi = {'<pad>': 0, '<unk>': 1}

        # 按频率排序，取最常见的词
        for word, _ in counter.most_common(max_size - 2):
            self.stoi[word] = len(self.itos)
            self.itos.append(word)

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens):
        """把 token 列表转为 ID 列表"""
        return [self.stoi.get(t, self.stoi['<unk>']) for t in tokens]

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])


class AGNewsDataset(Dataset):
    """AG_NEWS PyTorch Dataset"""

    def __init__(self, data, vocab, tokenizer, max_len):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        tokens = self.tokenizer(text)[:self.max_len]
        token_ids = self.vocab(tokens)
        return torch.tensor(token_ids, dtype=torch.long), label


def collate_fn(batch):
    """自定义 collate：处理变长序列，填充到同一长度"""
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, labels


def get_dataloaders():
    """
    加载 AG_NEWS 数据集，构建词表，返回 DataLoader

    AG_NEWS:
        - 4 类新闻：World(0), Sports(1), Business(2), Sci/Tech(3)
        - 训练集 120,000 条，测试集 7,600 条
    """
    # 下载数据
    print("准备 AG_NEWS 数据集...")
    _download_ag_news()

    # 读取数据
    train_data = _read_ag_news_csv('train')
    test_data = _read_ag_news_csv('test')
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")

    # 构建词表
    print("构建词表...")
    counter = Counter()
    for _, text in train_data:
        counter.update(simple_tokenizer(text))
    vocab = Vocab(counter, max_size=VOCAB_SIZE)
    print(f"  词表大小: {len(vocab)}")

    # 创建 Dataset 和 DataLoader
    train_dataset = AGNewsDataset(train_data, vocab, simple_tokenizer, MAX_LEN)
    test_dataset = AGNewsDataset(test_data, vocab, simple_tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, test_loader, vocab


# ==============================================================================
# 第 2 部分：Transformer 核心组件（从零实现）
# ==============================================================================
class ScaledDotProductAttention(nn.Module):
    """
    [论文 3.2.1 节] Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

    直觉：用 Q（问题）和 K（标题）算相关性，再用相关性对 V（内容）加权求和
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, n_heads, seq_len, d_k)
        K: (batch, n_heads, seq_len, d_k)
        V: (batch, n_heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)

        返回: (batch, n_heads, seq_len, d_v), attention_weights
        """
        d_k = Q.size(-1)

        # 步骤 1: 计算注意力分数 Q·K^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: (batch, n_heads, seq_len, seq_len)

        # 步骤 2: 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 步骤 3: Softmax 归一化
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 步骤 4: 加权求和
        output = torch.matmul(attn_weights, V)
        # output: (batch, n_heads, seq_len, d_v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    [论文 3.2.2 节] Multi-Head Attention

    把 Q, K, V 分成 n_heads 份，每份独立做 Scaled Dot-Product Attention，
    最后拼接起来做一次线性变换。

    多头的好处：不同头可以学不同的注意力模式
      - 某些头学语法关系（主语 ↔ 谓语）
      - 某些头学语义关系（同义词 ↔ 同义词）
      - 某些头学位置关系（相邻词）
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性变换矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        batch_size = Q.size(0)

        # 1. 线性变换
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. 拆分成多头: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 做注意力计算
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch, n_heads, seq_len, d_k)

        # 4. 拼接多头: (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 5. 最终线性变换
        output = self.W_O(attn_output)

        return output, attn_weights


class PositionalEncoding(nn.Module):
    """
    [论文 3.5 节] Positional Encoding

    因为 Transformer 没有 RNN 的循环结构，也没有 CNN 的卷积结构，
    它对序列顺序完全"无感"。所以需要额外注入位置信息。

    使用正弦/余弦函数：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    关键性质：PE(pos+k) 可以表示为 PE(pos) 的线性变换
    → 模型可以学到相对位置关系
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    [论文 3.1 节] 单层 Transformer Encoder

    结构：
        输入 x
         ├──→ Multi-Head Self-Attention ──→ Dropout
         │         │
         │    Residual: x + attn_output
         │         │
         │    Layer Normalization
         │         │
         ├──→ Feed-Forward Network ──→ Dropout
         │         │
         │    Residual: prev + ffn_output
         │         │
         │    Layer Normalization
         │         │
         └──→ 输出

    Self-Attention: 每个位置关注序列中所有其他位置
    FFN: 每个位置独立做非线性变换（升维 → ReLU → 降维）
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # 子层 1: Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 子层 2: Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model), attention_weights
        """
        # ---- 子层 1: Self-Attention ----
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))  # Residual + LayerNorm

        # ---- 子层 2: Feed-Forward ----
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))    # Residual + LayerNorm

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    [论文 3.1 节] Transformer Encoder = N 层 EncoderLayer 堆叠

    论文中 N=6，这里默认 N=2（简化版，训练更快）
    """

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        """x: (batch, seq_len, d_model)"""
        attention_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights_list.append(attn_weights)
        return x, attention_weights_list


# ==============================================================================
# 第 3 部分：完整的分类模型
# ==============================================================================
class TransformerClassifier(nn.Module):
    """
    Encoder-only Transformer 文本分类模型

    流程：
        输入 token IDs
         → 词嵌入 (Embedding) + 位置编码 (Positional Encoding)
         → Transformer Encoder (N 层)
         → Mean Pooling（对所有位置取平均）
         → 线性分类头
         → 4 类输出

    这和 BERT 的思路类似：用 Encoder 理解文本，然后做分类。
    区别是 BERT 用了 12/24 层 + 预训练，我们这里是从零训练的小模型。
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 n_classes, max_len, dropout=0.1):
        super().__init__()

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer Encoder
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # 嵌入缩放（论文 3.4 节提到乘以 √d_model）
        self.d_model = d_model

    def create_padding_mask(self, seq):
        """
        创建 padding mask：padding 位置为 0，非 padding 位置为 1
        seq: (batch, seq_len)
        返回: (batch, 1, 1, seq_len)
        """
        return (seq != 0).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len) — token ID 序列
        返回: logits (batch, n_classes), attention_weights
        """
        # 1. Padding mask
        mask = self.create_padding_mask(input_ids)

        # 2. 词嵌入 + 缩放 + 位置编码
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        # x: (batch, seq_len, d_model)

        # 3. Transformer Encoder
        encoder_output, attn_weights_list = self.encoder(x, mask)
        # encoder_output: (batch, seq_len, d_model)

        # 4. Mean Pooling（只对非 padding 位置取平均）
        padding_mask = (input_ids != 0).unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (encoder_output * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)
        # pooled: (batch, d_model)

        # 5. 分类
        logits = self.classifier(pooled)
        # logits: (batch, n_classes)

        return logits, attn_weights_list


# ==============================================================================
# 第 4 部分：训练和评估
# ==============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * texts.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 200 == 0:
            print(f"    batch {batch_idx+1} | loss = {loss.item():.4f}")

    return running_loss / total, correct / total


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            logits, _ = model(texts)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {'train_losses': [], 'train_accs': [], 'test_accs': []}

    start_time = time.time()
    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        test_acc = evaluate(model, test_loader)

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_accs'].append(test_acc)

        print(f"  训练 loss = {train_loss:.4f} | 训练准确率 = {train_acc:.4f} | 测试准确率 = {test_acc:.4f}")

    history['train_time'] = time.time() - start_time
    return history


# ==============================================================================
# 第 5 部分：可视化
# ==============================================================================
def plot_results(history, model, test_loader, vocab,
                 save_path='09_transform_result.png'):
    """绘制训练曲线 + 注意力可视化"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train_losses']) + 1)

    # ---- 左图：训练损失 ----
    axes[0].plot(epochs, history['train_losses'], 'b-o', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('训练损失')
    axes[0].set_title('训练损失曲线')
    axes[0].grid(True, alpha=0.3)

    # ---- 中图：准确率对比 ----
    axes[1].plot(epochs, history['train_accs'], 'b-o', label='训练准确率', markersize=4)
    axes[1].plot(epochs, history['test_accs'], 'r-s', label='测试准确率', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('训练 vs 测试准确率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ---- 右图：最后一个 epoch 的注意力热力图 ----
    model.eval()
    # 取一个测试样本
    sample_text = "Apple announced a new iPhone with advanced AI features"
    tokens = simple_tokenizer(sample_text)[:30]
    token_ids = vocab(tokens)
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        _, attn_weights_list = model(token_tensor)

    # 取第一层、第一个头的注意力权重
    attn = attn_weights_list[0][0, 0].cpu().numpy()  # (seq_len, seq_len)
    n_tokens = min(len(tokens), 20)

    im = axes[2].imshow(attn[:n_tokens, :n_tokens], cmap='Blues', aspect='auto')
    axes[2].set_xticks(range(n_tokens))
    axes[2].set_yticks(range(n_tokens))
    axes[2].set_xticklabels(tokens[:n_tokens], rotation=45, ha='right', fontsize=7)
    axes[2].set_yticklabels(tokens[:n_tokens], fontsize=7)
    axes[2].set_title('注意力权重热力图\n(Layer 1, Head 1)')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\n结果图已保存为 {save_path}")
    plt.close()


def predict_sample(model, vocab, text):
    """对单条文本做预测"""
    model.eval()
    tokens = simple_tokenizer(text)[:MAX_LEN]
    token_ids = vocab(tokens)
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits, _ = model(token_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()

    return CLASS_NAMES[pred_class], probs[0].cpu().numpy()


# ==============================================================================
# 第 6 部分：主函数
# ==============================================================================
def main():
    print("=" * 60)
    print("Transformer 文本分类实验 — AG_NEWS 数据集")
    print("论文: Attention Is All You Need (Vaswani et al., 2017)")
    print("=" * 60)

    # ----- 数据 -----
    print("\n[1/5] 加载数据...")
    train_loader, test_loader, vocab = get_dataloaders()
    print(f"  训练集 batch 数: {len(train_loader)}")
    print(f"  测试集 batch 数: {len(test_loader)}")

    # ----- 模型 -----
    print("\n[2/5] 构建模型...")
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        n_classes=4,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    print(f"  架构: d_model={D_MODEL}, n_heads={N_HEADS}, d_ff={D_FF}, n_layers={N_LAYERS}")

    # ----- 训练 -----
    print("\n[3/5] 开始训练...")
    history = train_model(model, train_loader, test_loader)

    print(f"\n训练完成！")
    print(f"  总耗时: {history['train_time']:.1f}s")
    print(f"  最终测试准确率: {history['test_accs'][-1]:.4f}")

    # ----- 预测示例 -----
    print("\n[4/5] 预测示例...")
    sample_texts = [
        "Apple announced a new iPhone with advanced AI features",
        "The stock market surged after the Federal Reserve meeting",
        "Lionel Messi scored a hat-trick in the World Cup final",
        "The United Nations held a summit on climate change",
    ]
    for text in sample_texts:
        label, probs = predict_sample(model, vocab, text)
        print(f"  文本: {text[:50]}...")
        print(f"  预测: {label} | 概率: {dict(zip(CLASS_NAMES, [f'{p:.3f}' for p in probs]))}")

    # ----- 可视化 -----
    print("\n[5/5] 生成可视化...")
    plot_results(history, model, test_loader, vocab)

    # ----- 总结 -----
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"  数据集: AG_NEWS（4 类新闻分类）")
    print(f"  模型: Transformer Encoder（{N_LAYERS} 层, {N_HEADS} 头）")
    print(f"  参数量: {total_params:,}")
    print(f"  测试准确率: {history['test_accs'][-1]:.4f}")
    print(f"  训练时间: {history['train_time']:.1f}s")
    print("=" * 60)


# ==============================================================================
if __name__ == '__main__':
    main()
    print("\n所有实验运行完成！")
