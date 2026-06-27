"""
Informer 长序列时间序列预测 - 单文件版

把数据准备、模型定义、训练、评估和可视化全部收进一个脚本里，
不再拆成多个模块，方便查看、运行和修改。

实验目标：
- MLP：无时序结构的基线模型
- LSTM：经典时序模型
- Informer：稀疏注意力长序列模型

数据：真实 ETTh1 数据集（电力变压器油温，约17420 小时，7 个特征）
任务：用历史 96 小时预测未来 24 小时油温 (OT)
"""

from __future__ import annotations

import copy
import math
import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================== 全局配置 ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "ETTh1.csv")
ETT_DATASET_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

RANDOM_SEED = 42
SEQ_LEN = 96
PRED_LEN = 24
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.65
VAL_RATIO = 0.15
TEST_RATIO = 0.20
TARGET_COL = "OT"

import matplotlib.font_manager as _fm

# 注册中文字体：优先 Microsoft YaHei，备选 SimHei
_CN_FONT_PATH = None
for _name in ["msyh.ttc", "simhei.ttf"]:
    _candidate = os.path.join("C:\\Windows\\Fonts", _name)
    if os.path.exists(_candidate):
        _CN_FONT_PATH = _candidate
        break
if _CN_FONT_PATH:
    _fm.fontManager.addfont(_CN_FONT_PATH)
    _cn_family = _fm.FontProperties(fname=_CN_FONT_PATH).get_name()
    plt.rcParams["font.sans-serif"] = [_cn_family] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    print(f"[OK] 中文字体已注册: {_cn_family} ({_CN_FONT_PATH})")
else:
    print("[WARN] 未找到中文字体，图表中文可能显示为方块")

sns.set_style("darkgrid")

# seaborn 会重置 rcParams，重新应用中文字体配置
if _CN_FONT_PATH:
    _cn_family2 = _fm.FontProperties(fname=_CN_FONT_PATH).get_name()
    plt.rcParams["font.sans-serif"] = [_cn_family2] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


# ============================== 工具函数 ==============================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================== 数据生成与加载 ==============================
def download_ett_dataset(save_dir: str = DATA_DIR) -> str:
    """从 GitHub 下载真实 ETTh1 数据集，如果已存在则跳过。"""
    ensure_dir(save_dir)
    file_path = os.path.join(save_dir, "ETTh1.csv")
    if not os.path.exists(file_path):
        print(f"✓ 正在下载真实 ETTh1 数据集...")
        print(f"  来源: {ETT_DATASET_URL}")
        import urllib.request
        urllib.request.urlretrieve(ETT_DATASET_URL, file_path)
        print(f"  已保存: {file_path}")
    else:
        print(f"✓ 数据文件已存在: {file_path}")
    return file_path


class TimeSeriesDataset:
    """时间序列数据集处理类：归一化、滑动窗口、反归一化。"""

    def __init__(self, data_path: str, seq_len: int = SEQ_LEN, pred_len: int = PRED_LEN, scale: bool = True, target_col: str = TARGET_COL):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.target_col = target_col
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.df = pd.read_csv(data_path)
        if "date" not in self.df.columns:
            raise ValueError("数据文件必须包含 date 列")
        if target_col not in self.df.columns:
            raise ValueError(f"目标列 {target_col} 不存在于数据文件中")

        self.feature_columns = [col for col in self.df.columns if col != "date"]
        self.target_idx = self.feature_columns.index(target_col)
        self.data = self.df[self.feature_columns].values.astype(np.float32)
        self.n_features = self.data.shape[1]

        print(f"✓ 数据加载成功: {self.df.shape}")
        print(f"  特征列: {self.feature_columns}")
        print(f"  目标列: {self.target_col} (索引 {self.target_idx})")

    def fit_scaler_on_train(self, train_end_idx: int) -> None:
        """仅在训练集上拟合归一化器，然后应用到全部数据。"""
        if not self.scale:
            return
        self.scaler.fit(self.data[:train_end_idx])
        self.data = self.scaler.transform(self.data)
        print("✓ 归一化器已在训练集上拟合，已应用到全部数据")

    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(data) < self.seq_len + self.pred_len:
            raise ValueError("数据长度不足，无法创建滑动窗口样本")

        x_list, y_list = [], []
        for idx in range(len(data) - self.seq_len - self.pred_len + 1):
            x_list.append(data[idx : idx + self.seq_len])
            y_list.append(data[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx])

        return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)

    def create_sliding_windows(self, test_ratio: float = TEST_RATIO, val_ratio: float = VAL_RATIO):
        total_len = len(self.data)
        train_len = int(total_len * (1 - test_ratio - val_ratio))
        val_len = int(total_len * val_ratio)

        # 先在训练集上拟合归一化器，再应用到全部数据
        self.fit_scaler_on_train(train_len)

        train_data = self.data[:train_len]
        val_data = self.data[train_len : train_len + val_len]
        test_data = self.data[train_len + val_len :]

        x_train, y_train = self._create_sequences(train_data)
        x_val, y_val = self._create_sequences(val_data)
        x_test, y_test = self._create_sequences(test_data)

        print("\n=== 数据集划分 ===")
        print(f"训练集: {x_train.shape[0]} 样本, X {x_train.shape}, y {y_train.shape}")
        print(f"验证集: {x_val.shape[0]} 样本, X {x_val.shape}, y {y_val.shape}")
        print(f"测试集: {x_test.shape[0]} 样本, X {x_test.shape}, y {y_test.shape}")

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def inverse_transform(self, data):
        if not self.scale:
            return data

        data = np.asarray(data)
        flat = data.reshape(-1)
        dummy = np.zeros((flat.shape[0], self.n_features), dtype=np.float32)
        dummy[:, self.target_idx] = flat
        restored = self.scaler.inverse_transform(dummy)
        return restored[:, self.target_idx].reshape(data.shape)


# ============================== 模型定义 ==============================
class MLPRegressor(nn.Module):
    """多层感知机：基础对照模型，没有显式时序结构。"""

    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = PRED_LEN, num_layers: int = 3):
        super().__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.net(x)


class LSTMRegressor(nn.Module):
    """LSTM 递归网络：经典时序对照模型。"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = PRED_LEN):
        super().__init__()
        dropout = 0.1 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class ProbSparseAttention(nn.Module):
    """ProbSparse 稀疏注意力：Informer 的核心思想。"""

    def __init__(self, d_model: int, n_heads: int = 8, factor: int = 5):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.shape

        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        top_k = max(1, min(seq_len, seq_len // self.factor))
        top_k_scores, top_k_indices = torch.topk(scores, top_k, dim=-1)

        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(-1, top_k_indices, top_k_scores)

        attention = torch.softmax(sparse_scores, dim=-1)
        attention = torch.where(torch.isinf(sparse_scores), torch.zeros_like(attention), attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.fc_out(context)


class InformerEncoderLayer(nn.Module):
    """Informer 编码器层：稀疏注意力 + 前馈网络 + 残差连接。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class InformerRegressor(nn.Module):
    """Informer 长序列预测模型。"""

    def __init__(self, input_size: int, seq_len: int, pred_len: int, d_model: int = 64, n_heads: int = 8, n_encoder_layers: int = 3, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.embedding = nn.Linear(input_size, d_model)
        pos_encoding = self._create_positional_encoding(seq_len, d_model)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)

        self.encoder_layers = nn.ModuleList(
            [InformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, pred_len)

    def _create_positional_encoding(self, seq_len: int, d_model: int):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = x.mean(dim=1)
        return self.fc(x)


# ============================== 训练与评估 ==============================
class EarlyStopping:
    """早停策略：防止验证损失长期不下降。"""

    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class Trainer:
    """训练器：训练、验证、测试、保存最佳权重。"""

    def __init__(self, model, train_loader, val_loader, test_loader, device: str = "cuda" if torch.cuda.is_available() else "cpu", learning_rate: float = LEARNING_RATE):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        self.early_stopping = EarlyStopping(patience=12)

        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for x_batch, y_batch in tqdm(self.train_loader, desc="训练中", leave=False):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()

        return total_loss / max(1, len(self.val_loader))

    def train(self, epochs: int = EPOCHS):
        print(f"\n开始训练 ({epochs} epochs)...")
        print(f"设备: {self.device}")
        print("=" * 70)

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss - 1e-8:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | 训练损失: {train_loss:.6f} | "
                    f"验证损失: {val_loss:.6f} | 最优: Epoch {self.best_epoch} | LR: {current_lr:.2e}"
                )

            if self.early_stopping(val_loss):
                print("\n⚠️  验证集损失长期没有改善，触发早停。")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        print(f"\n✓ 训练完成！最佳模型在第 {self.best_epoch} 个 epoch，验证损失 {self.best_val_loss:.6f}")

    def test(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_pred = self.model(x_batch)
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        test_loss = float(np.mean((all_preds - all_targets) ** 2))
        return all_preds, all_targets, test_loss

    def get_training_history(self):
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }


def create_data_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size: int = BATCH_SIZE):
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.FloatTensor(y_val)
    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class TimeSeriesMetrics:
    """时间序列指标：MAE、MSE、RMSE、R²、MAPE、SMAPE。"""

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def mape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def smape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        smape_value = np.zeros_like(denominator, dtype=np.float32)
        smape_value[mask] = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
        return float(np.mean(smape_value) * 100)

    @staticmethod
    def evaluate_all(y_true, y_pred, model_name: str = "Model"):
        metrics = {
            "MAE": TimeSeriesMetrics.mae(y_true, y_pred),
            "MSE": TimeSeriesMetrics.mse(y_true, y_pred),
            "RMSE": TimeSeriesMetrics.rmse(y_true, y_pred),
            "R²": TimeSeriesMetrics.r2(y_true, y_pred),
            "MAPE": TimeSeriesMetrics.mape(y_true, y_pred),
            "SMAPE": TimeSeriesMetrics.smape(y_true, y_pred),
        }

        print(f"\n{model_name} 评估结果:")
        print("=" * 60)
        for key, value in metrics.items():
            if key == "R²":
                print(f"  {key:8s}: {value:8.4f}  (越接近 1 越好)")
            elif key in {"MAPE", "SMAPE"}:
                print(f"  {key:8s}: {value:8.2f}%")
            else:
                print(f"  {key:8s}: {value:8.4f}")
        return metrics


class ErrorAnalysis:
    """误差分析工具。"""

    @staticmethod
    def time_steps_mae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim == 1:
            return np.abs(y_true - y_pred)
        return np.abs(y_true - y_pred).mean(axis=0)

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.ndim == 1:
            if y_true.shape[0] < 2:
                return 0.0
            true_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)
            return float((np.sign(true_diff) == np.sign(pred_diff)).mean() * 100)

        if y_true.shape[1] < 2:
            return 0.0
        true_diff = np.diff(y_true, axis=1)
        pred_diff = np.diff(y_pred, axis=1)
        return float((np.sign(true_diff) == np.sign(pred_diff)).mean() * 100)

    @staticmethod
    def peak_detection(y_true, y_pred, threshold: float = 0.75):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        y_true_peak = y_true > np.quantile(y_true, threshold)
        y_pred_peak = y_pred > np.quantile(y_pred, threshold)
        return float((y_true_peak == y_pred_peak).mean() * 100)


# ============================== 可视化 ==============================
def plot_training_history(histories, save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for model_name, history in histories.items():
        axes[0].plot(history["train_losses"], label=model_name, linewidth=2)
        axes[1].plot(history["val_losses"], label=model_name, linewidth=2)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("训练损失 (MSE)", fontsize=12)
    axes[0].set_title("训练损失曲线", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("验证损失 (MSE)", fontsize=12)
    axes[1].set_title("验证损失曲线", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 训练曲线已保存: {save_path}")
    plt.close(fig)


def plot_predictions_comparison(y_test, predictions_dict, time_steps: int = 100, save_path: str | None = None):
    if y_test.shape[0] == 0:
        print("❌ 没有测试数据")
        return

    y_test_plot = y_test[:time_steps].flatten()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        if idx >= 4:
            break
        y_pred_plot = y_pred[:time_steps].flatten()
        ax = axes[idx]
        ax.plot(y_test_plot, label="真实值", linewidth=2.2, color="blue")
        ax.plot(y_pred_plot, label=f"{model_name} 预测", linewidth=2, color="red", linestyle="--")
        ax.set_xlabel("时间步", fontsize=11)
        ax.set_ylabel("油温（归一化）", fontsize=11)
        ax.set_title(f"{model_name} - 预测结果对比", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(len(predictions_dict), 4):
        axes[idx].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 预测对比已保存: {save_path}")
    plt.close(fig)


def plot_metrics_comparison(metrics_dict, save_path: str | None = None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    metric_names = ["MAE", "MSE", "RMSE", "R²", "MAPE", "SMAPE"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"]
    models = list(metrics_dict.keys())

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        values = [metrics_dict[model][metric_name] for model in models]
        bars = ax.bar(models, values, color=colors[idx], alpha=0.85, edgecolor="black")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(metric_name, fontsize=11, fontweight="bold")
        ax.set_title(f"{metric_name} 对比", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        if metric_name == "R²":
            ax.set_ylim([0, 1])
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 指标对比已保存: {save_path}")
    plt.close(fig)


def plot_error_analysis(y_test, predictions_dict, save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for model_name, y_pred in predictions_dict.items():
        errors = np.abs(y_test.flatten() - y_pred.flatten())
        ax.hist(errors, bins=30, alpha=0.55, label=model_name, edgecolor="black")
    ax.set_xlabel("绝对误差", fontsize=11)
    ax.set_ylabel("频率", fontsize=11)
    ax.set_title("预测误差分布", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for model_name, y_pred in predictions_dict.items():
        mae_per_step = ErrorAnalysis.time_steps_mae(y_test, y_pred)
        ax.plot(range(len(mae_per_step)), mae_per_step, marker="o", linewidth=2.2, label=model_name)
    ax.set_xlabel("预测时步", fontsize=11)
    ax.set_ylabel("平均绝对误差 (MAE)", fontsize=11)
    ax.set_title("长序列误差增长分析", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 误差分析已保存: {save_path}")
    plt.close(fig)


def create_summary_report(metrics_dict, histories, save_path: str | None = None):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.28)
    fig.suptitle("Informer 长序列时间序列预测 - 单文件实验报告", fontsize=18, fontweight="bold", y=0.98)

    models = list(metrics_dict.keys())

    ax1 = fig.add_subplot(gs[0, 0])
    r2_scores = [metrics_dict[m]["R²"] for m in models]
    colors = ["#FF6B6B" if r2 < 0.7 else "#4ECDC4" if r2 < 0.85 else "#45B7D1" for r2 in r2_scores]
    bars = ax1.bar(models, r2_scores, color=colors, alpha=0.85, edgecolor="black")
    for bar, val in zip(bars, r2_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylim([0, 1])
    ax1.set_ylabel("R²", fontweight="bold")
    ax1.set_title("非线性拟合能力对比", fontweight="bold")
    ax1.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax2 = fig.add_subplot(gs[0, 1])
    rmse_scores = [metrics_dict[m]["RMSE"] for m in models]
    ax2.bar(models, rmse_scores, color="#FFA07A", alpha=0.85, edgecolor="black")
    ax2.set_ylabel("RMSE", fontweight="bold")
    ax2.set_title("均方根误差对比", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax3 = fig.add_subplot(gs[1, 0])
    for model, hist in histories.items():
        ax3.plot(hist["train_losses"], label=f"{model}-训练", linewidth=2)
        ax3.plot(hist["val_losses"], label=f"{model}-验证", linewidth=2, linestyle="--")
    ax3.set_xlabel("Epoch", fontweight="bold")
    ax3.set_ylabel("损失 (MSE)", fontweight="bold")
    ax3.set_title("训练过程曲线", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("tight")
    ax4.axis("off")
    table_data = [["模型", "R²", "RMSE", "MAE"]]
    for model in models:
        table_data.append([
            model,
            f"{metrics_dict[model]['R²']:.3f}",
            f"{metrics_dict[model]['RMSE']:.3f}",
            f"{metrics_dict[model]['MAE']:.3f}",
        ])
    table = ax4.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for row in range(len(table_data)):
        for col in range(4):
            cell = table[(row, col)]
            if row == 0:
                cell.set_facecolor("#4ECDC4")
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#E8F4F8" if row % 2 == 0 else "white")
    ax4.set_title("核心指标汇总", fontweight="bold", pad=16)

    best_model = max(models, key=lambda m: metrics_dict[m]["R²"])
    best_r2 = metrics_dict[best_model]["R²"]

    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    conclusion_text = f"""
【 实验结论 】

* 最优模型: {best_model} (R² = {best_r2:.4f})

* 关键发现:
  1. Informer 通过稀疏自注意力捕捉长序列依赖。
  2. LSTM 能建模时序，但长距离依赖能力有限。
  3. MLP 没有显式时间结构，属于最弱基线。
  4. 误差分析图能直观看到长序列预测的偏差增长情况。

* 本单文件版已把数据、模型、训练、评估、可视化全部收拢到一个脚本里，
  便于查看和修改，不再需要在多个文件间来回跳转。
"""
    ax5.text(
        0.03,
        0.95,
        conclusion_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        family=plt.rcParams["font.sans-serif"][0],
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.45),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 完整报告已保存: {save_path}")
    plt.close(fig)


# ============================== 主程序 ==============================
def main():
    set_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dir(DATA_DIR)
    ensure_dir(RESULTS_DIR)

    print("=" * 80)
    print("Informer 长序列时间序列预测 - 单文件实验")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"输入序列长度: {SEQ_LEN} 小时")
    print(f"预测长度: {PRED_LEN} 小时")
    print(f"训练轮数: {EPOCHS}")
    print(f"批大小: {BATCH_SIZE}")
    print("=" * 80)

    # 第1步：数据准备
    print("\n[第1步] 数据准备...")
    data_path = download_ett_dataset(DATA_DIR)

    dataset = TimeSeriesDataset(
        data_path=data_path,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        scale=True,
        target_col=TARGET_COL,
    )

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset.create_sliding_windows(
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size=BATCH_SIZE
    )

    # 第2步：建立模型
    print("\n[第2步] 建立模型...")
    model_specs = [
        ("MLP", MLPRegressor(input_size=SEQ_LEN * dataset.n_features, hidden_size=128, output_size=PRED_LEN, num_layers=3)),
        ("LSTM", LSTMRegressor(input_size=dataset.n_features, hidden_size=64, num_layers=2, output_size=PRED_LEN)),
        (
            "Informer",
            InformerRegressor(
                input_size=dataset.n_features,
                seq_len=SEQ_LEN,
                pred_len=PRED_LEN,
                d_model=64,
                n_heads=8,
                n_encoder_layers=3,
                d_ff=256,
                dropout=0.1,
            ),
        ),
    ]

    for model_name, model in model_specs:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {model_name:8s} 参数量: {param_count:,}")

    # 第3步：训练与测试
    print("\n[第3步] 训练与评估...")
    trainers = {}
    histories = {}
    predictions = {}
    all_metrics = {}

    for model_name, model in model_specs:
        print("\n" + "-" * 70)
        print(f"{model_name} 开始训练")
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device, learning_rate=LEARNING_RATE)
        trainer.train(epochs=EPOCHS)
        y_pred, y_true, test_loss = trainer.test()
        metrics = TimeSeriesMetrics.evaluate_all(y_true, y_pred, model_name)

        trainers[model_name] = trainer
        histories[model_name] = trainer.get_training_history()
        predictions[model_name] = y_pred
        all_metrics[model_name] = metrics

        dir_acc = ErrorAnalysis.directional_accuracy(y_true, y_pred)
        peak_acc = ErrorAnalysis.peak_detection(y_true, y_pred)
        print(f"  方向准确率: {dir_acc:.2f}%")
        print(f"  峰值检测准确率: {peak_acc:.2f}%")
        print(f"  测试集 MSE: {test_loss:.6f}")

    # 第4步：排序与总结
    metrics_df = pd.DataFrame(all_metrics).T.sort_values("R²", ascending=False).round(4)
    print("\n" + "=" * 80)
    print("所有模型回归指标对比表")
    print("=" * 80)
    print(metrics_df.to_string())

    best_model_name = metrics_df.index[0]
    best_r2 = metrics_df.loc[best_model_name, "R²"]
    print(f"\n✓ 最优模型: {best_model_name}")
    print(f"  R² 分数: {best_r2:.4f}")

    if "Informer" in all_metrics:
        informer_r2 = all_metrics["Informer"]["R²"]
        lstm_r2 = all_metrics["LSTM"]["R²"]
        mlp_r2 = all_metrics["MLP"]["R²"]
        print("\n✓ 非线性拟合提升：")
        print(f"  相比 LSTM：{(informer_r2 - lstm_r2):.4f}")
        print(f"  相比 MLP ：{(informer_r2 - mlp_r2):.4f}")

    # 第5步：可视化
    print("\n[第4步] 生成可视化图表...")
    plot_training_history(histories, save_path=os.path.join(RESULTS_DIR, "01_training_history.png"))
    plot_predictions_comparison(y_test, predictions, time_steps=100, save_path=os.path.join(RESULTS_DIR, "02_predictions_comparison.png"))
    plot_metrics_comparison(all_metrics, save_path=os.path.join(RESULTS_DIR, "03_metrics_comparison.png"))
    plot_error_analysis(y_test, predictions, save_path=os.path.join(RESULTS_DIR, "04_error_analysis.png"))
    create_summary_report(all_metrics, histories, save_path=os.path.join(RESULTS_DIR, "05_summary_report.png"))

    # 第6步：结论输出
    print("\n[第5步] 实验结论")
    print("=" * 80)
    print(f"最优模型: {best_model_name}")
    print(f"R² = {best_r2:.4f}")
    print("\n关键结论：")
    print("  - MLP 适合作为无时序结构基线。")
    print("  - LSTM 能利用短期依赖，但在长序列上通常不如注意力模型。")
    print("  - Informer 通过稀疏注意力更适合长序列复杂非线性预测。")
    print("  - 单文件版已把所有步骤收拢，便于直接查看和运行。")

    print("\n所有结果已保存到 results/ 文件夹：")
    print("  - 01_training_history.png")
    print("  - 02_predictions_comparison.png")
    print("  - 03_metrics_comparison.png")
    print("  - 04_error_analysis.png")
    print("  - 05_summary_report.png")
    print("=" * 80)
    print("实验完成！")


if __name__ == "__main__":
    main()
