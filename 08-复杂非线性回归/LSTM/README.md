# Informer 长序列时间序列预测 - 单文件版

MLP / LSTM / Informer 三种模型的时间序列预测对比实验，所有代码整合在 `main_single.py` 一个文件中。

## 运行

```bash
pip install -r requirements.txt
python main_single.py
```

自动检测 GPU/CPU，图表输出到 `results/` 目录。

## 数据

脚本自动生成 ETTh1 风格合成数据：8760 小时、7 个特征、目标列 `load`，历史 96 小时预测未来 24 小时。

## 输出

| 文件 | 内容 |
|------|------|
| `01_training_history.png` | 训练/验证损失曲线 |
| `02_predictions_comparison.png` | 真实值 vs 预测值 |
| `03_metrics_comparison.png` | MAE/MSE/RMSE/R²/MAPE/SMAPE 对比 |
| `04_error_analysis.png` | 误差分布与逐步误差 |
| `05_summary_report.png` | 实验总结 |

## 常见问题

- **CUDA out of memory**：改小 `BATCH_SIZE = 16` 或 `8`
- **训练慢**：减少 `EPOCHS` 或数据量，或使用 GPU
- **R² 偏低**：正常，时序预测难度大，可调超参数

## 可做的实验

1. 改变序列长度 `[48, 96, 192, 336]`，观察 Informer 优势变化
2. 改变预测长度 `[6, 12, 24, 48]`，对比长时预测稳定性
3. 改变 Informer 编码器层数 `[1, 3, 5, 7]`

## 参考论文

- [Informer (2020)](https://arxiv.org/abs/2012.07436)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
