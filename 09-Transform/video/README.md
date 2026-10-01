# 🎬 什么是 Transformer？—— 一部用 JavaScript 画出来的讲解视频

> 参考论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)（Vaswani et al., NeurIPS 2017）
> 目标观众：高中生也能看懂，但要真正讲到 Attention 的数学。

| 文件 | 说明 |
|---|---|
| `什么是Transformer.mp4` | 成片：1080p / 30fps / 约 9 分钟，中文配音 + 字幕 |
| `index.html` | 交互式播放器：双击即可在浏览器打开，可拖动进度、跳章节、开关字幕、倍速 |
| `playground.html` | **注意力实验室**：亲手拖动 Query/Key、调节 √dₖ、切换遮罩、观察位置编码 |
| `assets/subtitles.srt` | 字幕文件 |

## 内容（15 个章节）

| 章节 | 讲什么 | 数学 |
|---|---|---|
| 序章 | 2017 年的论文；GPT 里的 T | — |
| 01 把词变成数字 | 词向量 = 地图上的坐标；论文里 512 维 | 向量、坐标 |
| 02 旧方法的烦恼 | RNN 像传话游戏：越传越模糊、没法并行 | — |
| 03 为什么需要注意力 | “小猫没有跳上桌子，因为它太累 / 高了” | — |
| 04 Query、Key、Value | 图书馆类比；`q = x·W_Q` | 矩阵乘法 |
| 05 点积 | 对应相乘再相加；`a·b = |a||b|cosθ` | 点积、余弦 |
| 06 Softmax | 分数 → eˣ → 除以总和 | 指数函数 |
| 07 混合 Value | 把 Value 想成颜料，按比例调色 | 加权求和 |
| 08 最著名的公式 | `softmax(QKᵀ/√dₖ)V`；为什么除以 √dₖ（随机 64 维点积的真实分布） | 方差、√64 = 8 |
| 09 多头注意力 | 8 个头各看一种关系，8×64 = 512 拼接后乘 W_O | — |
| 10 位置编码 | sin/cos 像转速不同的时钟；相对位置 = 旋转 | 三角函数和角公式 |
| 11 编码器 | 注意力 + 前馈网络（512→2048→512, ReLU）+ 残差 & 层归一化，叠 6 层 | max(0, x) |
| 12 解码器 | 逐词生成；遮罩把未来设为 −∞（e^−∞ = 0）；交叉注意力 | — |
| 13 完整结构 | 论文 Figure 1；训练；BLEU 28.4 / 41.8；8 块 P100 训练 12 小时 / 3.5 天 | 梯度下降 |
| 尾声 | BERT、GPT、ChatGPT；图片/声音/蛋白质也能当“词” | — |

画面中的数字都是 **JS 实时算出来的**：点积、softmax、混合颜色、随机 64 维向量的点积分布、位置编码热力图都是真实计算结果，不是贴图。

## 它是怎么做出来的

```
src/script.js     旁白脚本（字幕文字 + 发音 + 停顿）
src/engine.js     Canvas 2D 小引擎：缓动、箭头、矩阵、迷你公式排版（分式 / 根号 / 上下标）
src/scenes-*.js   15 个分镜，每个都是 draw(ctx, 时间) 的纯函数
src/player.js     播放器 + 渲染模式（?render=1）
src/timing.js     自动生成：每句旁白的真实起止时间，动画据此对齐口型
src/fonts.js      自动生成：裁剪后的霞鹜文楷 + KaTeX 字体（只含用到的字，约 200 KB）
tools/            配音、字体裁剪、逐帧导出视频的脚本
```

**画面是时间的纯函数**：`render(t)` 只依赖 t，所以播放器可以任意拖动，导出时也能逐帧精确渲染。

### 重新生成

```bash
# 1) 配音：离线 TTS（sherpa-onnx + MeloTTS 中英混合模型），生成 timing.js / narration.m4a / 字幕
pip install sherpa-onnx soundfile numpy
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2
tar xjf vits-melo-tts-zh_en.tar.bz2
python tools/build_audio.py --model-dir vits-melo-tts-zh_en

# 2) 字体（改了文字后需要重跑）
pip install fonttools brotli
python tools/subset_fonts.py --wenkai LXGWWenKai-Medium.ttf --katex-dir node_modules/katex/dist/fonts

# 3) 导出 MP4（需要 playwright + ffmpeg）
node tools/render.js                 # 完整视频
node tools/render.js --snap 60,120   # 只截几帧调试；tools/sheet.sh 拼成预览图
```

改旁白只需编辑 `src/script.js` 后重跑第 1、2 步，动画会自动对齐新的时间轴。

## 致谢与许可

- 字体：[霞鹜文楷](https://github.com/lxgw/LxgwWenKai)、[KaTeX fonts](https://github.com/KaTeX/KaTeX)（SIL Open Font License 1.1）
- 配音：[MeloTTS](https://github.com/myshell-ai/MeloTTS)（MIT），经 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 离线运行
- 论文数据：Vaswani et al., *Attention Is All You Need*, 2017（表 2、第 5 节）
