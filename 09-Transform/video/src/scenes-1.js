/*
 * 分镜（上）：序章 → 词向量 → RNN → 歧义句 → QKV → 点积
 *
 * 每一幕是 draw(ctx, S)：
 *   S.t      该幕内的当前时间（秒）
 *   S.L(i)   第 i 句旁白开始的时间；S.E(i) 结束时间
 *   S.at(i, f)  第 i 句进行到比例 f 时的时间
 *   S.p(i, off, len)  从第 i 句开始 off 秒后、持续 len 秒的缓动进度（0→1）
 */
(function () {
  const T = window.TV;
  const { C, W, H, text, chip, arrow, arcArrow, line, circle, roundRect, vecCells, grid, math, M, panel, typeText } = T;
  const { clamp, lerp, ease, easeOut, easeBack, prog } = T;

  // 贯穿全片的例句和一组真实可算的数字
  const SENT = ['小猫', '没有', '跳上', '桌子', '因为', '它', '太', '累了'];
  const IT = 5; // 「它」的位置
  const Q_IT = [1, 0.5, 0, 0.5];
  const KEYS = [[2, 1, 0, 2], [0, -1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [-0.5, 0, 1, 0], [0.5, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 1]];
  const SCORES = KEYS.map((k) => T.dot(Q_IT, k));
  const WEIGHTS = T.softmax(SCORES);
  const VCOLORS = ['#ff9a3c', '#8a8fa8', '#4ade80', '#3b82f6', '#a3a3c2', '#e5e7eb', '#2dd4bf', '#b46cff'];
  window.SHARED = { SENT, IT, Q_IT, KEYS, SCORES, WEIGHTS, VCOLORS };

  const scenes = {};

  // ------------------------------------------------------------------ 序章
  scenes.intro = function (ctx, S) {
    const t = S.t;
    // 第一段：论文标题（打字机）
    const a1 = 1 - S.p(2, 0, 0.8);
    if (a1 > 0) {
      ctx.save(); ctx.globalAlpha = a1;
      const tp = prog(t, 0.6, 2.2);
      text(ctx, 'arXiv:1706.03762  ·  NeurIPS 2017', W / 2, 250, { size: 30, color: C.dim, family: 'main', alpha: S.p(0, 0.2, 0.8) });
      const title = 'Attention Is All You Need';
      const n = typeText(ctx, title, W / 2, 380, tp, { size: 104, family: 'main', weight: 400, glow: 18, glowColor: 'rgba(120,170,255,0.6)' });
      if (tp < 1 || Math.floor(t * 2) % 2 === 0) {
        const w = T.measure(ctx, title.slice(0, n), { size: 104, family: 'main' });
        const fullW = T.measure(ctx, title, { size: 104, family: 'main' });
        ctx.fillStyle = C.blue; ctx.fillRect(W / 2 - fullW / 2 + w + 6, 330, 5, 100);
      }
      const authors = ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar', 'Jakob Uszkoreit', 'Llion Jones', 'Aidan N. Gomez', 'Łukasz Kaiser', 'Illia Polosukhin'];
      authors.forEach((a, i) => {
        const x = W / 2 + (i % 4 - 1.5) * 380, y = 510 + Math.floor(i / 4) * 52;
        text(ctx, a, x, y, { size: 30, family: 'main', color: C.dim, alpha: S.p(0, 2.6 + i * 0.12, 0.6) });
      });
      text(ctx, 'Google Brain · Google Research', W / 2, 640, { size: 28, family: 'main', color: C.faint, alpha: S.p(0, 3.8, 0.8) });
      const zh = S.p(1, 0, 0.8);
      text(ctx, '“你只需要注意力”', W / 2, 760, { size: 64, color: C.Q, alpha: zh, glow: 20 * zh });
      ctx.restore();
    }

    // 第二段：Transformer 字母聚拢
    const a2 = S.p(2, 0, 0.6) * (1 - S.p(4, 0, 0.8));
    if (a2 > 0) {
      ctx.save(); ctx.globalAlpha = a2;
      const word = 'Transformer';
      const size = 150;
      const total = T.measure(ctx, word, { size, family: 'main' });
      const r = T.rng(7);
      const gather = easeOut(prog(t, S.L(2) + 0.1, 1.6));
      const yWord = lerp(430, 330, S.p(3, 0, 1));
      let x = W / 2 - total / 2;
      for (let i = 0; i < word.length; i++) {
        const ch = word[i];
        const cw = T.measure(ctx, ch, { size, family: 'main' });
        const sx = r() * W, sy = r() * H, rot = (r() - 0.5) * 4;
        const px = lerp(sx, x + cw / 2, gather), py = lerp(sy, yWord, gather);
        ctx.save(); ctx.translate(px, py); ctx.rotate(rot * (1 - gather));
        const isT = i === 0 && S.t > S.L(3);
        text(ctx, ch, 0, 0, { size, family: 'main', weight: 400, color: isT ? C.Q : C.text, glow: isT ? 30 : 12, glowColor: isT ? C.Q : 'rgba(120,170,255,0.7)' });
        ctx.restore();
        x += cw;
      }
      // GPT 展开
      const g = S.p(3, 1.2, 1.0);
      if (g > 0) {
        const parts = [['G', 'enerative', C.blue], ['P', 're-trained', C.green], ['T', 'ransformer', C.Q]];
        const y = 620;
        const xs = [W / 2 - 520, W / 2 - 70, W / 2 + 380];
        parts.forEach(([a, b, col], i) => {
          const gi = S.p(3, 1.2 + i * 0.35, 0.8);
          text(ctx, a, xs[i], y, { size: 120, family: 'main', color: col, alpha: gi, glow: 16 });
          const e = S.p(3, 2.6 + i * 0.3, 0.8);
          text(ctx, b, xs[i] + 44, y + 22, { size: 46, family: 'main', color: C.dim, align: 'left', alpha: e });
        });
        const tp = S.p(3, 3.8, 1.2);
        if (tp > 0) arcArrow(ctx, xs[2] - 10, y - 80, W / 2 - total / 2 + 50, yWord + 80, 60, { color: C.Q, lw: 4, progress: tp, glow: 10 });
      }
      ctx.restore();
    }

    // 第三段：学习路线
    const a3 = S.p(4, 0, 0.8);
    if (a3 > 0) {
      ctx.save(); ctx.globalAlpha = a3;
      text(ctx, '今天的路线图', W / 2, 220, { size: 52, color: C.text });
      const stops = ['词向量', '注意力', 'Q·K·V', 'Softmax', '多头', '位置编码', '编码器', '解码器', '完整模型'];
      const x0 = 220, x1 = W - 220, y = 470;
      const pp = easeOut(prog(t, S.L(4) + 0.3, 3.2));
      line(ctx, x0, y, lerp(x0, x1, pp), y, { color: C.faint, lw: 6 });
      stops.forEach((s, i) => {
        const x = lerp(x0, x1, i / (stops.length - 1));
        const ai = clamp((pp - i / (stops.length - 1)) * 6 + 0.5);
        const col = [C.blue, C.Q, C.K, C.V, C.purple, C.green, C.orange, C.red, C.Q][i];
        circle(ctx, x, y, 18 * easeBack(ai), { fill: col, glow: 20 });
        text(ctx, s, x, y + (i % 2 ? 78 : -72), { size: 36, color: col, alpha: ai });
      });
      text(ctx, '需要的数学：向量 · 乘法和加法 · 指数函数 eˣ · sin / cos', W / 2, 700, { size: 36, color: C.dim, alpha: S.p(4, 2.4, 1) });
      ctx.restore();
    }
  };

  // ------------------------------------------------------------------ 01 词向量
  scenes.embedding = function (ctx, S) {
    const t = S.t;
    // 左侧：词 → 数字
    const lx = 430;
    const leftFade = 1 - 0.65 * S.p(3, 0, 0.8);
    ctx.save(); ctx.globalAlpha = leftFade;
    const a0 = S.p(0, 0, 0.7);
    chip(ctx, '猫', lx, 300, { size: 80, w: 170, h: 150, alpha: a0, stroke: C.orange, glow: 12 });
    // 计算机屏幕
    const sc = S.p(0, 1.0, 0.7);
    roundRect(ctx, lx - 170, 470, 340, 200, 18, { fill: 'rgba(10,20,40,0.9)', stroke: C.dim, lw: 3, alpha: sc });
    roundRect(ctx, lx - 40, 672, 80, 30, 4, { fill: C.faint, alpha: sc });
    const toNum = S.p(1, 0, 1.0);
    const q = 1 - toNum;
    text(ctx, '？？？', lx, 595, { size: 60, color: C.red, alpha: sc * q });
    text(ctx, '01001011', lx, 525, { size: 34, family: 'mono', color: C.green, alpha: sc * q * 0.6 });
    arrow(ctx, lx, 385, lx, 460, { color: C.dim, progress: S.p(0, 1.2, 0.6) });
    // 变成向量
    if (toNum > 0) {
      const m = math(ctx, M.row('[', '0.55', ',', ' ', '0.55', ']'), lx, 570, { size: 64, alpha: toNum, color: C.orange });
      text(ctx, '词向量 = 一串数字', lx, 760, { size: 40, color: C.orange, alpha: S.p(1, 1.0, 0.8) });
    }
    ctx.restore();

    // 右侧：二维“词地图”
    const cx = 1290, cy = 470, R = 330;
    const pa = S.p(1, 1.6, 0.8);
    if (pa > 0) {
      ctx.save(); ctx.globalAlpha = pa * (1 - 0.75 * S.p(3, 0, 0.8));
      line(ctx, cx - R - 20, cy, cx + R + 20, cy, { color: C.faint, lw: 2 });
      line(ctx, cx, cy - R - 20, cx, cy + R + 20, { color: C.faint, lw: 2 });
      T.arrowHead(ctx, cx + R + 26, cy, 0, 14, C.faint); T.arrowHead(ctx, cx, cy - R - 26, -Math.PI / 2, 14, C.faint);
      text(ctx, '第 1 个数', cx + R - 20, cy + 34, { size: 26, color: C.dim });
      text(ctx, '第 2 个数', cx + 70, cy - R - 10, { size: 26, color: C.dim });
      const words = [
        ['猫', 0.55, 0.55, C.orange, 0], ['狗', 0.75, 0.38, C.orange, 1], ['兔子', 0.38, 0.75, C.orange, 1],
        ['汽车', -0.55, -0.45, C.blue, 2], ['卡车', -0.78, -0.62, C.blue, 2], ['自行车', -0.35, -0.72, C.blue, 2],
        ['苹果', -0.62, 0.6, C.green, 3], ['香蕉', -0.42, 0.8, C.green, 3],
      ];
      const P = (u, v) => [cx + u * R, cy - v * R];
      // 猫 的坐标投影线
      const [mx, my] = P(0.55, 0.55);
      const proj = S.p(1, 2.4, 0.8) * (1 - S.p(2, 0, 0.6));
      line(ctx, mx, my, mx, cy, { color: C.orange, dash: [6, 8], alpha: proj });
      line(ctx, mx, my, cx, my, { color: C.orange, dash: [6, 8], alpha: proj });
      text(ctx, '0.55', mx, cy + 28, { size: 26, family: 'main', color: C.orange, alpha: proj });
      text(ctx, '0.55', cx - 40, my, { size: 26, family: 'main', color: C.orange, alpha: proj });
      // 聚类光晕
      const halo = S.p(2, 1.2, 1);
      [[0.56, 0.56, C.orange], [-0.56, -0.6, C.blue], [-0.52, 0.7, C.green]].forEach(([u, v, col], k) => {
        const [x, y] = P(u, v);
        const g = ctx.createRadialGradient(x, y, 0, x, y, 150);
        g.addColorStop(0, T.rgba(col, 0.28 * halo)); g.addColorStop(1, T.rgba(col, 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 150, 0, Math.PI * 2); ctx.fill();
      });
      words.forEach(([w, u, v, col, grp], i) => {
        const appear = grp === 0 ? S.p(1, 2.0, 0.6) : S.p(2, 0.3 + i * 0.25, 0.6);
        if (appear <= 0) return;
        const [x, y] = P(u, v);
        const fly = easeOut(appear);
        const fx = lerp(cx, x, fly), fy = lerp(cy, y, fly);
        circle(ctx, fx, fy, 10, { fill: col, glow: 14, alpha: appear });
        text(ctx, w, fx, fy - 34, { size: 34, color: col, alpha: appear, weight: 600 });
      });
      ctx.restore();
    }

    // 512 维
    const d = S.p(3, 1.6, 1);
    if (d > 0) {
      ctx.save(); ctx.globalAlpha = d;
      panel(ctx, 560, 150, 800, 640, { fill: 'rgba(12,18,38,0.92)' });
      text(ctx, '论文中：每个词 = 512 个数', 960, 210, { size: 44, color: C.text });
      const r = T.rng(42);
      const vals = Array.from({ length: 512 }, () => T.gauss(r) * 0.5);
      const cols = 32, cs = 20, gap = 3;
      const x0 = 960 - (cols * (cs + gap)) / 2, y0 = 270;
      const rev = prog(t, S.L(3) + 2.0, 2.5);
      for (let i = 0; i < 512; i++) {
        if (i / 512 > rev) break;
        const r_ = Math.floor(i / cols), c_ = i % cols;
        roundRect(ctx, x0 + c_ * (cs + gap), y0 + r_ * (cs + gap), cs, cs, 4, { fill: T.diverge(vals[i]) });
      }
      text(ctx, '“猫”', 960, 700, { size: 40, color: C.orange, alpha: prog(t, S.L(3) + 4, 0.8) });
      text(ctx, '512 维空间里的一个点', 960, 750, { size: 32, color: C.dim, alpha: prog(t, S.L(3) + 4.5, 0.8) });
      ctx.restore();
    }
  };

  // ------------------------------------------------------------------ 02 RNN
  scenes.rnn = function (ctx, S) {
    const t = S.t;
    const n = SENT.length;
    const xs = SENT.map((_, i) => 330 + i * 180);
    const yTok = 640, yCell = 470;
    const toAttn = S.p(3, 0, 1.0);

    SENT.forEach((w, i) => chip(ctx, w, xs[i], yTok, { size: 36, w: 130, alpha: S.p(0, i * 0.08, 0.5) }));

    // RNN 单元按顺序处理：读到第几个词
    const readP = prog(t, S.L(0) + 2.0, S.E(1) - S.L(0) - 2.0);
    const step = readP * n;
    const rnnA = 1 - toAttn;
    if (rnnA > 0) {
      ctx.save(); ctx.globalAlpha = rnnA;
      for (let i = 0; i < n; i++) {
        const lit = clamp(step - i) > 0 ? 1 : 0;
        const active = step > i && step < i + 1;
        roundRect(ctx, xs[i] - 55, yCell - 45, 110, 90, 16, {
          fill: active ? 'rgba(96,165,250,0.35)' : lit ? 'rgba(96,165,250,0.14)' : 'rgba(255,255,255,0.04)',
          stroke: active ? C.blue : C.panelStroke, lw: active ? 3 : 1.5, glow: active ? 20 : 0,
        });
        text(ctx, 'RNN', xs[i], yCell, { size: 26, family: 'main', color: lit ? C.text : C.dim });
        arrow(ctx, xs[i], yTok - 36, xs[i], yCell + 50, { color: C.faint, lw: 2, head: 10 });
        if (i < n - 1) arrow(ctx, xs[i] + 58, yCell, xs[i + 1] - 60, yCell, { color: lit ? C.blue : C.faint, lw: 3, head: 12 });
      }
      // 记忆小球
      if (step > 0 && step < n + 0.5) {
        const k = Math.min(n - 1, Math.floor(step)), f = step - k;
        const bx = k < n - 1 ? lerp(xs[k], xs[k + 1], easeOut(clamp((f - 0.5) * 2))) : xs[k];
        circle(ctx, bx, yCell - 90, 16, { fill: C.Q, glow: 26 });
        text(ctx, '记忆', bx, yCell - 128, { size: 26, color: C.Q });
      }
      // 记忆中各个词的“清晰度”：越早读的越模糊
      const bars = S.p(1, 0, 0.8);
      if (bars > 0) {
        const bx0 = 330, by = 300;
        text(ctx, '记忆里每个词还剩多清楚：', bx0 - 60, 150, { size: 30, color: C.dim, align: 'left', alpha: bars });
        for (let i = 0; i < n; i++) {
          if (step <= i) continue;
          const h = 130 * Math.pow(0.62, Math.max(0, step - i - 1));
          roundRect(ctx, xs[i] - 30, by - h, 60, h, 6, { fill: T.rgba(C.Q, 0.25 + 0.6 * (h / 130)), alpha: bars });
        }
        line(ctx, 260, by, 1650, by, { color: C.faint, alpha: bars });
        text(ctx, '“小猫”快记不清了…', xs[0] + 40, by + 30, { size: 26, color: C.red, align: 'left', alpha: bars * prog(step, 5, 1) });
      }
      // 排队：GPU 核心
      const gq = S.p(2, 0.2, 0.8) * (1 - S.p(3, 0, 0.4));
      if (gq > 0) {
        panel(ctx, 560, 760, 800, 110, { alpha: gq, fill: 'rgba(10,15,30,0.9)' });
        text(ctx, 'GPU 核心：', 600, 815, { size: 30, color: C.dim, align: 'left', alpha: gq });
        const k = Math.floor((t * 2.5) % 8);
        for (let i = 0; i < 8; i++) {
          const on = i === k;
          roundRect(ctx, 780 + i * 66, 790, 50, 50, 8, { fill: on ? C.green : 'rgba(255,255,255,0.06)', alpha: gq, glow: on ? 16 : 0 });
        }
      }
      ctx.restore();
    }

    // 注意力：所有词同时互相连接
    if (toAttn > 0) {
      ctx.save(); ctx.globalAlpha = toAttn;
      let k = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const pp = prog(t, S.L(3) + 1.0 + k * 0.025, 0.8);
        const hue = [C.Q, C.K, C.V, C.blue, C.purple][k % 5];
        arcArrow(ctx, xs[i], yTok - 40, xs[j], yTok - 40, (j - i) * 34, { color: T.rgba(hue, 0.55), lw: 2.5, progress: pp, head: 0 });
        k++;
      }
      text(ctx, '每个词，同时看向所有词', W / 2, 190, { size: 56, color: C.Q, glow: 20, alpha: S.p(3, 2.4, 0.8) });
      const ga = S.p(3, 3.0, 0.8);
      panel(ctx, 560, 760, 800, 110, { alpha: ga, fill: 'rgba(10,15,30,0.9)' });
      text(ctx, 'GPU 核心：', 600, 815, { size: 30, color: C.dim, align: 'left', alpha: ga });
      for (let i = 0; i < 8; i++) roundRect(ctx, 780 + i * 66, 790, 50, 50, 8, { fill: C.green, alpha: ga, glow: 16 });
      ctx.restore();
    }
  };

  // ------------------------------------------------------------------ 03 歧义句
  scenes.ambiguity = function (ctx, S) {
    const t = S.t;
    const words = ['小猫', '没有', '跳上', '桌子', '因为', '它', '太', '累了'];
    const xs = [340, 520, 700, 880, 1080, 1260, 1420, 1580];
    const y = 560;
    const swap = S.p(1, 2.8, 1.0); // 累了 → 高了
    const wA = [0.62, 0.02, 0.05, 0.12, 0.02, 0.03, 0.02, 0.12];
    const wB = [0.12, 0.02, 0.06, 0.6, 0.02, 0.03, 0.01, 0.14];
    const w = wA.map((a, i) => lerp(a, wB[i], swap));
    const arrowsOn = S.p(0, 4.0, 1.0);

    words.forEach((wd, i) => {
      const isIt = i === IT;
      const a = S.p(0, i * 0.12, 0.5);
      let label = wd;
      if (i === 7) label = swap < 0.5 ? '累了' : '高了';
      const flip = i === 7 ? Math.abs(Math.cos(swap * Math.PI)) : 1;
      ctx.save();
      ctx.translate(xs[i], y); ctx.scale(1, flip); ctx.translate(-xs[i], -y);
      chip(ctx, label, xs[i], y, {
        size: 44, w: 130, alpha: a,
        stroke: isIt ? C.Q : i === 7 && swap > 0.5 ? C.red : undefined, glow: isIt ? 24 : 0, color: isIt ? C.Q : C.text,
      });
      ctx.restore();
      if (!isIt && arrowsOn > 0) {
        const wi = w[i];
        arcArrow(ctx, xs[IT], y - 60, xs[i], y - 60, (i < IT ? 1 : -1) * (40 + Math.abs(i - IT) * 34), {
          color: T.rgba(C.Q, 0.25 + 0.75 * wi), lw: 2 + 26 * wi, progress: arrowsOn, head: 10 + 20 * wi,
        });
        const pct = Math.round(wi * 100);
        roundRect(ctx, xs[i] - 26, y + 70, 52, 120 * wi + 2, 6, { fill: T.rgba(C.Q, 0.8), alpha: arrowsOn });
        text(ctx, pct + '%', xs[i], y + 70 + 120 * wi + 28, { size: 28, family: 'main', color: C.dim, alpha: arrowsOn });
      }
    });
    // 问号
    const qa = S.p(0, 2.0, 0.6) * (1 - arrowsOn);
    text(ctx, '它 = ？', xs[IT], 360, { size: 64, color: C.Q, alpha: qa, glow: 20 });
    const ans = swap < 0.5 ? '它 = 小猫' : '它 = 桌子';
    const ansA = S.p(1, 0, 0.6) * (1 - S.p(2, 0, 0.5)) * (swap > 0.05 && swap < 0.95 ? 0.2 : 1);
    text(ctx, ans, W / 2, 210, { size: 64, color: swap < 0.5 ? C.orange : C.blue, alpha: ansA, glow: 20 });
    const last = S.p(2, 1.6, 0.8);
    text(ctx, '注意力 = 回头看整句话 + 给每个词的相关程度打分', W / 2, 210, { size: 48, color: C.text, alpha: last });
  };

  // 简单图标
  function iconSearch(ctx, x, y, s, col) {
    ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = s * 0.12; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.arc(x - s * 0.1, y - s * 0.1, s * 0.32, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + s * 0.13, y + s * 0.13); ctx.lineTo(x + s * 0.42, y + s * 0.42); ctx.stroke(); ctx.restore();
  }
  function iconTag(ctx, x, y, s, col) {
    ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = s * 0.1; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(x - s * 0.45, y - s * 0.25); ctx.lineTo(x + s * 0.2, y - s * 0.25); ctx.lineTo(x + s * 0.45, y);
    ctx.lineTo(x + s * 0.2, y + s * 0.25); ctx.lineTo(x - s * 0.45, y + s * 0.25); ctx.closePath(); ctx.stroke();
    ctx.beginPath(); ctx.arc(x + s * 0.12, y, s * 0.06, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
  }
  function iconBox(ctx, x, y, s, col) {
    ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = s * 0.1; ctx.lineJoin = 'round';
    ctx.strokeRect(x - s * 0.4, y - s * 0.2, s * 0.8, s * 0.55);
    ctx.beginPath(); ctx.moveTo(x - s * 0.45, y - s * 0.2); ctx.lineTo(x - s * 0.3, y - s * 0.42); ctx.lineTo(x + s * 0.3, y - s * 0.42); ctx.lineTo(x + s * 0.45, y - s * 0.2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - s * 0.12, y); ctx.lineTo(x + s * 0.12, y); ctx.stroke(); ctx.restore();
  }
  window.ICONS = { iconSearch, iconTag, iconBox };

  // ------------------------------------------------------------------ 04 Q K V
  scenes.qkv = function (ctx, S) {
    const t = S.t;
    // 三张角色卡（第 0 句）
    const cardsA = S.p(0, 0, 0.6) * (1 - S.p(1, 0, 0.6));
    const cardsTop = S.p(4, 0, 0.6);
    if (cardsA > 0) {
      const defs = [['Query 查询', '我在找什么？', C.Q, iconSearch], ['Key 键', '我有什么标签？', C.K, iconTag], ['Value 值', '我的实际内容', C.V, iconBox]];
      defs.forEach(([name, desc, col, icon], i) => {
        const x = 560 + i * 400, a = S.p(0, 0.9 + i * 0.7, 0.6) * cardsA / Math.max(cardsA, 1e-6) * cardsA;
        ctx.save(); ctx.globalAlpha = a;
        ctx.translate(x, 470); const sc = 0.85 + 0.15 * easeBack(S.p(0, 0.9 + i * 0.7, 0.6)); ctx.scale(sc, sc);
        roundRect(ctx, -170, -200, 340, 400, 28, { fill: T.rgba(col, 0.08), stroke: col, lw: 3, glow: 16 });
        icon(ctx, 0, -80, 130, col);
        text(ctx, name, 0, 60, { size: 46, color: col, weight: 700 });
        text(ctx, desc, 0, 130, { size: 32, color: C.dim });
        ctx.restore();
      });
    }

    // 图书馆（第 1、2 句）
    const libA = S.p(1, 0.3, 0.7) * (1 - S.p(3, 0, 0.6));
    if (libA > 0) {
      ctx.save(); ctx.globalAlpha = libA;
      // 提问气泡
      const qx = 330, qy = 380;
      roundRect(ctx, qx - 210, qy - 80, 420, 160, 30, { fill: T.rgba(C.Q, 0.1), stroke: C.Q, lw: 3, glow: 14 });
      text(ctx, '怎么让机器', qx, qy - 22, { size: 38, color: C.Q });
      text(ctx, '读懂句子？', qx, qy + 26, { size: 38, color: C.Q });
      text(ctx, 'Query', qx, qy - 120, { size: 32, family: 'main', color: C.Q, alpha: S.p(1, 1.6, 0.6) });
      // 书架
      const books = [['烹饪大全', 0.02], ['深度学习', 0.52], ['唐诗三百首', 0.01], ['注意力机制', 0.43], ['世界地理', 0.02]];
      const contents = ['红烧肉…', '神经网络…', '床前明月光…', 'Q·K·V…', '七大洲…'];
      const bx0 = 830;
      const match = S.p(2, 0.3, 1.2);
      books.forEach(([name, wgt], i) => {
        const x = bx0 + i * 190, y = 380;
        const ba = S.p(1, 1.0 + i * 0.15, 0.5);
        const hl = match * wgt;
        roundRect(ctx, x - 60, y - 190, 120, 380, 10, { fill: `rgba(${80 + i * 20},${60 + i * 10},${120 - i * 10},0.55)`, stroke: hl > 0.2 ? C.K : 'rgba(255,255,255,0.25)', lw: hl > 0.2 ? 4 : 2, alpha: ba, glow: hl > 0.2 ? 20 : 0 });
        // 书脊标签 = Key
        roundRect(ctx, x - 48, y - 140, 96, 220, 8, { fill: T.rgba(C.K, 0.16), stroke: C.K, lw: 2, alpha: ba * S.p(1, 3.0, 0.6) + ba * 0.2 });
        ctx.save(); ctx.globalAlpha *= ba;
        Array.from(name).forEach((ch, k, arr) => text(ctx, ch, x, y - 30 + (k - (arr.length - 1) / 2) * 40, { size: 32, color: C.text }));
        ctx.restore();
        // 匹配度连线
        if (match > 0) {
          arcArrow(ctx, qx + 210, qy, x, y - 200, -80, { color: T.rgba(C.Q, 0.2 + 0.8 * wgt * 1.6), lw: 2 + 14 * wgt, progress: match, head: 8 + 10 * wgt });
          text(ctx, Math.round(wgt * 100) + '%', x, y + 225, { size: 32, family: 'main', color: wgt > 0.2 ? C.Q : C.dim, alpha: match });
        }
      });
      text(ctx, 'Key：书脊上的标签', bx0 + 380, 120, { size: 34, color: C.K, alpha: S.p(1, 3.0, 0.6) * (1 - match) });
      // 笔记 = Value 混合
      const note = S.p(2, 2.6, 0.8);
      if (note > 0) {
        roundRect(ctx, 130, 560, 400, 230, 20, { fill: T.rgba(C.V, 0.1), stroke: C.V, lw: 3, alpha: note });
        text(ctx, '我的笔记（Value 混合）', 330, 600, { size: 30, color: C.V, alpha: note });
        text(ctx, '52% 神经网络…', 330, 660, { size: 32, color: C.text, alpha: note });
        text(ctx, '43% Q·K·V…', 330, 705, { size: 32, color: C.text, alpha: note });
        text(ctx, '5% 其他', 330, 750, { size: 28, color: C.dim, alpha: note });
      }
      ctx.restore();
    }

    // 句子里的 Q K V（第 3 句）
    const sentA = S.p(3, 0.2, 0.7) * (1 - S.p(4, 0, 0.6));
    if (sentA > 0) {
      ctx.save(); ctx.globalAlpha = sentA;
      const xs = SENT.map((_, i) => 290 + i * 190);
      const y = 300;
      SENT.forEach((w, i) => chip(ctx, w, xs[i], y, { size: 38, w: 130, stroke: i === IT ? C.Q : i === 0 ? C.orange : undefined, color: i === IT ? C.Q : C.text }));
      // 它 的 Query 气泡
      const qa = S.p(3, 1.6, 0.6);
      roundRect(ctx, xs[IT] - 230, 420, 460, 120, 26, { fill: T.rgba(C.Q, 0.12), stroke: C.Q, lw: 3, alpha: qa, glow: 12 });
      text(ctx, 'Q：我指的是哪个名词？', xs[IT], 480, { size: 36, color: C.Q, alpha: qa });
      line(ctx, xs[IT], 345, xs[IT], 420, { color: C.Q, lw: 3, alpha: qa });
      // 小猫 的 Key 标签
      const ka = S.p(3, 5.0, 0.6);
      roundRect(ctx, xs[0] - 150, 420, 420, 120, 26, { fill: T.rgba(C.K, 0.12), stroke: C.K, lw: 3, alpha: ka, glow: 12 });
      text(ctx, 'K：我是会累的动物', xs[0] + 60, 480, { size: 36, color: C.K, alpha: ka });
      line(ctx, xs[0], 345, xs[0], 420, { color: C.K, lw: 3, alpha: ka });
      const ka2 = S.p(3, 6.0, 0.6);
      roundRect(ctx, xs[3] - 170, 640, 340, 100, 22, { fill: T.rgba(C.K, 0.08), stroke: T.rgba(C.K, 0.6), lw: 2, alpha: ka2 });
      text(ctx, 'K：我是高高的家具', xs[3], 690, { size: 30, color: C.K, alpha: ka2 });
      line(ctx, xs[3], 345, xs[3], 640, { color: T.rgba(C.K, 0.6), lw: 2, alpha: ka2, dash: [6, 6] });
      const match = S.p(3, 7.0, 0.8);
      arcArrow(ctx, xs[IT] - 230, 500, xs[0] + 270, 500, 50, { color: C.Q, lw: 5, progress: match, glow: 12 });
      text(ctx, '匹配！', (xs[0] + 270 + xs[IT] - 230) / 2 - 70, 590, { size: 40, color: C.Q, alpha: match, glow: 14 });
      ctx.restore();
    }

    // 矩阵乘法（第 4 句）
    if (cardsTop > 0) {
      ctx.save(); ctx.globalAlpha = cardsTop;
      const r = T.rng(11);
      const x = [0.9, -0.3, 0.5, 0.2];
      const mk = () => Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => Math.round(T.gauss(r) * 6) / 10));
      const Ws = [mk(), mk(), mk()];
      const names = [['W_Q', 'q', C.Q], ['W_K', 'k', C.K], ['W_V', 'v', C.V]];
      text(ctx, '「它」的词向量 x', 330, 150, { size: 32, color: C.dim });
      names.forEach(([wn, on, col], i) => {
        const y = 270 + i * 210;
        const a = S.p(4, 0.6 + i * 0.6, 0.6);
        ctx.save(); ctx.globalAlpha *= a;
        vecCells(ctx, x, 330, y, { cell: 62, nums: true, digits: 1, numSize: 24 });
        text(ctx, '×', 520, y, { size: 56, color: C.dim, family: 'main' });
        const g = grid(ctx, Ws[i], 740, y, { cell: 38, nums: true, digits: 1, numSize: 15, range: 1.2 });
        T.brackets(ctx, g.x0 - 10, g.y0 - 6, g.w + 20, g.h + 12, { color: col });
        math(ctx, M.sub('W', wn.split('_')[1]), 740 + g.w / 2 + 60, y, { size: 44, color: col });
        text(ctx, '=', 1040, y, { size: 56, color: C.dim, family: 'main' });
        const res = Ws[i][0].map((_, j) => x.reduce((s, xi, k) => s + xi * Ws[i][k][j], 0));
        const ra = S.p(4, 1.6 + i * 0.6, 0.8);
        vecCells(ctx, res, 1260, y, { cell: 62, nums: true, digits: 1, numSize: 24, color: col, range: 1.5, reveal: ra });
        math(ctx, on, 1460, y, { size: 52, color: col, alpha: ra });
        ctx.restore();
      });
      const lab = S.p(4, 4.5, 0.8);
      panel(ctx, 1520, 360, 340, 220, { alpha: lab, stroke: C.green });
      text(ctx, '矩阵里的数字', 1690, 420, { size: 34, color: C.green, alpha: lab });
      text(ctx, '一开始是随机的', 1690, 470, { size: 30, color: C.dim, alpha: lab });
      text(ctx, '训练中自己学出来', 1690, 520, { size: 30, color: C.dim, alpha: lab });
      ctx.restore();
    }
  };

  // ------------------------------------------------------------------ 05 点积
  scenes.dot = function (ctx, S) {
    const t = S.t;
    const ox = 560, oy = 520, u = 105;
    const P = (x, y) => [ox + x * u, oy - y * u];
    const planeA = S.p(0, 0.6, 0.8);
    ctx.save(); ctx.globalAlpha = planeA;
    for (let i = -4; i <= 4; i++) {
      line(ctx, ox + i * u, oy - 3.6 * u, ox + i * u, oy + 3.6 * u, { color: 'rgba(120,140,200,0.12)', lw: 1 });
      if (Math.abs(i) <= 3) line(ctx, ox - 4.2 * u, oy + i * u, ox + 4.2 * u, oy + i * u, { color: 'rgba(120,140,200,0.12)', lw: 1 });
    }
    line(ctx, ox - 4.3 * u, oy, ox + 4.3 * u, oy, { color: C.dim, lw: 2 });
    line(ctx, ox, oy + 3.7 * u, ox, oy - 3.7 * u, { color: C.dim, lw: 2 });
    ctx.restore();

    const a = [2, 1];
    const bLen = Math.hypot(1, 3), bAng0 = Math.atan2(3, 1);
    // 第 3 句：b 绕原点旋转一整圈
    const rot = prog(t, S.L(3) + 0.3, (S.E(3) - S.L(3)) + 0.6);
    const bAng = bAng0 + ease(rot) * Math.PI * 2;
    const b = rot > 0 ? [bLen * Math.cos(bAng), bLen * Math.sin(bAng)] : [1, 3];
    const va = S.p(1, 0, 0.8), vb = S.p(1, 0.5, 0.8);
    const [ax, ay] = P(...a), [bx, by] = P(...b);
    arrow(ctx, ox, oy, ax, ay, { color: C.Q, lw: 6, head: 24, progress: va, glow: 10 });
    arrow(ctx, ox, oy, bx, by, { color: C.K, lw: 6, head: 24, progress: vb, glow: 10 });
    math(ctx, M.row('a', '=', '(2, 1)'), ax + 90, ay + 10, { size: 40, color: C.Q, alpha: va * (1 - 0.6 * prog(rot, 0, 0.05)) });
    math(ctx, rot > 0 ? 'b' : M.row('b', '=', '(1, 3)'), bx + (rot > 0 ? 30 * Math.cos(bAng) : 90), by - (rot > 0 ? 30 * Math.sin(bAng) : 10), { size: 40, color: C.K, alpha: vb });

    // 夹角
    const angA = S.p(2, 0, 0.8);
    if (angA > 0) {
      const a1 = -Math.atan2(a[1], a[0]), a2 = -bAng;
      ctx.save(); ctx.globalAlpha = angA;
      ctx.strokeStyle = C.text; ctx.lineWidth = 3;
      ctx.beginPath();
      let d = a2 - a1; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
      ctx.arc(ox, oy, 70, a1, a1 + d, d < 0);
      ctx.stroke();
      const mid = a1 + d / 2;
      math(ctx, 'θ', ox + Math.cos(mid) * 105, oy + Math.sin(mid) * 105, { size: 40, color: C.text });
      ctx.restore();
    }

    // 右侧：计算
    const rx = 1390;
    text(ctx, '向量点积', rx, 180, { size: 52, color: C.text, alpha: S.p(0, 0.2, 0.6) });
    const s1 = S.p(1, 0.8, 0.6), s2 = S.p(1, 2.0, 0.6), s3 = S.p(1, 3.4, 0.6), s4 = S.p(1, 4.6, 0.6);
    const fade1 = 1 - 0.55 * S.p(2, 0, 0.6);
    math(ctx, M.row(M.id('a', 'a'), '·', M.id('b', 'b'), '=', M.id('t1', M.row('2', '×', '1')), '+', M.id('t2', M.row('1', '×', '3'))), rx, 300, {
      size: 58, alpha: s1 * fade1,
      style: (id) => ({ a: { color: C.Q }, b: { color: C.K }, t1: { alpha: s2 }, t2: { alpha: s3 } }[id]),
    });
    math(ctx, M.row('=', '5'), rx + 40, 390, { size: 64, color: C.green, alpha: s4 * fade1, glow: 12 });
    text(ctx, '对应位置相乘，再相加', rx, 470, { size: 34, color: C.dim, alpha: s2 * fade1 });

    const g1 = S.p(2, 0.4, 0.8), g2 = S.p(2, 2.4, 0.8);
    math(ctx, M.row(M.id('a', 'a'), '·', M.id('b', 'b'), '=', '|', 'a', '|', ' ', '|', 'b', '|', ' ', '\\cos', ' ', 'θ'), rx, 570, {
      size: 62, alpha: g1, style: (id) => ({ a: { color: C.Q }, b: { color: C.K } }[id]),
    });
    math(ctx, M.row('=', M.sqrt('5'), '×', M.sqrt('10'), '×', '\\cos', ' ', '45°', '=', '5', ' ', '✓'), rx, 670, { size: 46, color: C.dim, alpha: g2 * (1 - S.p(3, 0, 0.5)) });

    // 实时读数
    const live = S.p(3, 0, 0.6);
    if (live > 0) {
      const val = T.dot(a, b);
      const cos = val / (Math.hypot(...a) * bLen);
      ctx.save(); ctx.globalAlpha = live;
      math(ctx, M.row('a', '·', 'b', '=', T.fmtNum(val, 1)), rx, 680, { size: 64, color: val > 0.3 ? C.green : val < -0.3 ? C.red : C.text, glow: 10 });
      // 计量条
      const bw = 560, bx0 = rx - bw / 2, by0 = 770;
      roundRect(ctx, bx0, by0, bw, 26, 13, { fill: 'rgba(255,255,255,0.06)' });
      const mid = bx0 + bw / 2, vmax = Math.hypot(...a) * bLen;
      const vx = mid + (val / vmax) * (bw / 2);
      roundRect(ctx, Math.min(mid, vx), by0, Math.abs(vx - mid), 26, 13, { fill: val >= 0 ? C.green : C.red });
      line(ctx, mid, by0 - 8, mid, by0 + 34, { color: C.dim, lw: 2 });
      const tag = cos > 0.9 ? '方向一致 → 很大' : cos < -0.9 ? '方向相反 → 负数' : Math.abs(cos) < 0.12 ? '互相垂直 → 0' : '';
      text(ctx, tag, rx, 840, { size: 38, color: cos > 0.9 ? C.green : cos < -0.9 ? C.red : C.text });
      ctx.restore();
    }
  };

  window.SCENES = Object.assign(window.SCENES || {}, scenes);
})();
