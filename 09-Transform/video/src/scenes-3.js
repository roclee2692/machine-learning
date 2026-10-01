/*
 * 分镜（下）：编码器 → 解码器 → 完整结构 + 训练结果 → 尾声
 */
(function () {
  const T = window.TV;
  const { C, W, H, text, chip, arrow, arcArrow, line, circle, roundRect, vecCells, grid, math, M, panel } = T;
  const { clamp, lerp, ease, easeOut, easeBack, prog } = T;
  const scenes = {};

  const BLOCK = { attn: '#fb923c', norm: '#facc15', ffn: '#60a5fa', emb: '#f472b6', lin: '#a78bfa', soft: '#34d399' };

  function block(ctx, x, y, w, h, label, col, o = {}) {
    const a = o.alpha == null ? 1 : o.alpha;
    if (a <= 0) return;
    ctx.save();
    ctx.globalAlpha *= a;
    ctx.translate(x, y);
    const s = o.pop == null ? 1 : 0.85 + 0.15 * easeBack(o.pop);
    ctx.scale(s, s);
    roundRect(ctx, -w / 2, -h / 2, w, h, Math.min(14, h / 3), { fill: T.rgba(col, o.lit ? 0.42 : 0.2), stroke: col, lw: o.lit ? 4 : 2.5, glow: o.lit ? 26 : 0 });
    text(ctx, label, 0, 2, { size: o.size || Math.min(34, h * 0.5), color: C.text, weight: 600 });
    ctx.restore();
  }

  // ------------------------------------------------------------------ 11 编码器
  scenes.encoder = function (ctx, S) {
    const t = S.t;
    const stack = S.p(4, 0.2, 1.0);
    const L1 = 1 - stack;
    const bx = 620, bw = 440;
    const lit1 = S.p(1, 0, 0.5) * (1 - S.p(2, 0, 0.5));
    const lit2 = S.p(2, 0, 0.5) * (1 - S.p(3, 0, 0.5));
    const lit3 = S.p(3, 0, 0.5) * (1 - S.p(4, 0, 0.5));
    if (L1 > 0) {
      ctx.save(); ctx.globalAlpha = L1;
      const appear = (k) => S.p(0, 0.3 + k * 0.35, 0.6);
      // 输入词向量
      ['我', '爱', '学', '习'].forEach((w, i) => {
        const x = bx - 150 + i * 100;
        vecCells(ctx, [0.6, -0.3, 0.8, 0.1, -0.5].map((v) => v * Math.cos(i + v)), x, 790, { cell: 20, gap: 3, vertical: true, alpha: appear(0) });
        text(ctx, w, x, 870, { size: 26, color: C.dim, alpha: appear(0) });
      });
      arrow(ctx, bx, 720, bx, 660, { color: C.dim, lw: 3, alpha: appear(1) });
      block(ctx, bx, 610, bw, 90, '多头自注意力', BLOCK.attn, { alpha: appear(1), pop: appear(1), lit: lit1 > 0.5 });
      arrow(ctx, bx, 565, bx, 525, { color: C.dim, lw: 3, alpha: appear(2) });
      block(ctx, bx, 495, bw, 50, '相加 & 层归一化', BLOCK.norm, { alpha: appear(2), pop: appear(2), lit: lit3 > 0.5 });
      arrow(ctx, bx, 470, bx, 430, { color: C.dim, lw: 3, alpha: appear(3) });
      block(ctx, bx, 385, bw, 90, '前馈网络', BLOCK.ffn, { alpha: appear(3), pop: appear(3), lit: lit2 > 0.5 });
      arrow(ctx, bx, 340, bx, 300, { color: C.dim, lw: 3, alpha: appear(4) });
      block(ctx, bx, 270, bw, 50, '相加 & 层归一化', BLOCK.norm, { alpha: appear(4), pop: appear(4), lit: lit3 > 0.5 });
      arrow(ctx, bx, 245, bx, 190, { color: C.dim, lw: 3, alpha: appear(4) });
      text(ctx, '输出（同样 4 个 512 维向量）', bx, 165, { size: 26, color: C.dim, alpha: appear(4) });
      // 残差连接
      const rc = lit3 > 0.5 ? C.green : T.rgba(C.green, 0.5);
      const ra = Math.max(appear(4) * 0.6, lit3);
      [[690, 495], [450, 270]].forEach(([y0, y1]) => {
        ctx.save(); ctx.globalAlpha *= ra;
        ctx.strokeStyle = rc; ctx.lineWidth = lit3 > 0.5 ? 6 : 3; if (lit3 > 0.5) { ctx.shadowColor = C.green; ctx.shadowBlur = 16; }
        ctx.beginPath(); ctx.moveTo(bx, y0); ctx.lineTo(bx - bw / 2 - 50, y0); ctx.lineTo(bx - bw / 2 - 50, y1); ctx.lineTo(bx - bw / 2 - 8, y1); ctx.stroke();
        T.arrowHead(ctx, bx - bw / 2 - 2, y1, 0, 14, rc);
        ctx.restore();
        // 流动的光点
        if (lit3 > 0.5) {
          const f = (t * 0.8) % 1;
          const segs = [[bx, y0, bx - bw / 2 - 50, y0], [bx - bw / 2 - 50, y0, bx - bw / 2 - 50, y1], [bx - bw / 2 - 50, y1, bx - bw / 2, y1]];
          const k = Math.min(2, Math.floor(f * 3)), g = f * 3 - k;
          const [a, b, c, d] = segs[k];
          circle(ctx, lerp(a, c, g), lerp(b, d, g), 9, { fill: C.green, glow: 20 });
        }
      });
      text(ctx, '残差', bx - bw / 2 - 50, 585, { size: 26, color: C.green, alpha: ra, align: 'right' });

      // 右侧说明
      const rx = 1350;
      if (lit1 > 0) {
        ctx.save(); ctx.globalAlpha *= lit1;
        panel(ctx, 980, 200, 740, 560);
        text(ctx, '词与词交流信息', rx, 260, { size: 40, color: BLOCK.attn });
        const xs = [1100, 1270, 1440, 1610];
        ['我', '爱', '学', '习'].forEach((w, i) => chip(ctx, w, xs[i], 560, { size: 40, w: 100 }));
        let k = 0;
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
          if (i === j) continue;
          const pp = prog(t, S.L(1) + 0.4 + k * 0.06, 0.6);
          arcArrow(ctx, xs[i], 520, xs[j], 520, (j > i ? -1 : 1) * (40 + 25 * Math.abs(i - j)), { color: T.rgba(BLOCK.attn, 0.55), lw: 2.5, progress: pp, head: 10 });
          k++;
        }
        ctx.restore();
      }
      if (lit2 > 0) {
        ctx.save(); ctx.globalAlpha *= lit2;
        panel(ctx, 980, 170, 740, 620);
        text(ctx, '每个词各自“消化”', rx, 220, { size: 40, color: BLOCK.ffn });
        const layers = [[4, 1060, '512'], [10, 1260, '2048'], [4, 1460, '512']];
        const pts = layers.map(([n, x]) => Array.from({ length: n }, (_, i) => [x, 470 + (i - (n - 1) / 2) * 42]));
        const fp = S.p(2, 0.6, 1.5);
        for (let l = 0; l < 2; l++) pts[l].forEach(([x1, y1]) => pts[l + 1].forEach(([x2, y2]) => line(ctx, x1, y1, x2, y2, { color: T.rgba(BLOCK.ffn, 0.25), lw: 1, progress: clamp(fp * 2 - l) })));
        pts.forEach((ps, l) => ps.forEach(([x, y], i) => circle(ctx, x, y, 12, { fill: l === 1 && (i * 7) % 3 === 0 ? C.faint : BLOCK.ffn, alpha: clamp(fp * 2 - l + 0.5) })));
        layers.forEach(([n, x, lab]) => text(ctx, lab + ' 维', x, 720, { size: 28, family: 'main', color: C.dim }));
        // ReLU
        const gx = 1600, gy = 520;
        line(ctx, gx - 70, gy, gx + 70, gy, { color: C.dim, lw: 2 });
        line(ctx, gx, gy + 60, gx, gy - 80, { color: C.dim, lw: 2 });
        ctx.save(); ctx.strokeStyle = C.Q; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(gx - 70, gy); ctx.lineTo(gx, gy); ctx.lineTo(gx + 65, gy - 70); ctx.stroke(); ctx.restore();
        text(ctx, 'ReLU', gx, gy - 110, { size: 30, family: 'main', color: C.Q });
        math(ctx, M.row('\\max', M.paren('0', ',', 'x')), gx, gy + 100, { size: 32, color: C.Q });
        ctx.restore();
      }
      if (lit3 > 0) {
        ctx.save(); ctx.globalAlpha *= lit3;
        panel(ctx, 980, 200, 740, 560);
        text(ctx, '残差连接：给信息修一条“高速公路”', rx, 270, { size: 36, color: C.green });
        math(ctx, M.row('输出', '=', '\\LayerNorm', M.paren(M.id('x', 'x'), '+', '子层', M.paren('x'))), rx, 420, { size: 46, style: (id) => (id === 'x' ? { color: C.green } : null) });
        text(ctx, '即使子层什么都没学会，', rx, 540, { size: 32, color: C.dim });
        text(ctx, '原来的信息 x 也能原样传上去', rx, 590, { size: 32, color: C.dim });
        text(ctx, '层归一化：把数值拉回稳定范围', rx, 680, { size: 30, color: BLOCK.norm });
        ctx.restore();
      }
      ctx.restore();
    }

    // 叠 6 层
    if (stack > 0) {
      ctx.save(); ctx.globalAlpha = stack;
      const n = 6, x = 960;
      for (let k = 0; k < n; k++) {
        const y = 780 - k * 100;
        const a = S.p(4, 0.6 + k * 0.25, 0.5);
        roundRect(ctx, x - 300, y - 38, 600, 76, 14, { fill: 'rgba(251,146,60,0.12)', stroke: BLOCK.attn, lw: 2, alpha: a });
        text(ctx, '编码器层 ' + (k + 1), x - 140, y, { size: 30, color: C.text, alpha: a });
        roundRect(ctx, x + 20, y - 22, 120, 44, 8, { fill: T.rgba(BLOCK.attn, 0.4), alpha: a });
        roundRect(ctx, x + 150, y - 22, 120, 44, 8, { fill: T.rgba(BLOCK.ffn, 0.4), alpha: a });
      }
      // 流动的向量，越往上颜色越丰富
      const f = (t * 0.35) % 1;
      for (let j = 0; j < 4; j++) {
        const y = lerp(860, 180, f);
        const sat = clamp((860 - y) / 680);
        circle(ctx, x - 470 + j * 50, y, 14, { fill: T.mixColor('#4b5563', ['#ffc857', '#4fd1c5', '#f472b6', '#a78bfa'][j], sat), glow: 16 * sat });
      }
      text(ctx, '越往上，越懂上下文', x - 400, 140, { size: 30, color: C.dim });
      text(ctx, '× 6', x + 420, 530, { size: 96, family: 'main', color: BLOCK.attn, glow: 20, alpha: S.p(4, 2.2, 0.6) });
      ctx.restore();
    }
  };

  // ------------------------------------------------------------------ 12 解码器
  scenes.decoder = function (ctx, S) {
    const t = S.t;
    const part1 = 1 - S.p(2, 0, 0.6);
    if (part1 > 0) {
      ctx.save(); ctx.globalAlpha = part1;
      // 编码器
      const ex = 520, dx = 1300;
      roundRect(ctx, ex - 230, 330, 460, 260, 24, { fill: 'rgba(251,146,60,0.1)', stroke: BLOCK.attn, lw: 3, alpha: S.p(0, 0.2, 0.6) });
      text(ctx, '编码器', ex, 400, { size: 48, color: BLOCK.attn, alpha: S.p(0, 0.2, 0.6) });
      text(ctx, '读懂原句', ex, 470, { size: 32, color: C.dim, alpha: S.p(0, 0.2, 0.6) });
      ['我', '喜欢', '猫'].forEach((w, i) => {
        chip(ctx, w, ex - 140 + i * 140, 720, { size: 40, w: 120, alpha: S.p(0, 0.6 + i * 0.15, 0.5), stroke: C.orange });
        arrow(ctx, ex - 140 + i * 140, 680, ex - 140 + i * 140, 600, { color: C.dim, lw: 3, alpha: S.p(0, 0.8, 0.5) });
      });
      roundRect(ctx, dx - 260, 330, 520, 260, 24, { fill: 'rgba(96,165,250,0.1)', stroke: C.blue, lw: 3, alpha: S.p(0, 1.6, 0.6) });
      text(ctx, '解码器', dx, 400, { size: 48, color: C.blue, alpha: S.p(0, 1.6, 0.6) });
      text(ctx, '一个词一个词地写', dx, 470, { size: 32, color: C.dim, alpha: S.p(0, 1.6, 0.6) });
      arrow(ctx, ex + 235, 460, dx - 265, 460, { color: BLOCK.attn, lw: 5, progress: S.p(0, 2.4, 0.8), glow: 10 });
      // 逐词生成
      const outs = ['I', 'like', 'cats', '<结束>'];
      const ins = ['<开始>', 'I', 'like', 'cats'];
      const stepT = (k) => S.at(1, 0.25 + k * 0.22);
      for (let k = 0; k < 4; k++) {
        const x = dx - 195 + k * 130;
        const inA = k === 0 ? S.p(1, 0, 0.6) : prog(t, stepT(k - 1) + 0.6, 0.6);
        chip(ctx, ins[k], x, 720, { size: k === 0 ? 24 : 34, w: 118, alpha: inA, stroke: C.blue });
        arrow(ctx, x, 680, x, 600, { color: C.dim, lw: 3, alpha: inA });
        const oA = prog(t, stepT(k), 0.5);
        chip(ctx, outs[k], x, 210, { size: k === 3 ? 24 : 38, w: 118, alpha: oA, stroke: C.Q, color: C.Q, glow: 16 * T.pulse(t, stepT(k), 1) });
        arrow(ctx, x, 325, x, 255, { color: C.Q, lw: 3, alpha: oA });
        // 输出回到下一步的输入
        if (k < 3) {
          const fb = prog(t, stepT(k) + 0.1, 0.6);
          if (fb > 0 && fb < 1) arcArrow(ctx, x + 50, 210, x + 130 + 50, 720, -160, { color: T.rgba(C.Q, 0.6), lw: 3, progress: fb, dash: [8, 8] });
        }
      }
      ctx.restore();
    }

    // 遮罩矩阵
    const maskA = S.p(2, 0.2, 0.7) * (1 - S.p(4, 0, 0.6));
    if (maskA > 0) {
      ctx.save(); ctx.globalAlpha = maskA;
      const labs = ['<开始>', 'I', 'like', 'cats'];
      const cells = labs.map((_, i) => labs.map((_, j) => (j <= i ? 1 : 0)));
      const g = grid(ctx, cells, 520, 500, {
        cell: 110, gap: 6, reveal: S.p(2, 0.6, 1.5),
        fill: (v) => (v ? 'rgba(52,211,153,0.35)' : 'rgba(248,113,113,0.18)'),
        nums: (v) => (v ? '✓' : '−∞'), numSize: 40,
      });
      labs.forEach((l, i) => {
        text(ctx, l, g.x0 - 70, g.y0 + i * 116 + 55, { size: 26, family: 'main', color: C.text });
        text(ctx, l, g.x0 + i * 116 + 55, g.y0 - 28, { size: 26, family: 'main', color: C.text });
      });
      text(ctx, '正在写的位置 ↓        能看的位置 →', 520, 180, { size: 28, color: C.dim });
      // 一行的计算
      const ra = S.p(3, 0.4, 0.6);
      if (ra > 0) {
        roundRect(ctx, g.x0 - 8, g.y0 + 116 - 8, g.w + 16, 126, 10, { stroke: C.Q, lw: 4, alpha: ra, glow: 12 });
        const rx = 1340;
        const rows = [
          ['分数', ['1.2', '2.0', '−∞', '−∞'], C.text, 0.4],
          ['e^分数', ['3.3', '7.4', '0', '0'], C.green, 2.4],
          ['softmax', ['31%', '69%', '0%', '0%'], C.Q, 4.4],
        ];
        rows.forEach(([lab, vals, col, at], r) => {
          const a = S.p(3, at, 0.6);
          const y = 330 + r * 150;
          text(ctx, lab, rx - 330, y, { size: 30, color: col, alpha: a, align: 'right' });
          vals.forEach((v, j) => {
            roundRect(ctx, rx - 260 + j * 150, y - 45, 130, 90, 12, { fill: j >= 2 ? 'rgba(248,113,113,0.15)' : T.rgba(col === C.text ? '#60a5fa' : col, 0.2), alpha: a });
            text(ctx, v, rx - 195 + j * 150, y, { size: 36, family: 'main', color: j >= 2 && r > 0 ? C.red : col, alpha: a });
          });
        });
        math(ctx, M.row(M.sup('e', '−∞'), '=', '0'), rx + 50, 780, { size: 52, color: C.red, alpha: S.p(3, 3.2, 0.6), glow: 12 });
      }
      ctx.restore();
    }

    // 交叉注意力
    const crossA = S.p(4, 0.2, 0.7) * (1 - S.p(5, 0, 0.6));
    if (crossA > 0) {
      ctx.save(); ctx.globalAlpha = crossA;
      const src = ['我', '喜欢', '猫'], w = [0.08, 0.84, 0.08];
      text(ctx, '编码器（原文）：提供 Key 和 Value', 600, 690, { size: 32, color: C.K });
      text(ctx, '解码器：提供 Query', 1320, 200, { size: 32, color: C.Q });
      chip(ctx, '<开始> I ▢', 1320, 300, { size: 40, w: 340, stroke: C.Q });
      text(ctx, '下一个词该写什么？', 1320, 380, { size: 30, color: C.Q });
      src.forEach((s, i) => {
        const x = 380 + i * 220;
        chip(ctx, s, x, 600, { size: 44, w: 160, stroke: C.K, glow: i === 1 ? 20 * S.p(4, 3, 0.8) : 0 });
        arcArrow(ctx, 1180, 320, x, 550, 80, { color: T.rgba(C.Q, 0.25 + 0.75 * w[i]), lw: 2 + 20 * w[i], progress: S.p(4, 1.2 + i * 0.2, 1.0), head: 10 + 14 * w[i] });
        text(ctx, Math.round(w[i] * 100) + '%', x, 500, { size: 32, family: 'main', color: C.Q, alpha: S.p(4, 2.4, 0.6) });
      });
      text(ctx, '→ 重点看「喜欢」，于是写出 like', 1320, 560, { size: 34, color: C.text, alpha: S.p(4, 3.6, 0.8) });
      ctx.restore();
    }

    // 词表概率
    const vA = S.p(5, 0.2, 0.7);
    if (vA > 0) {
      ctx.save(); ctx.globalAlpha = vA;
      text(ctx, '整个词表上的概率（下一个词）', 960, 190, { size: 40, color: C.text });
      const vocab = [['cats', 0.81], ['cat', 0.07], ['dogs', 0.04], ['them', 0.03], ['it', 0.02], ['fish', 0.01], ['the', 0.01], ['…', 0.01]];
      vocab.forEach(([wd, p], i) => {
        const x = 420 + i * 155;
        const h = 480 * p * S.p(5, 0.6 + i * 0.08, 0.8);
        const best = i === 0;
        roundRect(ctx, x - 50, 760 - h, 100, Math.max(2, h), 8, { fill: best ? C.green : T.rgba(C.blue, 0.6), glow: best ? 20 : 0 });
        text(ctx, wd, x, 800, { size: 32, family: 'main', color: best ? C.green : C.text });
        text(ctx, Math.round(p * 100) + '%', x, 760 - h - 24, { size: 26, family: 'main', color: C.dim });
      });
      line(ctx, 340, 760, 1600, 760, { color: C.faint, lw: 2 });
      text(ctx, '选它！', 420, 230 + 30, { size: 40, color: C.green, alpha: S.p(5, 2.2, 0.6), glow: 14 });
      ctx.restore();
    }
  };

  // ------------------------------------------------------------------ 论文图 1：完整结构
  // reveal: 0~1 逐个出现；lit: 需要高亮的部件 id 集合；flow: 反向传播光点（0~1 循环）
  function drawArch(ctx, cx, cy, sc, reveal, o = {}) {
    ctx.save();
    ctx.translate(cx, cy); ctx.scale(sc, sc);
    const E = -290, D = 290, bw = 300;
    const items = [
      ['lab', E, 455, '输入'], ['blk', E, 390, '输入嵌入', BLOCK.emb, 56], ['pe', E, 315],
      ['nx', E, 255, -100], ['blk', E, 190, '多头注意力', BLOCK.attn, 70], ['blk', E, 120, '相加 & 归一化', BLOCK.norm, 40],
      ['blk', E, 35, '前馈网络', BLOCK.ffn, 70], ['blk', E, -35, '相加 & 归一化', BLOCK.norm, 40],
      ['lab', D, 455, '输出（右移一位）'], ['blk', D, 390, '输出嵌入', BLOCK.emb, 56], ['pe', D, 315],
      ['nx', D, 255, -340], ['blk', D, 195, '带遮罩的多头注意力', BLOCK.attn, 64], ['blk', D, 130, '相加 & 归一化', BLOCK.norm, 36],
      ['cross'], ['blk', D, 50, '多头注意力（交叉）', BLOCK.attn, 64], ['blk', D, -15, '相加 & 归一化', BLOCK.norm, 36],
      ['blk', D, -95, '前馈网络', BLOCK.ffn, 64], ['blk', D, -160, '相加 & 归一化', BLOCK.norm, 36],
      ['blk', D, -245, '线性层', BLOCK.lin, 50], ['blk', D, -315, 'Softmax', BLOCK.soft, 50], ['lab', D, -385, '输出概率'],
    ];
    const n = items.length;
    items.forEach((it, k) => {
      const a = clamp(reveal * n - k);
      if (a <= 0) return;
      const [kind, x, y] = it;
      const yy = -y; // 设计坐标 y 向上
      if (kind === 'lab') text(ctx, it[3], x, yy, { size: 30, color: C.dim, alpha: a });
      else if (kind === 'blk') {
        const lit = o.lit && o.lit.has(k);
        block(ctx, x, yy, bw, it[5], it[3], it[4], { alpha: a, pop: a, lit, size: 26 });
        // 竖向连线（下一个部件在下面）
        arrow(ctx, x, yy + it[5] / 2 + 26, x, yy + it[5] / 2 + 2, { color: C.dim, lw: 2.5, head: 10, alpha: a });
      } else if (kind === 'pe') {
        circle(ctx, x, yy, 20, { stroke: C.text, lw: 2.5, alpha: a });
        text(ctx, '+', x, yy, { size: 32, family: 'main', alpha: a });
        ctx.save(); ctx.globalAlpha *= a; ctx.strokeStyle = C.green; ctx.lineWidth = 2.5; ctx.beginPath();
        const sx = x + (x < 0 ? -110 : 60);
        for (let i = 0; i <= 30; i++) { const px = sx + i * 1.7, py = yy + Math.sin(i / 30 * Math.PI * 2) * 12; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
        ctx.stroke(); ctx.restore();
        text(ctx, '位置编码', x + (x < 0 ? -150 : 160), yy, { size: 22, color: C.green, alpha: a, align: x < 0 ? 'right' : 'left' });
        line(ctx, x + (x < 0 ? -57 : 57), yy, x + (x < 0 ? -22 : 22), yy, { color: C.dim, lw: 2, alpha: a });
        arrow(ctx, x, yy + 45, x, yy + 22, { color: C.dim, lw: 2.5, head: 10, alpha: a });
      } else if (kind === 'nx') {
        const top = -it[3] + 35, bot = -y;
        roundRect(ctx, x - bw / 2 - 30, bot, bw + 60, top - bot, 20, { stroke: C.faint, lw: 2.5, alpha: a, fill: 'rgba(255,255,255,0.025)' });
        text(ctx, 'N=6 ×', x + (x < 0 ? -bw / 2 - 90 : bw / 2 + 90), (top + bot) / 2, { size: 30, family: 'main', color: C.dim, alpha: a });
      } else if (kind === 'cross') {
        // 编码器顶端 → 解码器交叉注意力
        ctx.save(); ctx.globalAlpha *= a;
        const lit = o.lit && o.lit.has('cross');
        ctx.strokeStyle = lit ? C.K : C.dim; ctx.lineWidth = lit ? 5 : 3; if (lit) { ctx.shadowColor = C.K; ctx.shadowBlur = 14; }
        ctx.beginPath(); ctx.moveTo(E, 35 - 20); ctx.lineTo(E, -90); ctx.lineTo(0, -90); ctx.lineTo(0, -50); ctx.lineTo(D - bw / 2 - 10, -50); ctx.stroke();
        T.arrowHead(ctx, D - bw / 2, -50, 0, 12, lit ? C.K : C.dim);
        ctx.restore();
      }
    });
    // 反向传播的红色光点
    if (o.flow != null) {
      for (let k = 0; k < 10; k++) {
        const f = (o.flow + k / 10) % 1;
        const col = k % 2 ? E : D;
        const y = lerp(-330, 400, f);
        circle(ctx, col, y, 8, { fill: C.red, glow: 16, alpha: 0.85 });
      }
    }
    ctx.restore();
  }
  window.drawArch = drawArch;

  // ------------------------------------------------------------------ 13 完整结构 + 训练 + 结果
  scenes.architecture = function (ctx, S) {
    const t = S.t;
    const rev = prog(t, 0.3, S.E(0) + 2.4);
    const side = S.p(1, 0, 1.0);
    const cx = lerp(960, 470, side), sc = lerp(0.86, 0.78, side);
    const flow = S.t > S.L(1) + 2.5 && S.t < S.L(2) ? (t * 0.4) % 1 : null;
    drawArch(ctx, cx, 470, sc, rev, { flow });
    text(ctx, '（论文 Figure 1）', cx, 470 + 420 * sc, { size: 24, color: C.faint, alpha: rev });

    const rx = 1360;
    // 训练
    const trA = S.p(1, 0.4, 0.8) * (1 - S.p(2, 0, 0.6));
    if (trA > 0) {
      ctx.save(); ctx.globalAlpha = trA;
      panel(ctx, 900, 140, 920, 700);
      text(ctx, '训练：预测 → 对答案 → 调参数', rx, 200, { size: 40, color: C.text });
      const vocab = [['cats', 0.35, 1], ['cat', 0.25, 0], ['dogs', 0.2, 0], ['it', 0.2, 0]];
      vocab.forEach(([w, p, y], i) => {
        const x = 1060 + i * 105;
        roundRect(ctx, x - 35, 520 - p * 400, 70, p * 400, 6, { fill: T.rgba(C.blue, 0.7) });
        text(ctx, w, x, 550, { size: 24, family: 'main' });
        const x2 = 1520 + i * 85;
        roundRect(ctx, x2 - 28, 520 - y * 240, 56, Math.max(2, y * 240), 6, { fill: C.green });
        text(ctx, w, x2, 550, { size: 22, family: 'main' });
      });
      text(ctx, '模型的预测', 1215, 300, { size: 28, color: C.blue });
      text(ctx, '标准答案', 1650, 250, { size: 28, color: C.green });
      const la = S.p(1, 2.0, 0.8);
      math(ctx, M.row('误差', '=', '−', '\\log', ' ', '0.35', '≈', '1.05'), rx, 640, { size: 40, color: C.red, alpha: la });
      math(ctx, M.row('W', '←', 'W', '−', 'η', ' ', '·', ' ', '\\梯度'), rx, 730, { size: 44, color: C.Q, alpha: S.p(1, 3.5, 0.8) });
      text(ctx, '（往让误差变小的方向挪一小步）', rx, 790, { size: 26, color: C.dim, alpha: S.p(1, 4.2, 0.8) });
      ctx.restore();
    }
    // BLEU
    const bA = S.p(2, 0.2, 0.8) * (1 - S.p(3, 0, 0.6));
    if (bA > 0) {
      ctx.save(); ctx.globalAlpha = bA;
      panel(ctx, 900, 140, 920, 700);
      text(ctx, '翻译质量 BLEU（越高越好）', rx, 200, { size: 40, color: C.text });
      const groups = [['英 → 德', [['此前最好', 26.36], ['Transformer', 28.4]]], ['英 → 法', [['此前最好', 41.29], ['Transformer', 41.8]]]];
      groups.forEach(([g, bars], gi) => {
        const gx = 1130 + gi * 460;
        text(ctx, g, gx, 790, { size: 34, color: C.text });
        bars.forEach(([lab, v], bi) => {
          const x = gx - 90 + bi * 180;
          const p = S.p(2, 0.8 + gi * 1.2 + bi * 0.5, 1.0);
          const h = (v / 45) * 460 * easeOut(p);
          const best = bi === 1;
          roundRect(ctx, x - 60, 740 - h, 120, h, 10, { fill: best ? C.Q : 'rgba(141,150,184,0.5)', glow: best ? 20 : 0 });
          text(ctx, v.toFixed(v === 41.8 || v === 28.4 ? 1 : 2), x, 740 - h - 28, { size: 34, family: 'main', color: best ? C.Q : C.dim, alpha: p });
          text(ctx, lab, x, 740 + 22, { size: 22, family: 'main', color: C.dim });
        });
      });
      text(ctx, '此前最好 = ConvS2S 集成模型；数据来自论文表 2', rx, 252, { size: 22, color: C.dim });
      ctx.restore();
    }
    // 训练时间
    const gA = S.p(3, 0.2, 0.8);
    if (gA > 0) {
      ctx.save(); ctx.globalAlpha = gA;
      panel(ctx, 900, 140, 920, 700);
      text(ctx, '并行计算 = 训练快', rx, 200, { size: 40, color: C.text });
      for (let i = 0; i < 8; i++) {
        const x = 1040 + (i % 4) * 150, y = 320 + Math.floor(i / 4) * 130;
        roundRect(ctx, x, y, 120, 90, 10, { fill: 'rgba(52,211,153,0.15)', stroke: C.green, lw: 2, glow: 10 * (0.5 + 0.5 * Math.sin(t * 5 + i)) });
        text(ctx, 'P100', x + 60, y + 45, { size: 26, family: 'main', color: C.green });
      }
      text(ctx, '8 块 NVIDIA P100 GPU', rx, 590, { size: 30, color: C.green });
      const a1 = S.p(3, 2.0, 0.8), a2 = S.p(3, 4.4, 0.8);
      text(ctx, '基础模型：12 小时', rx, 680, { size: 46, color: C.Q, alpha: a1, glow: 10 });
      text(ctx, '大模型：3.5 天', rx, 770, { size: 46, color: C.V, alpha: a2, glow: 10 });
      ctx.restore();
    }
  };

  // ------------------------------------------------------------------ 尾声
  scenes.legacy = function (ctx, S) {
    const t = S.t;
    // 家族树
    const treeA = S.p(0, 0, 0.8) * (1 - S.p(1, 0, 0.6));
    if (treeA > 0) {
      ctx.save(); ctx.globalAlpha = treeA;
      const root = [960, 230];
      chip(ctx, 'Transformer  2017', root[0], root[1], { size: 44, stroke: C.Q, color: C.Q, glow: 20 });
      const kids = [
        [520, 470, 'BERT  2018', '只用编码器 · 擅长“理解”', BLOCK.attn, 1.6],
        [1400, 470, 'GPT  2018', '只用解码器 · 擅长“生成”', C.blue, 3.4],
      ];
      kids.forEach(([x, y, a, b, col, at]) => {
        const p = S.p(0, at, 0.8);
        arrow(ctx, root[0], root[1] + 50, x, y - 55, { color: col, lw: 4, progress: p });
        chip(ctx, a, x, y, { size: 42, stroke: col, color: col, alpha: p });
        text(ctx, b, x, y + 70, { size: 30, color: C.dim, alpha: p });
      });
      const chain = [['GPT-2  2019', 5.6], ['GPT-3  2020', 6.4], ['ChatGPT  2022', 7.4]];
      chain.forEach(([s, at], i) => {
        const p = S.p(0, at, 0.7);
        const x = 1100 + i * 250, y = 700;
        chip(ctx, s, x, y, { size: 32, stroke: C.blue, alpha: p, glow: i === 2 ? 20 * p : 0 });
        if (i === 0) arrow(ctx, 1400, 560, x, y - 45, { color: C.blue, lw: 3, progress: p });
        else arrow(ctx, x - 250 + 100, y, x - 100, y, { color: C.blue, lw: 3, progress: p });
      });
      text(ctx, '模型越做越大', 1350, 800, { size: 30, color: C.dim, alpha: S.p(0, 7.0, 0.8) });
      ctx.restore();
    }
    // 一切皆可为“词”
    const modA = S.p(1, 0, 0.8) * (1 - S.p(2, 0, 0.6));
    if (modA > 0) {
      ctx.save(); ctx.globalAlpha = modA;
      const cols = [[430, '图片 → 小方块', C.V], [960, '声音 → 小片段', C.K], [1490, '蛋白质 → 氨基酸', C.green]];
      cols.forEach(([x, lab, col], k) => {
        const p = S.p(1, 0.6 + k * 1.4, 0.8);
        ctx.save(); ctx.globalAlpha *= p;
        if (k === 0) {
          for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
            const sep = 8 * S.p(1, 1.4, 0.8);
            const hue = `hsl(${200 + i * 20 + j * 10}, 60%, ${40 + (i + j) * 4}%)`;
            roundRect(ctx, x - 160 + j * (80 + sep), 280 + i * (80 + sep), 80, 80, 4, { fill: hue });
          }
        } else if (k === 1) {
          ctx.strokeStyle = col; ctx.lineWidth = 4; ctx.beginPath();
          for (let i = 0; i <= 300; i++) { const px = x - 170 + i * 1.13, py = 440 + Math.sin(i * 0.15) * Math.sin(i * 0.021) * 100; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
          ctx.stroke();
          for (let i = 1; i < 6; i++) line(ctx, x - 170 + i * 57, 320, x - 170 + i * 57, 560, { color: C.faint, lw: 2, dash: [6, 6] });
        } else {
          for (let i = 0; i < 9; i++) {
            const a = i * 0.9;
            const px = x - 140 + i * 35, py = 440 + Math.sin(a) * 80;
            if (i) line(ctx, x - 140 + (i - 1) * 35, 440 + Math.sin((i - 1) * 0.9) * 80, px, py, { color: C.dim, lw: 3 });
            circle(ctx, px, py, 20, { fill: ['#34d399', '#60a5fa', '#f472b6', '#facc15'][i % 4] });
          }
        }
        text(ctx, lab, x, 700, { size: 36, color: col });
        ctx.restore();
      });
      text(ctx, '切成小块，就都能当成「词」', 960, 210, { size: 46, color: C.text, alpha: S.p(1, 3.5, 0.8) });
      ctx.restore();
    }
    // 回顾
    const recA = S.p(2, 0, 0.6) * (1 - S.p(3, 0, 0.8));
    if (recA > 0) {
      ctx.save(); ctx.globalAlpha = recA;
      const steps = [['词 → 向量', '+ 位置编码', C.blue, 0.0], ['Q · K', '点积打分', C.Q, 0.22], ['softmax', '变成比例', C.green, 0.4], ['混合 V', '加权求和', C.V, 0.58], ['多头 × 6 层', '并行 · 堆叠', C.purple, 0.78]];
      steps.forEach(([a, b, col, f], i) => {
        const p = prog(t, S.at(2, f), 0.6);
        const x = 260 + i * 350;
        ctx.save(); ctx.globalAlpha *= p; ctx.translate(x, 470); const s = 0.8 + 0.2 * easeBack(p); ctx.scale(s, s);
        roundRect(ctx, -140, -120, 280, 240, 24, { fill: T.rgba(col, 0.12), stroke: col, lw: 3, glow: 14 });
        text(ctx, String(i + 1), 0, -70, { size: 36, family: 'main', color: col });
        text(ctx, a, 0, 0, { size: 40, color: C.text, weight: 700 });
        text(ctx, b, 0, 60, { size: 28, color: C.dim });
        ctx.restore();
        if (i < 4) arrow(ctx, x + 150, 470, x + 200, 470, { color: C.dim, lw: 3, alpha: p });
      });
      ctx.restore();
    }
    // 结尾
    const endA = S.p(3, 0.2, 1.2);
    if (endA > 0) {
      ctx.save(); ctx.globalAlpha = endA;
      math(ctx, window.SHARED.FORMULA, 960, 300, {
        size: 72,
        style: (id) => ({ qk: { color: C.Q }, sd: { color: C.red }, sm: { color: C.green }, v: { color: C.V } }[id]),
      });
      const tA = S.p(3, 3.0, 1.5);
      text(ctx, 'Attention Is All You Need', 960, 560, { size: 88, family: 'main', weight: 400, alpha: tA, glow: 24, glowColor: 'rgba(120,170,255,0.7)' });
      text(ctx, '感谢观看', 960, 700, { size: 44, color: C.dim, alpha: S.p(3, 5.0, 1.0) });
      ctx.restore();
    }
  };

  window.SCENES = Object.assign(window.SCENES || {}, scenes);
})();
