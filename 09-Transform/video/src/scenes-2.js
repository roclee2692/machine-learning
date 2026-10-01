/*
 * 分镜（中）：Softmax → 混合 Value → 公式与缩放 → 多头 → 位置编码
 */
(function () {
  const T = window.TV;
  const { C, W, H, text, chip, arrow, arcArrow, line, circle, roundRect, vecCells, grid, math, M, panel } = T;
  const { clamp, lerp, ease, easeOut, easeBack, prog } = T;
  const { SENT, IT, Q_IT, KEYS, SCORES, WEIGHTS, VCOLORS } = window.SHARED;
  const scenes = {};

  // ------------------------------------------------------------------ 06 Softmax
  scenes.softmax = function (ctx, S) {
    const t = S.t;
    const xs = SENT.map((_, i) => 480 + i * 182);
    const lx = 210;
    const base = 540, unit = 40;
    const expP = S.p(2, 0.6, 1.4);         // 分数 → e^分数
    const normP = S.p(2, 4.2, 1.0);        // e^分数 → 比例
    const exps = SCORES.map(Math.exp);
    const sumE = exps.reduce((a, b) => a + b, 0);
    const hl = S.p(3, 0, 0.8);

    // 左侧标签
    text(ctx, '词', lx, 170, { size: 32, color: C.dim, alpha: S.p(0, 0, 0.5) });
    math(ctx, M.row('Key', ' ', 'k'), lx, 250, { size: 32, color: C.K, alpha: S.p(0, 0.3, 0.5) });
    const rowLabel = normP > 0.5 ? '注意力比例' : expP > 0.5 ? 'e 的分数次方' : '分数 q·k';
    text(ctx, rowLabel, lx, 330, { size: 32, color: normP > 0.5 ? C.Q : C.text, alpha: S.p(0, 0.6, 0.5) });
    // 「它」的 Query
    const qa = S.p(0, 0.6, 0.6);
    roundRect(ctx, 40, 400, 330, 120, 18, { fill: T.rgba(C.Q, 0.08), stroke: C.Q, lw: 2, alpha: qa });
    text(ctx, '「它」的 Query', 205, 432, { size: 28, color: C.Q, alpha: qa });
    vecCells(ctx, Q_IT, 205, 482, { cell: 52, nums: true, color: C.Q, range: 1, numSize: 22, alpha: qa });

    line(ctx, 400, base, 1800, base, { color: C.faint, lw: 2, alpha: S.p(0, 1, 0.6) });

    SENT.forEach((w, i) => {
      const x = xs[i];
      const ap = S.p(0, 0.2 + i * 0.08, 0.5);
      const isCat = i === 0;
      chip(ctx, w, x, 170, { size: 34, w: 120, alpha: ap, stroke: i === IT ? C.Q : isCat && hl > 0 ? C.orange : undefined, glow: isCat ? 20 * hl : 0 });
      vecCells(ctx, KEYS[i], x, 250, { cell: 36, gap: 3, nums: true, numSize: 15, color: C.K, range: 2, alpha: ap });
      // 分数逐个算出来
      const sp = S.p(0, 1.8 + i * 0.32, 0.5);
      if (sp <= 0) return;
      const score = SCORES[i];
      const ex = exps[i];
      // 柱高：分数 → e^分数（统一缩放到最高 190 像素）
      const hScore = score * unit;
      const hExp = (ex / exps[0]) * 160;
      const h = lerp(hScore, hExp, expP) * easeOut(sp);
      const col = normP > 0 ? T.mixColor('#60a5fa', C.Q, normP) : expP > 0 ? T.mixColor('#60a5fa', C.green, expP) : C.blue;
      const dimmed = hl > 0 && !isCat ? 1 - 0.5 * hl : 1;
      roundRect(ctx, x - 34, h >= 0 ? base - h : base, 68, Math.max(2, Math.abs(h)), 8, { fill: col, alpha: sp * dimmed, glow: isCat ? 24 * hl : 0 });
      let label;
      if (normP > 0.5) label = (WEIGHTS[i] * 100).toFixed(1) + '%';
      else if (expP > 0.5) label = ex.toFixed(ex < 10 ? 2 : 1);
      else label = T.fmtNum(score, 1);
      text(ctx, label, x, 330, { size: 34, family: 'main', color: normP > 0.5 ? C.Q : C.text, alpha: sp * dimmed, weight: 400 });
    });

    // softmax 公式
    const fa = S.p(1, 1.6, 0.8);
    math(ctx, M.row('\\softmax', M.paren(M.sub('x', 'i')), '=', M.frac(M.sup('e', M.sub('x', 'i')), M.row('Σ', M.sub('', 'j'), ' ', M.sup('e', M.sub('x', 'j'))))), 960, 720, { size: 60, alpha: fa });
    const s1 = S.p(2, 0.4, 0.6) * (1 - S.p(2, 4.0, 0.4));
    text(ctx, '① 取指数：全变正数，大的更大', 960, 830, { size: 32, color: C.green, alpha: s1 });
    const s2 = S.p(2, 3.6, 0.6);
    text(ctx, '② 除以总和 ' + sumE.toFixed(1) + '：加起来 = 1', 960, 830, { size: 32, color: C.Q, alpha: s2 });
    // 小猫高亮
    if (hl > 0) {
      text(ctx, (WEIGHTS[0] * 100).toFixed(0) + '% 的注意力给了「小猫」', W / 2 + 180, 440, { size: 46, color: C.orange, alpha: hl, glow: 16 });
    }
  };

  // ------------------------------------------------------------------ 07 混合 Value
  scenes.mix = function (ctx, S) {
    const t = S.t;
    const xs = SENT.map((_, i) => 330 + i * 168);
    const py = 230;
    const bowl = [760, 610], br = 140;
    const pour = (i) => prog(t, S.L(1) + 0.6 + i * 0.45, 1.2);
    // 颜料
    SENT.forEach((w, i) => {
      const a = S.p(0, i * 0.1, 0.5);
      circle(ctx, xs[i], py, 40, { fill: VCOLORS[i], alpha: a, glow: 16 });
      text(ctx, w, xs[i], py - 72, { size: 32, color: C.text, alpha: a });
      text(ctx, (WEIGHTS[i] * 100).toFixed(0) + '%', xs[i], py + 70, { size: 30, family: 'main', color: C.Q, alpha: a });
      math(ctx, M.sub('v', String(i + 1)), xs[i], py, { size: 30, color: '#0a0f1f', alpha: a });
      // 倒入
      const p = pour(i);
      if (p > 0 && p < 1) {
        const n = Math.max(2, Math.round(WEIGHTS[i] * 30));
        for (let k = 0; k < n; k++) {
          const f = clamp(p * 1.6 - k / n * 0.6);
          if (f <= 0 || f >= 1) continue;
          const x = lerp(xs[i], bowl[0], f), y = lerp(py + 40, bowl[1] - 40, f) - Math.sin(f * Math.PI) * 120;
          circle(ctx, x, y, 6 + WEIGHTS[i] * 14, { fill: VCOLORS[i] });
        }
      }
    });
    // 碗：颜色 = 已倒入部分的加权平均
    let wsum = 0, acc = [0, 0, 0];
    SENT.forEach((_, i) => {
      const p = clamp(pour(i) * 1.2 - 0.2);
      const w = WEIGHTS[i] * p;
      const c = T.hexToRgb(VCOLORS[i]);
      wsum += w; acc = acc.map((v, k) => v + c[k] * w);
    });
    const mixed = wsum > 0 ? acc.map((v) => v / wsum) : [40, 50, 80];
    const bowlA = S.p(0, 1.0, 0.6);
    ctx.save(); ctx.globalAlpha = bowlA;
    ctx.beginPath(); ctx.arc(bowl[0], bowl[1], br, 0, Math.PI); ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.05)'; ctx.fill();
    const level = wsum; // 0~1
    if (level > 0) {
      ctx.save(); ctx.clip();
      ctx.fillStyle = T.rgb(mixed);
      ctx.fillRect(bowl[0] - br, bowl[1] + br - br * level - 2, br * 2, br * level + 4);
      ctx.restore();
    }
    ctx.beginPath(); ctx.arc(bowl[0], bowl[1], br, 0, Math.PI); ctx.closePath();
    ctx.strokeStyle = C.dim; ctx.lineWidth = 4; ctx.stroke();
    ctx.restore();

    // 右侧：加权求和公式
    const fx = 1380;
    const fa = S.p(1, 0.2, 0.6) * (1 - S.p(2, 0, 0.5));
    math(ctx, M.row('输出', '=', 'Σ', ' ', M.sub('w', 'i'), ' ', M.sub('v', 'i')), fx, 420, { size: 56, alpha: fa });
    const order = WEIGHTS.map((w, i) => [w, i]).sort((a, b) => b[0] - a[0]).slice(0, 4);
    order.forEach(([w, i], k) => {
      const a = S.p(1, 1.0 + k * 0.6, 0.5) * (1 - S.p(2, 0, 0.5));
      const y = 510 + k * 58;
      text(ctx, (k ? '+ ' : '= ') + w.toFixed(2) + ' ×', fx - 40, y, { size: 36, family: 'main', align: 'right', alpha: a });
      circle(ctx, fx + 0, y, 18, { fill: VCOLORS[i], alpha: a });
      text(ctx, SENT[i], fx + 32, y, { size: 32, align: 'left', alpha: a, color: C.dim });
    });
    text(ctx, '+ ……', fx - 40, 510 + 4 * 58, { size: 36, family: 'main', align: 'right', alpha: S.p(1, 3.4, 0.5) * (1 - S.p(2, 0, 0.5)) });

    // 结果：它 → 它'
    const ra = S.p(2, 0.4, 0.8);
    if (ra > 0) {
      ctx.save(); ctx.globalAlpha = ra;
      panel(ctx, 1080, 340, 680, 440, { fill: 'rgba(10,15,30,0.94)' });
      chip(ctx, '它', 1240, 500, { size: 56, w: 150, h: 110, fill: 'rgba(229,231,235,0.15)' });
      text(ctx, '之前', 1240, 590, { size: 28, color: C.dim });
      arrow(ctx, 1340, 500, 1460, 500, { color: C.text, lw: 4, progress: S.p(2, 1.0, 0.6) });
      const g = S.p(2, 1.4, 0.8);
      chip(ctx, '它′', 1590, 500, { size: 56, w: 150, h: 110, fill: T.rgb(mixed), stroke: C.orange, glow: 30 * g, alpha: g, color: '#0a0f1f' });
      text(ctx, '之后：带着「小猫」的信息', 1420, 690, { size: 34, color: C.orange, alpha: S.p(2, 2.6, 0.8) });
      ctx.restore();
    }
  };

  // ------------------------------------------------------------------ 08 公式与缩放
  const FORMULA = M.row(
    M.id('att', M.row('\\Attention', M.paren('Q', ',', 'K', ',', 'V'))), '=',
    M.id('sm', M.row('\\softmax', M.paren(M.frac(M.id('qk', M.row('Q', M.sup('K', '\\T'))), M.id('sd', M.sqrt(M.sub('d', 'k'))))))),
    ' ', M.id('v', 'V'));
  window.SHARED.FORMULA = FORMULA;

  scenes.formula = function (ctx, S) {
    const t = S.t;
    const up = S.p(3, 0, 1.0);
    const fy = lerp(220, 150, up), fs = lerp(84, 64, up);
    const hlQK = S.p(1, 0, 0.6) * (1 - S.p(2, 0, 0.6));
    const hlSM = S.p(2, 0, 0.6) * (1 - S.p(3, 0, 0.6));
    const hlSD = S.p(3, 0, 0.6);
    const box = math(ctx, FORMULA, W / 2, fy, {
      size: fs, alpha: S.p(0, 0.2, 1.0),
      style: (id) => {
        if (id === 'qk') return { color: hlQK > 0.1 ? C.Q : C.text, glow: 14 * hlQK };
        if (id === 'sd') return { color: hlSD > 0.1 ? C.red : C.text, glow: 16 * hlSD };
        if (id === 'sm') return hlSM > 0.1 ? { color: C.green } : null;
        if (id === 'v') return { color: hlSM > 0.1 ? C.V : C.text };
        return null;
      },
    });

    // 矩阵演示：Q × Kᵀ → 分数表 → softmax → × V
    const matA = S.p(1, 0.4, 0.8) * (1 - S.p(3, 0, 0.6));
    if (matA > 0) {
      ctx.save(); ctx.globalAlpha = matA;
      const r = T.rng(5);
      const Q = SENT.map((_, i) => (i === IT ? Q_IT : Q_IT.map(() => Math.round(T.gauss(r) * 8) / 10)));
      const K = KEYS;
      const Vm = SENT.map((_, i) => T.hexToRgb(VCOLORS[i]).map((c) => c / 255).concat([0.5]));
      const scores = T.matmul(Q, T.transpose(K));
      const weights = scores.map((row) => T.softmax(row));
      const out = T.matmul(weights, Vm);
      const y = 590;
      const gQ = grid(ctx, Q, 250, y, { cell: 30, range: 2, reveal: S.p(1, 0.4, 1.0) });
      math(ctx, 'Q', 250, y - gQ.h / 2 - 36, { size: 44, color: C.Q });
      SENT.forEach((w, i) => text(ctx, w, 250 - gQ.w / 2 - 44, gQ.y0 + i * (gQ.ch + gQ.gap) + gQ.ch / 2, { size: 20, color: i === IT ? C.Q : C.dim }));
      text(ctx, '×', 400, y, { size: 48, color: C.dim, family: 'main' });
      const gK = grid(ctx, T.transpose(K), 600, y, { cell: 30, range: 2, reveal: S.p(1, 0.9, 1.0), byCol: true });
      math(ctx, M.sup('K', '\\T'), 600, y - gK.h / 2 - 36, { size: 44, color: C.K });
      text(ctx, '=', 800, y, { size: 48, color: C.dim, family: 'main' });
      const sp = S.p(1, 2.0, 2.0);
      const smP = S.p(2, 0.6, 1.4);
      const M8 = scores.map((row, i) => row.map((s, j) => lerp(s / 4, weights[i][j], smP)));
      const gS = grid(ctx, M8, 1060, y, {
        cell: 44, gap: 3, reveal: sp,
        fill: (v) => (smP > 0.5 ? T.heat(v * 1.6) : T.diverge(v)),
        cellAlpha: (r_, c_) => (r_ === IT ? 1 : 0.85),
      });
      SENT.forEach((w, i) => {
        text(ctx, w, gS.x0 + i * (gS.cw + gS.gap) + gS.cw / 2, gS.y0 - 22, { size: 18, color: C.dim, alpha: sp });
        text(ctx, w, gS.x0 - 32, gS.y0 + i * (gS.ch + gS.gap) + gS.ch / 2, { size: 18, color: i === IT ? C.Q : C.dim, alpha: sp });
      });
      roundRect(ctx, gS.x0 - 4, gS.y0 + IT * (gS.ch + gS.gap) - 4, gS.w + 8, gS.ch + 8, 6, { stroke: C.Q, lw: 3, alpha: sp });
      text(ctx, smP > 0.5 ? '注意力表（每行加起来 = 1）' : '分数表：所有词两两打分', 1060, y + gS.h / 2 + 36, { size: 28, color: smP > 0.5 ? C.green : C.Q, alpha: sp });
      const va = S.p(2, 2.4, 0.8);
      text(ctx, '×', 1330, y, { size: 48, color: C.dim, family: 'main', alpha: va });
      const gV = grid(ctx, Vm, 1440, y, { cell: 30, fill: (v, r_, c_) => (c_ < 3 ? VCOLORS[r_] : 'rgba(255,255,255,0.15)'), alpha: va });
      math(ctx, 'V', 1440, y - gV.h / 2 - 36, { size: 44, color: C.V, alpha: va });
      const oa = S.p(2, 3.2, 0.8);
      text(ctx, '=', 1550, y, { size: 48, color: C.dim, family: 'main', alpha: oa });
      grid(ctx, out, 1680, y, { cell: 30, fill: (v, r_, c_) => (c_ < 3 ? T.rgb(out[r_].slice(0, 3).map((x) => x * 255)) : 'rgba(255,255,255,0.15)'), alpha: oa });
      text(ctx, '新表示', 1680, y + gV.h / 2 + 36, { size: 26, color: C.dim, alpha: oa });
      ctx.restore();
    }

    // 为什么除以 √d_k
    const dk = S.p(3, 0.8, 0.8);
    if (dk > 0) {
      const bx = box.boxes.sd;
      if (bx) roundRect(ctx, bx.x - 10, bx.y - 8, bx.w + 20, bx.h + 16, 10, { stroke: C.red, lw: 3, alpha: dk * (1 - S.p(4, 0, 0.5)) });
      math(ctx, M.row(M.sub('d', 'k'), '=', '64', ' ', '（每个 Q、K 向量有 64 个数）'), W / 2, 290, { size: 44, color: C.text, alpha: dk * (1 - S.p(5, 3.6, 0.4)) });
    }
    const demo = S.p(4, 0.2, 0.8);
    if (demo > 0) {
      const sc = 1 / lerp(1, 8, S.p(5, 4.6, 2.2)); // 除以 √64 = 8
      ctx.save(); ctx.globalAlpha = demo;
      // 左：分数分布直方图
      panel(ctx, 110, 360, 820, 480, { fill: 'rgba(10,15,30,0.9)' });
      text(ctx, '随机 64 维向量的点积分布', 520, 400, { size: 30, color: C.dim });
      const r = T.rng(99);
      const samples = [];
      for (let k = 0; k < 1500; k++) {
        let s = 0; for (let d = 0; d < 64; d++) s += T.gauss(r) * T.gauss(r);
        samples.push(s * sc);
      }
      const bins = 60, lo = -30, hi = 30, bw = (hi - lo) / bins;
      const counts = new Array(bins).fill(0);
      samples.forEach((s) => { const b = Math.floor((s - lo) / bw); if (b >= 0 && b < bins) counts[b]++; });
      const cmax = 330;
      const ax0 = 150, ax1 = 890, ay = 780;
      counts.forEach((c, b) => {
        const h = Math.min(285, (c / 1500) * cmax * 9);
        roundRect(ctx, ax0 + (b / bins) * (ax1 - ax0) + 1, ay - h, (ax1 - ax0) / bins - 2, h, 2, { fill: sc < 0.5 ? C.green : C.red });
      });
      line(ctx, ax0, ay, ax1, ay, { color: C.dim, lw: 2 });
      [-30, -15, -8, 0, 8, 15, 30].forEach((v) => text(ctx, String(v).replace('-', '−'), ax0 + ((v - lo) / (hi - lo)) * (ax1 - ax0), ay + 26, { size: 22, family: 'main', color: C.dim }));
      const sdv = 8 * sc;
      text(ctx, '典型大小 ≈ ±' + sdv.toFixed(1), 520, 450, { size: 34, color: sc < 0.5 ? C.green : C.red });
      // 右：softmax 结果
      panel(ctx, 990, 360, 820, 480, { fill: 'rgba(10,15,30,0.9)' });
      text(ctx, '一行分数做 softmax 之后', 1400, 400, { size: 30, color: C.dim });
      const raw = [11.5, -6.2, 3.1, 8.7, -2.4, 0.9, -9.8, 5.3];
      const w = T.softmax(raw.map((x) => x * sc));
      w.forEach((v, i) => {
        const x = 1080 + i * 88, h = v * 300;
        roundRect(ctx, x, 780 - h, 60, Math.max(2, h), 6, { fill: sc < 0.5 ? C.green : C.red });
        text(ctx, (v * 100).toFixed(0) + '%', x + 30, 780 - h - 22, { size: 22, family: 'main', color: C.text });
        text(ctx, T.fmtNum(raw[i] * sc, 1), x + 30, 806, { size: 20, family: 'main', color: C.dim });
      });
      const bad = sc > 0.5;
      text(ctx, bad ? '几乎全给了一个词 → 难以学习' : '分配得更柔和 → 好学习', 1400, 450, { size: 34, color: bad ? C.red : C.green });
      ctx.restore();
      const ff = S.p(5, 4.0, 0.8);
      math(ctx, M.row('÷', M.sqrt('64'), '=', '÷', '8'), W / 2, 290, { size: 56, color: C.green, alpha: ff, glow: 14 });
    }
  };

  // ------------------------------------------------------------------ 09 多头注意力
  function headPattern(h) {
    const n = SENT.length, r = T.rng(100 + h);
    return SENT.map((_, i) => {
      const s = SENT.map((_, j) => T.gauss(r) * 0.4);
      if (h === 0) { s[i] += 1.5; if (i === IT) { s[0] += 4; s[i] -= 1; } if (i === 7) s[0] += 2.5; }
      else if (h === 1) { const pair = { 0: 2, 2: 0, 3: 2, 7: 5, 5: 7 }; if (pair[i] != null) s[pair[i]] += 3.5; else s[i] += 1.5; }
      else if (h === 2) { if (i > 0) s[i - 1] += 3.5; else s[0] += 3; }
      else if (h === 3) { if (i < n - 1) s[i + 1] += 3.5; else s[i] += 3; }
      else if (h === 4) { s[4] += 2.5; }
      else if (h === 5) { s[i] += 3.5; }
      else if (h === 6) { s[0] += 2; s[3] += 2; }
      else { s.forEach((_, j) => (s[j] += j <= i ? 0.8 : 0)); }
      return T.softmax(s);
    });
  }
  const HEADS = Array.from({ length: 8 }, (_, h) => headPattern(h));
  const HEAD_COLORS = [C.Q, C.K, C.V, C.blue, C.purple, C.green, C.orange, C.red];
  const HEAD_NAMES = ['找代词指谁', '主语 ↔ 动词', '看前一个词', '看后一个词', '关注连接词', '看自己', '关注名词', '看前文整体'];

  scenes.multihead = function (ctx, S) {
    const t = S.t;
    // 第 0 句：同一句话里的多种关系
    const relA = S.p(0, 0, 0.6) * (1 - S.p(1, 0, 0.6));
    if (relA > 0) {
      ctx.save(); ctx.globalAlpha = relA;
      const xs = SENT.map((_, i) => 290 + i * 190), y = 560;
      SENT.forEach((w, i) => chip(ctx, w, xs[i], y, { size: 40, w: 130 }));
      const rels = [[IT, 0, C.Q, '指代', 1.0], [2, 0, C.K, '主语', 2.2], [2, 3, C.V, '宾语', 3.4], [7, 6, C.purple, '修饰', 4.6]];
      rels.forEach(([a, b, col, name, at]) => {
        const p = S.p(0, at, 0.8);
        const up = (a + b) % 2 === 0 ? 1 : -1;
        const bend = -up * (60 + Math.abs(a - b) * 30);
        const q = arcArrow(ctx, xs[a], y - up * 50, xs[b], y - up * 50, bend * Math.sign(b - a || 1), { color: col, lw: 4, progress: p, glow: 8 });
        if (q) { const [lx, ly] = q(0.5); text(ctx, name, lx, ly - up * 30, { size: 34, color: col, alpha: p }); }
      });
      ctx.restore();
    }
    // 8 个头
    const headsA = S.p(1, 0.5, 0.8);
    const merge = S.p(3, 0.4, 1.6);
    if (headsA > 0 && merge < 1) {
      for (let h = 0; h < 8; h++) {
        const col = HEAD_COLORS[h];
        const cx = 290 + (h % 4) * 447, cy = h < 4 ? 340 : 600;
        const a = S.p(1, 0.5 + h * 0.18, 0.6);
        const focus = S.p(2, 0, 0.6);
        const isKey = h < 3;
        const dimF = isKey ? 1 : 1 - 0.55 * focus;
        // 合并时飞向中间的长条
        const tx = 960 - 4 * 90 + h * 90 + 45, ty = 470;
        const mx = lerp(cx, tx, ease(merge)), my = lerp(cy, ty, ease(merge));
        ctx.save(); ctx.globalAlpha = a * dimF * (1 - merge);
        ctx.translate(mx, my); const sc = 1 - 0.7 * ease(merge); ctx.scale(sc, sc);
        roundRect(ctx, -205, -115, 410, 230, 18, { fill: T.rgba(col, 0.07), stroke: col, lw: isKey && focus > 0 ? 3 : 1.5, glow: isKey ? 14 * focus : 0 });
        grid(ctx, HEADS[h], -95, 0, { cell: 21, gap: 2, fill: (v) => T.rgba(col, clamp(0.06 + v * 1.6, 0, 1)), stroke: 'rgba(255,255,255,0.05)' });
        text(ctx, '头 ' + (h + 1), 105, -60, { size: 32, color: col, weight: 700 });
        math(ctx, M.row(M.sup(M.sub('W', 'Q'), String(h + 1)), ' ', M.sup(M.sub('W', 'K'), String(h + 1)), ' ', M.sup(M.sub('W', 'V'), String(h + 1))), 105, -10, { size: 22, color: C.dim });
        text(ctx, focus > 0 ? (isKey ? HEAD_NAMES[h] : '……') : '', 105, 50, { size: 28, color: isKey ? C.text : C.dim, alpha: focus });
        ctx.restore();
      }
    }
    // 拼接 + W_O
    if (merge > 0) {
      const a = S.p(3, 1.4, 0.8);
      ctx.save();
      for (let h = 0; h < 8; h++) {
        const x = 960 - 4 * 90 + h * 90;
        roundRect(ctx, x + 4, 430, 82, 80, 6, { fill: T.rgba(HEAD_COLORS[h], 0.75), alpha: merge });
        text(ctx, '64', x + 45, 470, { size: 26, family: 'main', color: '#0a0f1f', alpha: a });
      }
      ctx.restore();
      text(ctx, '拼接：8 × 64 = 512 维', 960, 370, { size: 40, color: C.text, alpha: a });
      const wo = S.p(3, 3.0, 0.8);
      arrow(ctx, 960, 525, 960, 600, { color: C.dim, lw: 4, progress: wo });
      roundRect(ctx, 860, 610, 200, 90, 14, { fill: 'rgba(167,139,250,0.2)', stroke: C.purple, lw: 3, alpha: wo });
      math(ctx, M.row('×', ' ', M.sub('W', 'O')), 960, 655, { size: 44, color: C.purple, alpha: wo });
      const out = S.p(3, 4.4, 0.8);
      arrow(ctx, 1065, 655, 1210, 655, { color: C.dim, lw: 4, progress: out });
      const r = T.rng(3);
      vecCells(ctx, Array.from({ length: 16 }, () => T.gauss(r)), 1500, 655, { cell: 26, gap: 3, alpha: out });
      text(ctx, '汇总成一份 512 维的报告', 1500, 720, { size: 30, color: C.dim, alpha: out });
    }
  };

  // ------------------------------------------------------------------ 10 位置编码
  const PE = (pos, i, d = 128) => (i % 2 === 0 ? Math.sin(pos / Math.pow(10000, i / d)) : Math.cos(pos / Math.pow(10000, (i - 1) / d)));

  function clockFace(ctx, x, y, r, ang, col, o = {}) {
    circle(ctx, x, y, r, { fill: 'rgba(255,255,255,0.03)', stroke: T.rgba(col, 0.6), lw: 3, alpha: o.alpha });
    for (let k = 0; k < 12; k++) {
      const a = (k / 12) * Math.PI * 2;
      line(ctx, x + Math.cos(a) * r * 0.85, y + Math.sin(a) * r * 0.85, x + Math.cos(a) * r * 0.95, y + Math.sin(a) * r * 0.95, { color: T.rgba(col, 0.4), lw: 2, alpha: o.alpha });
    }
    // 角度从正上方开始顺时针：x = sin, y = cos
    const hx = x + Math.sin(ang) * r * 0.8, hy = y - Math.cos(ang) * r * 0.8;
    line(ctx, x, y, hx, hy, { color: col, lw: 6, alpha: o.alpha, glow: 10 });
    circle(ctx, x, y, 7, { fill: col, alpha: o.alpha });
  }

  scenes.position = function (ctx, S) {
    const t = S.t;
    // 第 0 句：猫追狗 vs 狗追猫
    const bagA = S.p(0, 0, 0.6) * (1 - S.p(1, 0, 0.6));
    if (bagA > 0) {
      ctx.save(); ctx.globalAlpha = bagA;
      const sents = [['猫', '追', '狗'], ['狗', '追', '猫']];
      const fall = S.p(0, 2.6, 1.2);
      sents.forEach((s, k) => {
        const cx = k ? 1340 : 580;
        text(ctx, k ? '狗追猫 🐕→🐈' : '猫追狗 🐈→🐕', cx, 230, { size: 40, color: C.dim });
        s.forEach((w, i) => {
          // 落进“袋子”后按同一顺序（字典序）排列 —— 看不出谁在前
          const sortedIdx = ['猫', '狗', '追'].indexOf(w);
          const x0 = cx + (i - 1) * 140, y0 = 340;
          const x1 = cx + (sortedIdx - 1) * 110, y1 = 600;
          const f = ease(fall);
          chip(ctx, w, lerp(x0, x1, f), lerp(y0, y1, f) - Math.sin(f * Math.PI) * 40, { size: 48, w: 100, h: 90, stroke: w === '猫' ? C.orange : w === '狗' ? C.blue : undefined });
        });
        ctx.save(); ctx.globalAlpha *= fall;
        ctx.beginPath(); ctx.ellipse(cx, 610, 220, 110, 0, 0, Math.PI * 2); ctx.strokeStyle = C.dim; ctx.lineWidth = 3; ctx.setLineDash([10, 10]); ctx.stroke();
        ctx.restore();
      });
      const eq = S.p(0, 3.8, 0.6);
      text(ctx, '=', 960, 610, { size: 110, color: C.red, alpha: eq, glow: 16 });
      text(ctx, '在注意力看来，一模一样', 960, 790, { size: 40, color: C.red, alpha: eq });
      ctx.restore();
    }

    // 第 1 句：词向量 + 位置向量
    const addA = S.p(1, 0.2, 0.6) * (1 - S.p(2, 0, 0.6));
    if (addA > 0) {
      ctx.save(); ctx.globalAlpha = addA;
      const words = ['猫', '追', '狗'];
      const r = T.rng(8);
      const emb = { 猫: [], 追: [], 狗: [] };
      Object.keys(emb).forEach((k) => (emb[k] = Array.from({ length: 8 }, () => T.gauss(r) * 0.6)));
      text(ctx, '词向量', 620, 220, { size: 34, color: C.text });
      text(ctx, '位置向量', 1080, 220, { size: 34, color: C.green });
      text(ctx, '实际输入', 1560, 220, { size: 34, color: C.Q });
      words.forEach((w, i) => {
        const y = 330 + i * 170;
        const a = S.p(1, 0.6 + i * 0.4, 0.6);
        ctx.save(); ctx.globalAlpha *= a;
        chip(ctx, w, 220, y, { size: 44, w: 90, h: 80 });
        text(ctx, '位置 ' + i, 220, y + 60, { size: 24, color: C.green });
        vecCells(ctx, emb[w], 620, y, { cell: 40, gap: 4 });
        text(ctx, '+', 850, y, { size: 56, color: C.dim, family: 'main' });
        const pe = Array.from({ length: 8 }, (_, k) => PE(i, k, 8));
        vecCells(ctx, pe, 1080, y, { cell: 40, gap: 4 });
        text(ctx, '=', 1310, y, { size: 56, color: C.dim, family: 'main' });
        const sumP = S.p(1, 2.4 + i * 0.3, 0.8);
        vecCells(ctx, emb[w].map((v, k) => v + pe[k]), 1560, y, { cell: 40, gap: 4, reveal: sumP });
        ctx.restore();
      });
      ctx.restore();
    }

    // 第 2~3 句：公式 + 时钟 + 热力图
    const clkA = S.p(2, 0.2, 0.8) * (1 - S.p(4, 0, 0.6));
    const pos = 24 * ease(prog(t, S.L(2) + 2.5, S.E(3) - S.L(2) - 2.0)); // 位置从 0 走到 24
    if (clkA > 0) {
      ctx.save(); ctx.globalAlpha = clkA;
      const frac = M.frac('pos', M.sup('10000', M.row('2', 'i', '/', 'd')));
      math(ctx, M.row('PE', M.paren('pos', ',', '2', 'i'), '=', '\\sin', M.paren(frac)), 520, 210, { size: 40 });
      math(ctx, M.row('PE', M.paren('pos', ',', '2', 'i', '+', '1'), '=', '\\cos', M.paren(frac)), 1400, 210, { size: 40 });
      // 四个时钟：不同的 i → 不同的转速
      const dims = [0, 16, 32, 64]; // i（第几对）
      const d = 512;
      dims.forEach((i, k) => {
        const w = 1 / Math.pow(10000, (2 * i) / d);
        const x = 220 + (k % 2) * 300, y = 430 + Math.floor(k / 2) * 280;
        const a = S.p(2, 2.0 + k * 0.3, 0.6);
        const col = [C.red, C.orange, C.Q, C.green][k];
        clockFace(ctx, x, y, 100, pos * w, col, { alpha: a });
        text(ctx, ['最快', '较快', '较慢', '最慢'][k] + '  i=' + i, x, y + 130, { size: 26, color: col, alpha: a });
      });
      text(ctx, 'pos = ' + pos.toFixed(1), 370, 300, { size: 34, family: 'main', color: C.text, alpha: S.p(2, 2.0, 0.6) });
      // 热力图
      const ha = S.p(3, 0.2, 0.8);
      const hx0 = 820, hy0 = 300, cw = 14, ch = 18, dims2 = 64, P = 26;
      text(ctx, '维度 →', hx0 + 60, hy0 - 26, { size: 24, color: C.dim, alpha: ha, align: 'left' });
      text(ctx, '位置 ↓', hx0 - 14, hy0 - 26, { size: 24, color: C.dim, alpha: ha, align: 'right' });
      for (let p = 0; p < P; p++) {
        if (p > pos + 0.5 && ha < 1) continue;
        for (let i = 0; i < dims2; i++) {
          const v = PE(p, i, dims2);
          ctx.fillStyle = T.diverge(v);
          ctx.globalAlpha = clkA * ha;
          ctx.fillRect(hx0 + i * cw, hy0 + p * ch, cw - 1, ch - 1);
        }
      }
      ctx.globalAlpha = clkA;
      const cur = Math.round(pos);
      roundRect(ctx, hx0 - 4, hy0 + cur * ch - 3, dims2 * cw + 8, ch + 4, 4, { stroke: C.Q, lw: 3, alpha: ha, glow: 12 });
      text(ctx, '每一行 = 一个位置独一无二的“指纹”', hx0 + (dims2 * cw) / 2, hy0 + P * ch + 34, { size: 30, color: C.Q, alpha: S.p(3, 4.0, 0.8) });
      ctx.restore();
    }

    // 第 4 句：相对位置 = 旋转
    const relA = S.p(4, 0.2, 0.8);
    if (relA > 0) {
      ctx.save(); ctx.globalAlpha = relA;
      const cx = 600, cy = 520, R = 220;
      const w = 0.35, k = 3;
      const p = 2 + 8 * ease(prog(t, S.L(4) + 1.5, S.E(4) - S.L(4)));
      const a1 = p * w, a2 = (p + k) * w;
      circle(ctx, cx, cy, R, { fill: 'rgba(255,255,255,0.03)', stroke: C.faint, lw: 3 });
      ctx.save(); ctx.strokeStyle = C.Q; ctx.lineWidth = 8; ctx.shadowColor = C.Q; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.45, a1 - Math.PI / 2, a2 - Math.PI / 2); ctx.stroke(); ctx.restore();
      line(ctx, cx, cy, cx + Math.sin(a1) * R * 0.85, cy - Math.cos(a1) * R * 0.85, { color: C.blue, lw: 7, glow: 10 });
      line(ctx, cx, cy, cx + Math.sin(a2) * R * 0.85, cy - Math.cos(a2) * R * 0.85, { color: C.green, lw: 7, glow: 10 });
      text(ctx, 'pos', cx + Math.sin(a1) * R * 1.12, cy - Math.cos(a1) * R * 1.12, { size: 32, family: 'main', color: C.blue });
      text(ctx, 'pos+k', cx + Math.sin(a2) * R * 1.15, cy - Math.cos(a2) * R * 1.15, { size: 32, family: 'main', color: C.green });
      const am = (a1 + a2) / 2;
      text(ctx, '两根指针的夹角始终是 kω', cx, cy + R + 50, { size: 30, color: C.Q });
      // 旋转矩阵
      const mA = S.p(4, 2.5, 0.8);
      math(ctx, M.row('\\sin', ' ', 'ω', '(', 'p', '+', 'k', ')', '=', '\\sin', ' ', 'ω', 'p', '·', '\\cos', ' ', 'ω', 'k', '+', '\\cos', ' ', 'ω', 'p', '·', '\\sin', ' ', 'ω', 'k'), 1340, 420, { size: 40, alpha: mA });
      math(ctx, M.row('\\cos', ' ', 'ω', '(', 'p', '+', 'k', ')', '=', '\\cos', ' ', 'ω', 'p', '·', '\\cos', ' ', 'ω', 'k', '−', '\\sin', ' ', 'ω', 'p', '·', '\\sin', ' ', 'ω', 'k'), 1340, 510, { size: 40, alpha: mA });
      text(ctx, '高中的“和角公式”！', 1340, 610, { size: 36, color: C.Q, alpha: S.p(4, 4.0, 0.8) });
      text(ctx, '位置 p+k 的编码 = 位置 p 的编码 “转” 一个固定角度', 1340, 680, { size: 30, color: C.dim, alpha: S.p(4, 5.0, 0.8) });
      ctx.restore();
    }
  };

  window.SCENES = Object.assign(window.SCENES || {}, scenes);
  window.SHARED.HEADS = HEADS;
})();
