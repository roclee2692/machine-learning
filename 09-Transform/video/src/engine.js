/*
 * 动画引擎：纯 Canvas 2D，没有任何依赖。
 *
 * 核心思想：画面是时间的纯函数 —— render(t) 只依赖 t，不依赖“上一帧”。
 * 这样播放器可以随意拖动进度条，导出视频时也能逐帧精确渲染。
 */
(function () {
  const W = 1920, H = 1080;

  // ---------- 配色 ----------
  const C = {
    bg0: '#0a0f1f', bg1: '#121a33',
    text: '#eef1fb', dim: '#8d96b8', faint: '#3a4466', grid: 'rgba(120,140,200,0.08)',
    Q: '#ffc857', K: '#4fd1c5', V: '#f472b6',
    blue: '#60a5fa', green: '#34d399', red: '#f87171', purple: '#a78bfa', orange: '#fb923c',
    panel: 'rgba(255,255,255,0.045)', panelStroke: 'rgba(255,255,255,0.12)',
  };

  const FONT_CJK = '"LXGW WenKai", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", sans-serif';
  const FONT_MATH = '"KaTeX_Math", "Times New Roman", serif';
  const FONT_MAIN = '"KaTeX_Main", "Times New Roman", serif';
  const FONT_MONO = '"JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace';

  // ---------- 数学小工具 ----------
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => (t = clamp(t), t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const easeOut = (t) => (t = clamp(t), 1 - Math.pow(1 - t, 3));
  const easeIn = (t) => (t = clamp(t), t * t * t);
  const easeBack = (t) => { t = clamp(t); const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
  const prog = (t, start, dur = 1) => clamp((t - start) / dur);
  const pulse = (t, start, dur) => { const p = prog(t, start, dur); return Math.sin(p * Math.PI); };

  function softmax(xs) {
    const m = Math.max(...xs.filter((x) => x !== -Infinity));
    const e = xs.map((x) => (x === -Infinity ? 0 : Math.exp(x - m)));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  }
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, x, k) => s + x * B[k][j], 0)));
  const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));

  // 可复现的伪随机数（mulberry32），保证每次渲染画面一致
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { // Box–Muller
    const u = Math.max(1e-9, r()), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---------- 颜色 ----------
  function hexToRgb(h) {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const rgb = (c) => `rgb(${c.map((x) => Math.round(clamp(x, 0, 255))).join(',')})`;
  const rgba = (h, a) => { const c = Array.isArray(h) ? h : hexToRgb(h); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };
  const mixColor = (a, b, t) => { const A = hexToRgb(a), B = hexToRgb(b); return rgb(A.map((x, i) => lerp(x, B[i], t))); };
  // 热力图配色：深蓝 → 青 → 黄
  function heat(t) {
    t = clamp(t);
    const stops = [[20, 28, 60], [40, 90, 170], [79, 209, 197], [255, 200, 87], [255, 245, 200]];
    const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
    return rgb(stops[i].map((v, k) => lerp(v, stops[i + 1][k], f)));
  }
  // 正负值配色：负 = 蓝，零 = 暗，正 = 橙
  function diverge(v) {
    v = clamp(v, -1, 1);
    const z = [28, 34, 60];
    const pos = [255, 170, 80], neg = [80, 150, 255];
    const c = v >= 0 ? pos : neg;
    return rgb(z.map((x, i) => lerp(x, c[i], Math.abs(v))));
  }

  // ---------- 基础绘图 ----------
  function font(size, opts = {}) {
    const fam = opts.family === 'math' ? FONT_MATH : opts.family === 'main' ? FONT_MAIN : opts.family === 'mono' ? FONT_MONO : FONT_CJK;
    return `${opts.italic ? 'italic ' : ''}${opts.weight || 500} ${size}px ${fam}`;
  }

  function text(ctx, str, x, y, o = {}) {
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    ctx.font = font(o.size || 36, o);
    ctx.textAlign = o.align || 'center';
    ctx.textBaseline = o.baseline || 'middle';
    if (o.glow) { ctx.shadowColor = o.glowColor || o.color || C.text; ctx.shadowBlur = o.glow; }
    if (o.stroke) { ctx.lineWidth = o.strokeWidth || 6; ctx.strokeStyle = o.stroke; ctx.lineJoin = 'round'; ctx.strokeText(str, x, y); }
    ctx.fillStyle = o.color || C.text;
    if (o.maxWidth) ctx.fillText(str, x, y, o.maxWidth); else ctx.fillText(str, x, y);
    ctx.restore();
  }
  function measure(ctx, str, o = {}) {
    ctx.save(); ctx.font = font(o.size || 36, o); const w = ctx.measureText(str).width; ctx.restore(); return w;
  }

  // 逐字出现的文字（打字机效果）
  function typeText(ctx, str, x, y, p, o = {}) {
    const chars = Array.from(str);
    const n = Math.floor(chars.length * clamp(p));
    text(ctx, chars.slice(0, n).join(''), x, y, o);
    return n;
  }

  function roundRect(ctx, x, y, w, h, r, o = {}) {
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    ctx.beginPath();
    r = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    if (o.glow) { ctx.shadowColor = o.glowColor || o.stroke || o.fill; ctx.shadowBlur = o.glow; }
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
    ctx.shadowBlur = 0;
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.lw || 2; if (o.dash) ctx.setLineDash(o.dash); ctx.stroke(); }
    ctx.restore();
  }

  function circle(ctx, x, y, r, o = {}) {
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    ctx.beginPath(); ctx.arc(x, y, Math.max(0, r), 0, Math.PI * 2);
    if (o.glow) { ctx.shadowColor = o.glowColor || o.fill || o.stroke; ctx.shadowBlur = o.glow; }
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
    ctx.shadowBlur = 0;
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.lw || 2; ctx.stroke(); }
    ctx.restore();
  }

  function line(ctx, x1, y1, x2, y2, o = {}) {
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    ctx.strokeStyle = o.color || C.text; ctx.lineWidth = o.lw || 2; ctx.lineCap = 'round';
    if (o.dash) ctx.setLineDash(o.dash);
    if (o.glow) { ctx.shadowColor = o.color; ctx.shadowBlur = o.glow; }
    const p = o.progress == null ? 1 : o.progress;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(lerp(x1, x2, p), lerp(y1, y2, p)); ctx.stroke();
    ctx.restore();
  }

  function arrowHead(ctx, x, y, ang, size, color) {
    ctx.save();
    ctx.translate(x, y); ctx.rotate(ang);
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-size, size * 0.55); ctx.lineTo(-size * 0.75, 0); ctx.lineTo(-size, -size * 0.55); ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function arrow(ctx, x1, y1, x2, y2, o = {}) {
    const p = o.progress == null ? 1 : clamp(o.progress);
    if (p <= 0) return;
    const ex = lerp(x1, x2, p), ey = lerp(y1, y2, p);
    const ang = Math.atan2(y2 - y1, x2 - x1);
    const hs = o.head || 16;
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    if (o.glow) { ctx.shadowColor = o.color; ctx.shadowBlur = o.glow; }
    line(ctx, x1, y1, ex - Math.cos(ang) * hs * 0.6, ey - Math.sin(ang) * hs * 0.6, { color: o.color, lw: o.lw || 3, dash: o.dash });
    arrowHead(ctx, ex, ey, ang, hs, o.color || C.text);
    ctx.restore();
  }

  // 二次贝塞尔弧线箭头，bend 为弯曲程度（正负决定方向）
  function arcArrow(ctx, x1, y1, x2, y2, bend, o = {}) {
    const p = o.progress == null ? 1 : clamp(o.progress);
    if (p <= 0) return;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const cx = mx - (dy / len) * bend, cy = my + (dx / len) * bend;
    const q = (t) => [
      (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * cx + t * t * x2,
      (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * cy + t * t * y2,
    ];
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    ctx.strokeStyle = o.color || C.text; ctx.lineWidth = o.lw || 3; ctx.lineCap = 'round';
    if (o.glow) { ctx.shadowColor = o.color; ctx.shadowBlur = o.glow; }
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath();
    const N = 40;
    for (let i = 0; i <= N; i++) {
      const [x, y] = q((i / N) * p * (o.head === 0 ? 1 : 0.97));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    if (o.head !== 0) {
      const [ax, ay] = q(p), [bx, by] = q(Math.max(0, p - 0.03));
      ctx.setLineDash([]);
      arrowHead(ctx, ax, ay, Math.atan2(ay - by, ax - bx), o.head || 14, o.color || C.text);
    }
    ctx.restore();
    return q;
  }

  // 一个“词卡片”
  function chip(ctx, str, x, y, o = {}) {
    const size = o.size || 40;
    const w = o.w || measure(ctx, str, { size }) + size * 1.0;
    const h = o.h || size * 1.6;
    const s = o.scale == null ? 1 : o.scale;
    ctx.save();
    ctx.translate(x, y); ctx.scale(s, s);
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    roundRect(ctx, -w / 2, -h / 2, w, h, o.r || 14, {
      fill: o.fill || 'rgba(255,255,255,0.06)', stroke: o.stroke || 'rgba(255,255,255,0.22)', lw: o.lw || 2, glow: o.glow,
    });
    text(ctx, str, 0, 2, { size, color: o.color || C.text, weight: o.weight || 600 });
    ctx.restore();
    return { w, h };
  }

  // 向量画成一排小格子（颜色表示正负大小），可选显示数字
  function vecCells(ctx, v, x, y, o = {}) {
    const cs = o.cell || 34, gap = o.gap == null ? 4 : o.gap, vertical = !!o.vertical;
    const n = v.length;
    const total = n * cs + (n - 1) * gap;
    const reveal = o.reveal == null ? 1 : o.reveal;
    const scale = o.range || 1;
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    for (let i = 0; i < n; i++) {
      const a = clamp(reveal * n - i);
      if (a <= 0) continue;
      const px = vertical ? x - cs / 2 : x - total / 2 + i * (cs + gap);
      const py = vertical ? y - total / 2 + i * (cs + gap) : y - cs / 2;
      const fill = o.color ? rgba(o.color, 0.18 + 0.75 * clamp(Math.abs(v[i]) / scale)) : diverge(v[i] / scale);
      roundRect(ctx, px, py, cs, cs, Math.min(8, cs * 0.2), { fill, alpha: a, stroke: o.stroke || 'rgba(255,255,255,0.18)', lw: 1.5 });
      if (o.nums) {
        const s = typeof o.nums === 'function' ? o.nums(v[i]) : fmtNum(v[i], o.digits);
        text(ctx, s, px + cs / 2, py + cs / 2 + 1, { size: o.numSize || cs * 0.42, color: o.numColor || C.text, alpha: a, family: 'main', weight: 400 });
      }
    }
    ctx.restore();
    return total;
  }
  function fmtNum(x, digits = 1) {
    if (x === -Infinity) return '−∞';
    const s = Math.abs(x) < 1e-9 ? '0' : x.toFixed(digits).replace(/\.?0+$/, '') || '0';
    return s.replace('-', '−');
  }

  // 矩阵（网格）
  function grid(ctx, M, x, y, o = {}) {
    const rows = M.length, cols = M[0].length;
    const cw = o.cell || 40, ch = o.cellH || cw, gap = o.gap == null ? 3 : o.gap;
    const tw = cols * cw + (cols - 1) * gap, th = rows * ch + (rows - 1) * gap;
    const x0 = x - tw / 2, y0 = y - th / 2;
    const reveal = o.reveal == null ? 1 : o.reveal;
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const k = o.byCol ? c * rows + r : r * cols + c;
        const a = clamp(reveal * rows * cols - k);
        if (a <= 0) continue;
        const v = M[r][c];
        const fill = o.fill ? o.fill(v, r, c) : diverge(v / (o.range || 1));
        const px = x0 + c * (cw + gap), py = y0 + r * (ch + gap);
        roundRect(ctx, px, py, cw, ch, Math.min(6, cw * 0.15), { fill, alpha: a * (o.cellAlpha ? o.cellAlpha(r, c) : 1), stroke: o.stroke || 'rgba(255,255,255,0.12)', lw: 1 });
        if (o.nums) {
          const s = typeof o.nums === 'function' ? o.nums(v, r, c) : fmtNum(v, o.digits);
          text(ctx, s, px + cw / 2, py + ch / 2 + 1, { size: o.numSize || Math.min(cw, ch) * 0.4, color: o.numColor || C.text, alpha: a * (o.cellAlpha ? o.cellAlpha(r, c) : 1), family: 'main', weight: 400 });
        }
      }
    }
    ctx.restore();
    return { x0, y0, w: tw, h: th, cw, ch, gap };
  }

  // 方括号（画矩阵 / 向量外框）
  function brackets(ctx, x, y, w, h, o = {}) {
    const k = o.k || 12;
    ctx.save();
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    ctx.strokeStyle = o.color || C.dim; ctx.lineWidth = o.lw || 3;
    ctx.beginPath();
    ctx.moveTo(x + k, y); ctx.lineTo(x, y); ctx.lineTo(x, y + h); ctx.lineTo(x + k, y + h);
    ctx.moveTo(x + w - k, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - k, y + h);
    ctx.stroke();
    ctx.restore();
  }

  // ---------- 迷你数学排版 ----------
  // 节点：字符串 | {t:'row',c:[]} | {t:'sub',b,s} | {t:'sup',b,s} | {t:'frac',n,d} | {t:'sqrt',c} | {t:'paren',c}
  // 字符串约定：以 "\" 开头为直立正体（如 "\softmax"），其余拉丁字母用数学斜体；中文自动用正文字体。
  // 任何节点都可以带 id，绘制时通过 style(id) 改颜色 / 透明度。
  const M = {
    row: (...c) => ({ t: 'row', c }),
    sub: (b, s) => ({ t: 'sub', b, s }),
    sup: (b, s) => ({ t: 'sup', b, s }),
    frac: (n, d) => ({ t: 'frac', n, d }),
    sqrt: (c) => ({ t: 'sqrt', c }),
    paren: (...c) => ({ t: 'paren', c: { t: 'row', c } }),
    id: (id, node) => (typeof node === 'string' ? { t: 'str', s: node, id } : Object.assign({}, node, { id })),
  };

  function atomFont(s, size) {
    if (/[一-鿿]/.test(s)) return font(size * 0.92, { weight: 500 });
    if (s.startsWith('\\')) return font(size, { family: 'main', weight: 400 });
    if (/^[A-Za-z]+$/.test(s)) return font(size, { family: 'math', italic: false, weight: 400 });
    return font(size, { family: 'main', weight: 400 });
  }
  function atomText(s) { return s.startsWith('\\') ? s.slice(1) : s; }
  const OPS = new Set(['=', '+', '−', '×', '·', '≈', ',', '→']);

  function layout(ctx, node, size) {
    if (typeof node === 'string') node = { t: 'str', s: node };
    if (node.t === 'str') {
      ctx.font = atomFont(node.s, size);
      const str = atomText(node.s);
      const pad = OPS.has(str) ? size * (str === ',' ? 0.08 : 0.25) : 0;
      const w = ctx.measureText(str).width + pad * 2;
      return { node, w, asc: size * 0.72, desc: size * 0.22, size, pad, str };
    }
    if (node.t === 'row') {
      const kids = node.c.map((c) => layout(ctx, c, size));
      return { node, kids, w: kids.reduce((s, k) => s + k.w, 0), asc: Math.max(...kids.map((k) => k.asc)), desc: Math.max(...kids.map((k) => k.desc)), size };
    }
    if (node.t === 'sub' || node.t === 'sup') {
      const b = layout(ctx, node.b, size), s = layout(ctx, node.s, size * 0.62);
      const shift = node.t === 'sub' ? size * 0.22 : -size * 0.42;
      return { node, b, s, shift, w: b.w + s.w + size * 0.04, asc: Math.max(b.asc, node.t === 'sup' ? -shift + s.asc : 0), desc: Math.max(b.desc, node.t === 'sub' ? shift + s.desc : 0), size };
    }
    if (node.t === 'frac') {
      const n = layout(ctx, node.n, size * 0.9), d = layout(ctx, node.d, size * 0.9);
      const w = Math.max(n.w, d.w) + size * 0.4, axis = size * 0.28;
      return { node, n, d, w, axis, asc: axis + size * 0.12 + n.desc + n.asc, desc: -axis + size * 0.12 + d.asc + d.desc, size };
    }
    if (node.t === 'sqrt') {
      const c = layout(ctx, node.c, size);
      return { node, c, w: c.w + size * 0.75, asc: c.asc + size * 0.16, desc: c.desc + size * 0.06, size };
    }
    if (node.t === 'paren') {
      const c = layout(ctx, node.c, size);
      const pw = size * 0.38;
      return { node, c, pw, w: c.w + pw * 2, asc: c.asc + size * 0.06, desc: c.desc + size * 0.06, size };
    }
    throw new Error('bad node ' + node.t);
  }

  function drawLayout(ctx, L, x, y, style, inherited) {
    const st = Object.assign({ color: C.text, alpha: 1 }, inherited || {}, (L.node.id && style && style(L.node.id)) || {});
    if (st.alpha <= 0) return;
    ctx.save();
    ctx.globalAlpha *= st.alpha;
    const sub = { color: st.color, alpha: 1, glow: st.glow };
    if (st.glow) { ctx.shadowColor = st.color; ctx.shadowBlur = st.glow; }
    const t = L.node.t;
    if (t === 'str') {
      ctx.font = atomFont(L.node.s, L.size);
      ctx.fillStyle = st.color; ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
      ctx.fillText(L.str, x + L.pad, y);
    } else if (t === 'row') {
      let cx = x;
      for (const k of L.kids) { drawLayout(ctx, k, cx, y, style, sub); cx += k.w; }
    } else if (t === 'sub' || t === 'sup') {
      drawLayout(ctx, L.b, x, y, style, sub);
      drawLayout(ctx, L.s, x + L.b.w + L.size * 0.04, y + L.shift, style, sub);
    } else if (t === 'frac') {
      const ay = y - L.axis;
      ctx.fillStyle = st.color;
      ctx.fillRect(x + L.size * 0.1, ay - L.size * 0.03, L.w - L.size * 0.2, L.size * 0.06);
      drawLayout(ctx, L.n, x + (L.w - L.n.w) / 2, ay - L.size * 0.14 - L.n.desc, style, sub);
      drawLayout(ctx, L.d, x + (L.w - L.d.w) / 2, ay + L.size * 0.14 + L.d.asc, style, sub);
    } else if (t === 'sqrt') {
      const s = L.size, top = y - L.asc + s * 0.04, bot = y + L.desc - s * 0.02;
      ctx.strokeStyle = st.color; ctx.lineWidth = s * 0.055; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x + s * 0.06, y - s * 0.2);
      ctx.lineTo(x + s * 0.2, y - s * 0.28);
      ctx.lineTo(x + s * 0.38, bot);
      ctx.lineTo(x + s * 0.62, top);
      ctx.lineTo(x + L.w - s * 0.04, top);
      ctx.stroke();
      drawLayout(ctx, L.c, x + s * 0.68, y, style, sub);
    } else if (t === 'paren') {
      const top = y - L.asc, bot = y + L.desc, h = bot - top, pw = L.pw;
      ctx.strokeStyle = st.color; ctx.lineWidth = L.size * 0.055; ctx.lineCap = 'round';
      const bow = Math.min(pw * 0.75, h * 0.2);
      ctx.beginPath();
      ctx.moveTo(x + pw * 0.85, top); ctx.quadraticCurveTo(x + pw * 0.85 - bow * 1.6, (top + bot) / 2, x + pw * 0.85, bot);
      const rx = x + L.w - pw * 0.85;
      ctx.moveTo(rx, top); ctx.quadraticCurveTo(rx + bow * 1.6, (top + bot) / 2, rx, bot);
      ctx.stroke();
      drawLayout(ctx, L.c, x + pw, y, style, sub);
    }
    ctx.restore();
  }

  // 返回布局信息，可以用 boxes 拿到某个 id 节点的位置（用于画指示框）
  function math(ctx, node, x, y, o = {}) {
    const size = o.size || 60;
    ctx.save();
    const L = layout(ctx, node, size);
    let x0 = x;
    if (o.align === 'center' || o.align == null) x0 = x - L.w / 2;
    else if (o.align === 'right') x0 = x - L.w;
    const y0 = o.baseline === 'middle' || o.baseline == null ? y + (L.asc - L.desc) / 2 : y;
    ctx.globalAlpha *= o.alpha == null ? 1 : o.alpha;
    drawLayout(ctx, L, x0, y0, o.style, { color: o.color || C.text, alpha: 1 });
    ctx.restore();
    const boxes = {};
    (function walk(l, lx, ly) {
      if (l.node.id) boxes[l.node.id] = { x: lx, y: ly - l.asc, w: l.w, h: l.asc + l.desc };
      if (l.node.t === 'row') { let cx = lx; for (const k of l.kids) { walk(k, cx, ly); cx += k.w; } }
      else if (l.node.t === 'sub' || l.node.t === 'sup') { walk(l.b, lx, ly); walk(l.s, lx + l.b.w, ly + l.shift); }
      else if (l.node.t === 'frac') { const ay = ly - l.axis; walk(l.n, lx + (l.w - l.n.w) / 2, ay - l.size * 0.14 - l.n.desc); walk(l.d, lx + (l.w - l.d.w) / 2, ay + l.size * 0.14 + l.d.asc); }
      else if (l.node.t === 'sqrt') walk(l.c, lx + l.size * 0.68, ly);
      else if (l.node.t === 'paren') walk(l.c, lx + l.pw, ly);
    })(L, x0, y0);
    return { w: L.w, h: L.asc + L.desc, x: x0, top: y0 - L.asc, boxes };
  }

  // ---------- 背景与装饰 ----------
  let bgCache = null;
  function background(ctx) {
    if (!bgCache) {
      const c = document.createElement('canvas'); c.width = W; c.height = H;
      const g = c.getContext('2d');
      const grd = g.createRadialGradient(W * 0.5, H * 0.35, 100, W * 0.5, H * 0.5, W * 0.75);
      grd.addColorStop(0, C.bg1); grd.addColorStop(1, C.bg0);
      g.fillStyle = grd; g.fillRect(0, 0, W, H);
      g.strokeStyle = C.grid; g.lineWidth = 1;
      for (let x = 0; x <= W; x += 60) { g.beginPath(); g.moveTo(x + 0.5, 0); g.lineTo(x + 0.5, H); g.stroke(); }
      for (let y = 0; y <= H; y += 60) { g.beginPath(); g.moveTo(0, y + 0.5); g.lineTo(W, y + 0.5); g.stroke(); }
      const v = g.createRadialGradient(W / 2, H / 2, H * 0.4, W / 2, H / 2, W * 0.72);
      v.addColorStop(0, 'rgba(0,0,0,0)'); v.addColorStop(1, 'rgba(0,0,0,0.55)');
      g.fillStyle = v; g.fillRect(0, 0, W, H);
      bgCache = c;
    }
    ctx.drawImage(bgCache, 0, 0);
  }

  function panel(ctx, x, y, w, h, o = {}) {
    roundRect(ctx, x, y, w, h, o.r || 24, { fill: o.fill || C.panel, stroke: o.stroke || C.panelStroke, lw: o.lw || 1.5, alpha: o.alpha, glow: o.glow });
  }

  // 字幕：支持 "W_Q" 这类下标写法，长句自动折成两行
  function richRuns(str) {
    const runs = [];
    const re = /_([A-Za-z0-9])/g;
    let last = 0, m;
    while ((m = re.exec(str))) {
      if (m.index > last) runs.push({ s: str.slice(last, m.index) });
      runs.push({ s: m[1], sub: true });
      last = m.index + m[0].length;
    }
    if (last < str.length) runs.push({ s: str.slice(last) });
    return runs;
  }
  function richWidth(ctx, runs, size) {
    let w = 0;
    for (const r of runs) { ctx.font = font(r.sub ? size * 0.65 : size, { weight: 500 }); w += ctx.measureText(r.s).width; }
    return w;
  }
  function richDraw(ctx, runs, x, y, size) {
    for (const r of runs) {
      const s = r.sub ? size * 0.65 : size;
      ctx.font = font(s, { weight: 500 });
      ctx.fillText(r.s, x, y + (r.sub ? size * 0.22 : 0));
      x += ctx.measureText(r.s).width;
    }
  }
  function wrapLines(ctx, str, size, maxW) {
    if (richWidth(ctx, richRuns(str), size) <= maxW) return [str];
    // 优先在标点处断开，使两行长度接近
    const chars = Array.from(str);
    let best = null;
    for (let i = 4; i < chars.length - 3; i++) {
      if (!/[，。：；、！？,]/.test(chars[i - 1])) continue;
      const a = chars.slice(0, i).join(''), b = chars.slice(i).join('');
      const wa = richWidth(ctx, richRuns(a), size), wb = richWidth(ctx, richRuns(b), size);
      if (wa > maxW || wb > maxW) continue;
      const score = Math.abs(wa - wb);
      if (!best || score < best.score) best = { a, b, score };
    }
    if (best) return [best.a, best.b];
    const half = Math.ceil(chars.length / 2);
    return [chars.slice(0, half).join(''), chars.slice(half).join('')];
  }

  function subtitle(ctx, str, alpha) {
    if (!str || alpha <= 0) return;
    const size = 40, maxW = 1500;
    ctx.save();
    ctx.globalAlpha = alpha;
    const lines = wrapLines(ctx, str, size, maxW);
    const lh = size * 1.45;
    const widths = lines.map((l) => richWidth(ctx, richRuns(l), size));
    const bw = Math.max(...widths) + 64, bh = lines.length * lh + 26;
    const by = H - 52 - bh;
    roundRect(ctx, W / 2 - bw / 2, by, bw, bh, 18, { fill: 'rgba(6,9,20,0.72)', stroke: 'rgba(255,255,255,0.08)', lw: 1 });
    ctx.fillStyle = '#f5f7ff'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    lines.forEach((l, i) => richDraw(ctx, richRuns(l), W / 2 - widths[i] / 2, by + 13 + lh * (i + 0.5) + 2, size));
    ctx.restore();
  }

  window.TV = {
    W, H, C, FONT_CJK, FONT_MATH, FONT_MAIN,
    clamp, lerp, ease, easeOut, easeIn, easeBack, prog, pulse,
    softmax, dot, matmul, transpose, rng, gauss,
    hexToRgb, rgb, rgba, mixColor, heat, diverge,
    font, text, measure, typeText, roundRect, circle, line, arrow, arrowHead, arcArrow, chip, vecCells, grid, brackets, fmtNum,
    M, math, background, panel, subtitle,
  };
})();
