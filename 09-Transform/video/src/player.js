/*
 * 播放器：把旁白时间轴、分镜和音频组合起来。
 *   - 普通模式：带进度条 / 章节 / 字幕开关 / 倍速的播放器，以音频时间为主时钟
 *   - 渲染模式（?render=1）：只留 1920×1080 画布，供 tools/render.mjs 逐帧导出视频
 */
(function () {
  const T = window.TV;
  const { W, H, C } = T;
  const params = new URLSearchParams(location.search);
  const RENDER = params.has('render');

  // ---------- 时间轴 ----------
  function buildTimeline() {
    const timing = window.TIMING;
    let t = 0;
    return window.SCRIPT.map((sc, k) => {
      let start, dur, lines;
      const tm = timing && timing.scenes[k] && timing.scenes[k].id === sc.id ? timing.scenes[k] : null;
      if (tm) ({ start, dur, lines } = tm);
      else { // 没有合成语音时，按每秒约 6 个字估算
        let lt = sc.lead || 0.8;
        lines = sc.lines.map((l) => { const d = Array.from(l.text).length / 6; const r = [lt, lt + d]; lt += d + 0.35 + (l.pause || 0); return r; });
        start = t; dur = lt - 0.35 + (sc.tail || 0.9);
      }
      t = start + dur;
      return { id: sc.id, title: sc.title, start, dur, lines, texts: sc.lines.map((l) => l.text) };
    });
  }
  const TL = buildTimeline();
  const TOTAL = TL[TL.length - 1].start + TL[TL.length - 1].dur;

  function sceneAt(t) {
    for (let k = TL.length - 1; k >= 0; k--) if (t >= TL[k].start) return k;
    return 0;
  }

  function makeS(sc, lt) {
    const L = (i) => sc.lines[Math.min(i, sc.lines.length - 1)][0];
    const E = (i) => sc.lines[Math.min(i, sc.lines.length - 1)][1];
    return {
      t: lt, dur: sc.dur, L, E,
      at: (i, f) => L(i) + (E(i) - L(i)) * f,
      p: (i, off = 0, len = 0.8) => T.ease(T.prog(lt, L(i) + off, len)),
      pl: (i, off = 0, len = 0.8) => T.prog(lt, L(i) + off, len),
    };
  }

  let showSubs = params.get('subs') !== '0';

  function renderAt(ctx, t) {
    t = T.clamp(t, 0, TOTAL - 1e-3);
    const k = sceneAt(t);
    const sc = TL[k];
    const lt = t - sc.start;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    T.background(ctx);
    const fn = window.SCENES[sc.id];
    try {
      if (fn) fn(ctx, makeS(sc, lt));
    } catch (e) {
      console.error(e);
      T.text(ctx, '渲染出错：' + sc.id + ' ' + e.message, W / 2, H / 2, { size: 30, color: C.red });
    }
    ctx.restore();

    // 章节标题
    if (k > 0 || lt > 31) {
      const a = T.prog(lt, 0.1, 0.6) * (1 - T.prog(lt, sc.dur - 0.5, 0.4));
      ctx.save(); ctx.globalAlpha = a;
      T.roundRect(ctx, 40, 34, T.measure(ctx, sc.title, { size: 30 }) + 48, 56, 28, { fill: 'rgba(255,255,255,0.06)', stroke: 'rgba(255,255,255,0.12)', lw: 1 });
      T.text(ctx, sc.title, 64, 63, { size: 30, align: 'left', color: C.text });
      ctx.restore();
    }
    T.text(ctx, '什么是 Transformer？', W - 50, 63, { size: 26, align: 'right', color: C.faint });

    // 场景切换：淡入淡出
    const fade = Math.min(T.prog(lt, 0, 0.5), 1 - T.prog(lt, sc.dur - 0.45, 0.45));
    if (fade < 1) {
      ctx.save(); ctx.globalAlpha = 1 - T.easeOut(fade); ctx.fillStyle = C.bg0; ctx.fillRect(0, 0, W, H); ctx.restore();
    }

    // 字幕
    if (showSubs) {
      let sub = null, sa = 0;
      sc.lines.forEach(([a, b], i) => {
        if (lt >= a - 0.1 && lt <= b + 0.35) { sub = sc.texts[i]; sa = Math.min(T.prog(lt, a - 0.1, 0.2), 1 - T.prog(lt, b + 0.15, 0.2)); }
      });
      if (sub) T.subtitle(ctx, sub, sa);
    }
    return { scene: k, local: lt };
  }

  // ---------- 字体 ----------
  async function loadFonts() {
    const data = window.FONT_DATA || {};
    const defs = [['LXGW WenKai', data.wenkai], ['KaTeX_Math', data.katexMath], ['KaTeX_Main', data.katexMain]];
    const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer;
    await Promise.all(defs.filter((d) => d[1]).map(async ([name, src]) => {
      try { const f = new FontFace(name, b64(src)); await f.load(); document.fonts.add(f); } catch (e) { console.warn('字体加载失败', name, e); }
    }));
  }

  // ---------- 渲染模式 ----------
  if (RENDER) {
    document.documentElement.classList.add('render');
    const cv = document.getElementById('stage');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    window.__duration = TOTAL;
    window.__renderAt = (t) => { renderAt(ctx, t); };
    window.__frame = async (t, type, q) => {
      renderAt(ctx, t);
      const blob = await new Promise((r) => cv.toBlob(r, type || 'image/png', q));
      const buf = new Uint8Array(await blob.arrayBuffer());
      let s = ''; const CH = 0x8000;
      for (let i = 0; i < buf.length; i += CH) s += String.fromCharCode.apply(null, buf.subarray(i, i + CH));
      return btoa(s);
    };
    loadFonts().then(() => { renderAt(ctx, 0); window.__ready = true; });
    return;
  }

  // ---------- 播放器界面 ----------
  const cv = document.getElementById('stage');
  const ctx = cv.getContext('2d');
  const audio = document.getElementById('narration');
  const $ = (id) => document.getElementById(id);
  const playBtn = $('play'), seek = $('seek'), timeLbl = $('time'), chapters = $('chapters'), subsBtn = $('subs'), speedSel = $('speed'), fsBtn = $('fs');
  let hasAudio = false, playing = false, clock = 0, lastNow = 0;

  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const box = cv.getBoundingClientRect();
    cv.width = Math.round(box.width * dpr); cv.height = Math.round(box.height * dpr);
  }
  window.addEventListener('resize', resize);
  resize();

  // 章节刻度与列表
  const ticks = $('ticks');
  TL.forEach((sc, k) => {
    const tick = document.createElement('i');
    tick.style.left = (sc.start / TOTAL) * 100 + '%';
    ticks.appendChild(tick);
    const li = document.createElement('button');
    li.className = 'chapter';
    li.innerHTML = `<span>${sc.title}</span><em>${fmt(sc.start)}</em>`;
    li.onclick = () => { setTime(sc.start + 0.01); if (!playing) play(); };
    chapters.appendChild(li);
  });

  function fmt(s) { s = Math.max(0, s); return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0'); }
  function now() { return hasAudio ? audio.currentTime : clock; }
  function setTime(t) {
    t = T.clamp(t, 0, TOTAL);
    clock = t;
    if (hasAudio) audio.currentTime = t;
    draw();
  }
  function play() {
    if (now() >= TOTAL - 0.05) setTime(0);
    playing = true; playBtn.classList.add('on'); playBtn.setAttribute('aria-label', '暂停');
    lastNow = performance.now();
    if (hasAudio) audio.play().catch(() => { hasAudio = false; });
    document.body.classList.add('playing');
  }
  function pause() {
    playing = false; playBtn.classList.remove('on'); playBtn.setAttribute('aria-label', '播放');
    if (hasAudio) audio.pause();
    document.body.classList.remove('playing');
  }
  playBtn.onclick = () => (playing ? pause() : play());
  cv.onclick = () => (playing ? pause() : play());
  seek.addEventListener('input', () => setTime((seek.value / 1000) * TOTAL));
  subsBtn.onclick = () => { showSubs = !showSubs; subsBtn.classList.toggle('on', showSubs); draw(); };
  subsBtn.classList.toggle('on', showSubs);
  speedSel.onchange = () => { audio.playbackRate = +speedSel.value; };
  fsBtn.onclick = () => { const el = $('player'); document.fullscreenElement ? document.exitFullscreen() : el.requestFullscreen && el.requestFullscreen(); };
  document.addEventListener('fullscreenchange', () => setTimeout(() => { resize(); draw(); }, 50));
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); playing ? pause() : play(); }
    else if (e.code === 'ArrowRight') setTime(now() + 5);
    else if (e.code === 'ArrowLeft') setTime(now() - 5);
    else if (e.key === 'c' || e.key === 'C') subsBtn.onclick();
    else if (e.key === 'f' || e.key === 'F') fsBtn.onclick();
  });
  audio.addEventListener('loadedmetadata', () => { hasAudio = true; audio.currentTime = clock; });
  audio.addEventListener('error', () => { hasAudio = false; });
  audio.addEventListener('ended', pause);
  if (audio.readyState >= 1) hasAudio = true;

  let lastScene = -1;
  // 画面始终先画到 1920×1080 的离屏画布，再缩放到屏幕上
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const offCtx = off.getContext('2d');
  function draw() {
    const t = now();
    const info = renderAt(offCtx, t);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, cv.width, cv.height);
    seek.value = Math.round((t / TOTAL) * 1000);
    seek.style.setProperty('--p', (t / TOTAL) * 100 + '%');
    timeLbl.textContent = fmt(t) + ' / ' + fmt(TOTAL);
    if (info.scene !== lastScene) {
      [...chapters.children].forEach((el, i) => el.classList.toggle('active', i === info.scene));
      lastScene = info.scene;
    }
  }
  function loop(ts) {
    if (playing && !hasAudio) {
      clock += ((ts - lastNow) / 1000) * (+speedSel.value || 1);
      if (clock >= TOTAL) { clock = TOTAL; pause(); }
    }
    lastNow = ts;
    draw();
    requestAnimationFrame(loop);
  }
  loadFonts().then(() => { $('loading').remove(); draw(); requestAnimationFrame(loop); });

  if (params.has('t')) setTime(+params.get('t'));
})();
