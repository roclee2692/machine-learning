#!/usr/bin/env node
/*
 * 把 index.html 逐帧渲染成 MP4（无头 Chromium + ffmpeg）。
 *
 *   node tools/render.js                       # 完整视频 → 什么是Transformer.mp4
 *   node tools/render.js --fps 30 --workers 4  # 多个浏览器页面并行渲染
 *   node tools/render.js --snap 12,40.5,90     # 只导出这几个时间点的截图（调试用）
 *
 * 依赖：npm i -g playwright（使用已安装的 Chromium）、ffmpeg
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

function loadPlaywright() {
  try { return require('playwright'); } catch (e) {}
  const { execSync } = require('child_process');
  const root = execSync('npm root -g').toString().trim();
  return require(path.join(root, 'playwright'));
}
const { chromium } = loadPlaywright();

const ROOT = path.resolve(__dirname, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]);
  return acc;
}, []));
const FPS = +(args.fps || 30);
const WORKERS = +(args.workers || 4);
const OUT = path.resolve(args.out || path.join(ROOT, '什么是Transformer.mp4'));
const URL = 'file://' + path.join(ROOT, 'index.html') + '?render=1';

async function openPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('pageerror', (e) => console.error('页面错误:', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.error('控制台:', m.text()); });
  await page.goto(URL);
  await page.waitForFunction('window.__ready === true', null, { timeout: 60000 });
  return page;
}

async function snap(browser, times) {
  const page = await openPage(browser);
  const dir = path.resolve(args.dir || path.join(ROOT, 'tools', '.snaps'));
  fs.mkdirSync(dir, { recursive: true });
  for (const t of times) {
    const b64 = await page.evaluate(([t]) => window.__frame(t, 'image/png'), [t]);
    const f = path.join(dir, `t${String(t.toFixed(1)).padStart(6, '0')}.png`);
    fs.writeFileSync(f, Buffer.from(b64, 'base64'));
    console.log(f);
  }
}

async function renderRange(browser, from, to, segPath, onFrame) {
  const page = await openPage(browser);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-tune', 'animation', '-pix_fmt', 'yuv420p', '-r', String(FPS), segPath], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', (c) => (c === 0 ? res() : rej(new Error('ffmpeg 退出码 ' + c)))));
  for (let f = from; f < to; f++) {
    const b64 = await page.evaluate(([t]) => window.__frame(t, 'image/jpeg', 0.95), [f / FPS]);
    const buf = Buffer.from(b64, 'base64');
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    onFrame();
  }
  ff.stdin.end();
  await done;
  await page.close();
}

(async () => {
  const browser = await chromium.launch({ args: ['--font-render-hinting=none', '--disable-gpu-vsync'] });
  try {
    if (args.snap) {
      await snap(browser, String(args.snap).split(',').map(Number));
      return;
    }
    const probe = await openPage(browser);
    const duration = await probe.evaluate('window.__duration');
    await probe.close();
    const total = Math.ceil(duration * FPS);
    const per = Math.ceil(total / WORKERS);
    const tmp = path.join(ROOT, 'tools', '.render');
    fs.mkdirSync(tmp, { recursive: true });
    let doneFrames = 0;
    const t0 = Date.now();
    const tick = () => {
      doneFrames++;
      if (doneFrames % (FPS * 10) === 0 || doneFrames === total) {
        const el = (Date.now() - t0) / 1000;
        process.stdout.write(`\r  ${doneFrames}/${total} 帧  ${(doneFrames / el).toFixed(1)} 帧/秒  剩余约 ${Math.round((total - doneFrames) / (doneFrames / el))} 秒   `);
      }
    };
    console.log(`时长 ${duration.toFixed(1)} 秒，共 ${total} 帧，${WORKERS} 路并行`);
    const segs = [];
    await Promise.all(Array.from({ length: WORKERS }, (_, w) => {
      const a = w * per, b = Math.min(total, (w + 1) * per);
      const seg = path.join(tmp, `seg${w}.mp4`);
      segs.push(seg);
      return renderRange(browser, a, b, seg, tick);
    }));
    console.log('\n合并视频与音频…');
    const list = path.join(tmp, 'list.txt');
    fs.writeFileSync(list, segs.map((s) => `file '${s}'`).join('\n'));
    const audio = path.join(ROOT, 'assets', 'narration.m4a');
    const ffArgs = ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list];
    if (fs.existsSync(audio)) ffArgs.push('-i', audio, '-c:a', 'copy', '-shortest');
    ffArgs.push('-c:v', 'copy', '-movflags', '+faststart', '-metadata', 'title=什么是 Transformer？', OUT);
    await new Promise((res, rej) => spawn('ffmpeg', ffArgs, { stdio: 'inherit' }).on('close', (c) => (c === 0 ? res() : rej(new Error('合并失败')))));
    fs.rmSync(tmp, { recursive: true, force: true });
    console.log('完成 →', OUT);
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error(e); process.exit(1); });
