#!/usr/bin/env python3
"""
合成旁白 + 生成时间轴。

  1. 用 node 读取 src/script.js 里的旁白脚本
  2. 用 sherpa-onnx + MeloTTS（离线中英混合语音模型）逐句合成语音（有缓存）
  3. 按每句真实时长排时间轴，写出 src/timing.js（动画据此同步）
  4. 拼接成完整旁白音轨 assets/narration.m4a，叠加一段很轻的环境音铺底
  5. 顺带导出字幕 assets/subtitles.srt

依赖：pip install sherpa-onnx soundfile numpy
模型：https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2
用法：python tools/build_audio.py --model-dir /path/to/vits-melo-tts-zh_en
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 44100

LEAD = 0.8   # 每一幕开头的留白（秒）
GAP = 0.35   # 句与句之间的停顿
TAIL = 0.9   # 每一幕结尾的留白

# 语音模型词典里没有的英文词，改成它认识的读法
PRONOUNCE = {
    'softmax': 'soft max',
    'ChatGPT': 'Chat G P T',
    'GPU': 'G P U',
    'ReLU': 'R E L U',
}


def pronounce(s):
    for k, v in PRONOUNCE.items():
        s = s.replace(k, v)
    return s


def load_script():
    out = subprocess.check_output(
        ['node', '-e', 'process.stdout.write(JSON.stringify(require(process.argv[1])))',
         os.path.join(ROOT, 'src', 'script.js')])
    return json.loads(out)


def make_tts(model_dir, threads):
    import sherpa_onnx
    d = model_dir
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f'{d}/model.onnx', lexicon=f'{d}/lexicon.txt',
                tokens=f'{d}/tokens.txt', dict_dir=f'{d}/dict'),
            num_threads=threads),
        rule_fsts=f'{d}/date.fst,{d}/phone.fst,{d}/number.fst')
    return sherpa_onnx.OfflineTts(cfg)


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def trim(x, thr=0.01):
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(0.03 * SR))
    b = min(len(x), idx[-1] + int(0.08 * SR))
    return x[a:b]


def ambient_pad(n):
    """一段很轻的环境音：四个和弦缓慢循环，正弦波 + 慢起慢落。n 为采样点数。"""
    seconds = n / SR
    t = np.arange(n) / SR
    chords = [  # Am7 - Fmaj7 - C - G（低音 + 三个和弦音，频率 Hz）
        [110.00, 220.00, 261.63, 329.63],
        [87.31, 174.61, 220.00, 329.63],
        [130.81, 196.00, 261.63, 329.63],
        [98.00, 196.00, 246.94, 293.66],
    ]
    seg = 8.0
    out = np.zeros_like(t)
    for k in range(int(np.ceil(seconds / seg)) + 1):
        start = k * seg - 1.0
        freqs = chords[k % 4]
        a, b = int(max(0, start) * SR), int(min(seconds, start + seg + 2.0) * SR)
        if a >= b:
            continue
        tt = t[a:b] - start
        env = np.clip(tt / 2.5, 0, 1) * np.clip((seg + 2.0 - tt) / 2.5, 0, 1)
        env = env * env * (3 - 2 * env)
        s = sum(np.sin(2 * np.pi * f * t[a:b] + i) * (0.5 if i == 0 else 0.3) for i, f in enumerate(freqs))
        s += 0.08 * np.sin(2 * np.pi * freqs[-1] * 2 * t[a:b])
        out[a:b] += s * env
    out /= np.max(np.abs(out)) + 1e-9
    fade = np.clip(t / 3, 0, 1) * np.clip((seconds - t) / 4, 0, 1)
    return (out * fade * 0.035).astype(np.float32)


def fmt_srt(s):
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{sec:02d},{ms:03d}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--cache', default=os.path.join(ROOT, 'tools', '.tts-cache'))
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--sid', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--no-music', action='store_true')
    args = ap.parse_args()

    script = load_script()
    os.makedirs(args.cache, exist_ok=True)
    tts = None

    clips = []
    for scene in script:
        for line in scene['lines']:
            say = pronounce(line.get('say', line['text']))
            key = hashlib.sha1(f'{say}|{args.speed}|{args.sid}'.encode()).hexdigest()[:16]
            path = os.path.join(args.cache, key + '.wav')
            if not os.path.exists(path):
                if tts is None:
                    tts = make_tts(args.model_dir, args.threads)
                a = tts.generate(say, sid=args.sid, speed=args.speed)
                x = resample(np.asarray(a.samples, dtype=np.float32), a.sample_rate, SR)
                x = trim(x)
                x = x / (np.max(np.abs(x)) + 1e-9) * 0.85
                sf.write(path, x, SR)
                print(f'  合成 {len(x) / SR:5.2f}s  {say[:40]}', file=sys.stderr)
            x, _ = sf.read(path, dtype='float32')
            clips.append(x)

    # ---- 排时间轴 ----
    timing = {'scenes': [], 'total': 0}
    t = 0.0
    i = 0
    srt = []
    placements = []
    for scene in script:
        s0 = t
        lt = scene.get('lead', LEAD)
        lines = []
        for line in scene['lines']:
            d = len(clips[i]) / SR
            lines.append([round(lt, 3), round(lt + d, 3)])
            placements.append((s0 + lt, clips[i]))
            srt.append((s0 + lt, s0 + lt + d, line['text']))
            lt += d + GAP + line.get('pause', 0)
            i += 1
        dur = lt - GAP + scene.get('tail', TAIL)
        timing['scenes'].append({'id': scene['id'], 'start': round(s0, 3), 'dur': round(dur, 3), 'lines': lines})
        t = s0 + dur
    timing['total'] = round(t, 3)

    with open(os.path.join(ROOT, 'src', 'timing.js'), 'w', encoding='utf-8') as f:
        f.write('// 由 tools/build_audio.py 自动生成：每一幕的开始时间、时长，以及每句旁白在该幕内的起止时间（秒）\n')
        f.write('window.TIMING = ' + json.dumps(timing, ensure_ascii=False) + ';\n')

    # ---- 混音 ----
    total = np.zeros(int((t + 0.5) * SR), dtype=np.float32)
    for start, x in placements:
        a = int(start * SR)
        total[a:a + len(x)] += x
    if not args.no_music:
        total += ambient_pad(len(total))
    total = np.clip(total, -1, 1)
    wav = os.path.join(args.cache, 'narration.wav')
    sf.write(wav, total, SR)
    out = os.path.join(ROOT, 'assets', 'narration.m4a')
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav,
                           '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11', '-ar', str(SR),
                           '-c:a', 'aac', '-b:a', '128k', out])

    with open(os.path.join(ROOT, 'assets', 'subtitles.srt'), 'w', encoding='utf-8') as f:
        for k, (a, b, txt) in enumerate(srt, 1):
            f.write(f'{k}\n{fmt_srt(a)} --> {fmt_srt(b)}\n{txt.replace("_", "")}\n\n')

    print(f'总时长 {t / 60:.1f} 分钟，{len(clips)} 句旁白 → {out}', file=sys.stderr)


if __name__ == '__main__':
    main()
