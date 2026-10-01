#!/usr/bin/env python3
"""
把字体裁剪成“只包含本项目用到的字”，再以 base64 内嵌到 src/fonts.js。
这样页面直接双击打开（file://）也能用上统一的字体，体积也只有几百 KB。

  正文：霞鹜文楷 LXGW WenKai（SIL OFL 1.1）https://github.com/lxgw/LxgwWenKai
  公式：KaTeX 字体（SIL OFL 1.1 / MIT）  npm: katex/dist/fonts

依赖：pip install fonttools brotli
用法：python tools/subset_fonts.py --wenkai LXGWWenKai-Medium.ttf --katex-dir node_modules/katex/dist/fonts
"""
import argparse
import base64
import glob
import io
import os
import string

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def used_chars():
    files = glob.glob(os.path.join(ROOT, 'src', '*.js')) + glob.glob(os.path.join(ROOT, '*.html'))
    chars = set(string.printable)
    for f in files:
        if f.endswith('fonts.js') or f.endswith('timing.js'):
            continue
        with open(f, encoding='utf-8') as fh:
            chars |= set(fh.read())
    chars |= set('，。、：；！？「」『』（）《》“”‘’…—·×÷±−∞√≈→←↑↓∑Σθωη′✓▢')
    return ''.join(sorted(c for c in chars if c.isprintable()))


def subset_to_woff2(path, text):
    font = TTFont(path)
    opts = subset.Options()
    opts.flavor = 'woff2'
    opts.layout_features = ['*']
    opts.name_IDs = ['*']
    opts.notdef_outline = True
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(font)
    buf = io.BytesIO()
    font.flavor = 'woff2'
    font.save(buf)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wenkai', required=True)
    ap.add_argument('--katex-dir', required=True)
    args = ap.parse_args()

    text = used_chars()
    out = {
        'wenkai': subset_to_woff2(args.wenkai, text),
        'katexMath': subset_to_woff2(os.path.join(args.katex_dir, 'KaTeX_Math-Italic.ttf'), text),
        'katexMain': subset_to_woff2(os.path.join(args.katex_dir, 'KaTeX_Main-Regular.ttf'), text),
    }
    with open(os.path.join(ROOT, 'src', 'fonts.js'), 'w', encoding='utf-8') as f:
        f.write('// 由 tools/subset_fonts.py 自动生成：裁剪后的字体（SIL Open Font License 1.1），base64 内嵌\n')
        f.write('window.FONT_DATA = {\n')
        for k, v in out.items():
            f.write(f'  {k}: "{base64.b64encode(v).decode()}",\n')
        f.write('};\n')
    print(f'{len(text)} 个字符 → ' + ', '.join(f'{k} {len(v) // 1024} KB' for k, v in out.items()))


if __name__ == '__main__':
    main()
