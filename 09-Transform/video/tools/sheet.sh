#!/bin/bash
# 把 tools/.snaps 下的截图每 4 张拼成一张 2×2 的预览图
cd "$(dirname "$0")/.snaps"
rm -f sheet*.png
files=(t*.png); n=${#files[@]}; k=0
for ((i=0; i<n; i+=4)); do
  k=$((k+1)); set -- "${files[@]:i:4}"
  while [ $# -lt 4 ]; do set -- "$@" "$1"; done
  ffmpeg -y -loglevel error -i $1 -i $2 -i $3 -i $4 -filter_complex "[0]scale=960:540[a];[1]scale=960:540[b];[2]scale=960:540[c];[3]scale=960:540[d];[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" sheet$k.png
done
ls sheet*.png
