#!/bin/bash
set -e

BASE="data/nvidia_rodynrf/video_7"
H=480
W=640

echo "=== Step 1: images_2/ ==="
mkdir -p "$BASE/images_2"
for f in "$BASE/images"/image*.png; do
    basename=$(basename "$f")
    num=$(echo "$basename" | sed 's/image0*//;s/\.png//')
    newname=$(printf "%03d.png" $((10#$num - 1)))
    if [ ! -f "$BASE/images_2/$newname" ]; then
        cp "$f" "$BASE/images_2/$newname"
    fi
done
echo "images_2: $(ls "$BASE/images_2" | wc -l) files"

echo "=== Step 2: gt/ ==="
mkdir -p "$BASE/gt"
for f in "$BASE/images_2"/*.png; do
    basename=$(basename "$f" .png)
    num=$((10#$basename))
    newname=$(printf "v000_t%03d.png" $num)
    if [ ! -f "$BASE/gt/$newname" ]; then
        cp "$f" "$BASE/gt/$newname"
    fi
done
echo "gt: $(ls "$BASE/gt" | wc -l) files"

echo "=== Step 3: instance_mask/ ==="
for f in "$BASE/images_2"/*.png; do
    basename=$(basename "$f" .png)
    subdir="$BASE/instance_mask/$basename"
    mkdir -p "$subdir"
    if [ ! -f "$subdir/000.png" ]; then
        python3 -c "
from PIL import Image
import numpy as np
Image.fromarray(np.zeros(($H, $W), dtype=np.uint8)).save('$subdir/000.png')
"
    fi
done
echo "instance_mask: $(ls "$BASE/instance_mask" | wc -l) dirs"

echo "=== Setup complete ==="
