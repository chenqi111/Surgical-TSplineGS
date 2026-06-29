import numpy as np
from PIL import Image
import os, glob, shutil

base = "data/nvidia_rodynrf/video_1"
H, W = 480, 640

# 1. Create gt/ directory (symlink images as v000_t{idx:03d}.png)
gt_dir = os.path.join(base, "gt")
os.makedirs(gt_dir, exist_ok=True)
for i in range(114):
    src = os.path.join(base, "images_2", f"{i:03d}.png")
    dst = os.path.join(base, "gt", f"v000_t{i:03d}.png")
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
print(f"gt/: {len(os.listdir(gt_dir))} files")

# 2. Create instance_mask/ directories with empty black mask
inst_dir = os.path.join(base, "instance_mask")
os.makedirs(inst_dir, exist_ok=True)
for i in range(114):
    subdir = os.path.join(inst_dir, f"{i:03d}")
    os.makedirs(subdir, exist_ok=True)
    mask_path = os.path.join(subdir, "000.png")
    if not os.path.exists(mask_path):
        Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(mask_path)
print(f"instance_mask/: created {len(os.listdir(inst_dir))} frame dirs")

print("Done!")
