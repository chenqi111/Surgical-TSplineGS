import numpy as np
from PIL import Image
import os

mask_dir = "data/nvidia_rodynrf/video_1/motion_masks"
os.makedirs(mask_dir, exist_ok=True)
print(f"Created dir: {mask_dir}")

H, W = 480, 640
cx, cy = W // 2, H // 2
rw, rh = int(W * 0.3), int(H * 0.3)

for i in range(114):
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[cy-rh:cy+rh, cx-rw:cx+rw] = 255
    Image.fromarray(mask).save(os.path.join(mask_dir, f"{i:03d}.png"))

files = os.listdir(mask_dir)
print(f"Created {len(files)} masks")
