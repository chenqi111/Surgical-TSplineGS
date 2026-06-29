import numpy as np
from PIL import Image
import os, glob

inst_dir = "data/nvidia_rodynrf/pulling/instance_mask"
H, W = 256, 320

for d in sorted(os.listdir(inst_dir)):
    dpath = os.path.join(inst_dir, d)
    if os.path.isdir(dpath):
        mask_path = os.path.join(dpath, "000.png")
        if not os.path.exists(mask_path):
            mask = np.zeros((H, W), dtype=np.uint8)
            Image.fromarray(mask).save(mask_path)

print("Done: created dummy instance masks")
