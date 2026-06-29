import os
import numpy as np
from PIL import Image
import torch

gt_dir = "data/nvidia_rodynrf/video_1/gt"
val_render_dir = "output/video_1/fine_render/val/images"

num_frames = 114
H, W = 480, 640

# Load ground truth images (all frames)
gts = []
for idx in range(num_frames):
    gt_path = os.path.join(gt_dir, f"v000_t{idx:03d}.png")
    gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
    gts.append(gt)
gts = np.stack(gts, axis=0)  # (T, H, W, 3)

# Load rendered images (extract second panel)
renders = []
for idx in range(num_frames):
    render_path = os.path.join(val_render_dir, f"v000_t{idx:03d}.jpg")
    render_full = np.array(Image.open(render_path)).astype(np.float32) / 255.0
    render_panel = render_full[:, W:2*W, :]  # second panel is the render
    renders.append(render_panel)
renders = np.stack(renders, axis=0)  # (T, H, W, 3)

# Mean Absolute Error of reconstruction
mae_recon = np.mean(np.abs(renders - gts))

# Mean Absolute Error of naive forecast (previous frame)
# For t=1..T-1, naive prediction = GT_{t-1}
mae_naive = np.mean(np.abs(gts[1:] - gts[:-1]))

mase = mae_recon / mae_naive

print(f"Video 1 Reconstruction MASE: {mase:.6f}")
print(f"  MAE (reconstruction): {mae_recon:.6f}")
print(f"  MAE (naive forecast): {mae_naive:.6f}")
print(f"  Total frames: {num_frames}")
print(f"  Image size: {W}x{H}")
