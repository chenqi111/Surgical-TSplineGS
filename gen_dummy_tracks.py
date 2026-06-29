import numpy as np
import os, glob

image_dir = "data/nvidia_rodynrf/video_1/images_2"
out_dirs = [
    "data/nvidia_rodynrf/video_1/bootscotracker_dynamic",
    "data/nvidia_rodynrf/video_1/bootscotracker_static",
]

frame_names = sorted(glob.glob(os.path.join(image_dir, "*.png")))
num_frames = len(frame_names)

# Read first image to get dimensions
from PIL import Image
img = Image.open(frame_names[0])
W, H = img.size

print(f"{num_frames} frames, {W}x{H}")

grid_size = 64
grid_x = np.linspace(0, W-1, grid_size, dtype=np.float32)
grid_y = np.linspace(0, H-1, grid_size, dtype=np.float32)
xx, yy = np.meshgrid(grid_x, grid_y)
points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)
N = points.shape[0]
print(f"Grid {grid_size}x{grid_size} = {N} points")

for out_dir in out_dirs:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)

for t in range(num_frames):
    name_t = f"{t:03d}"
    for out_dir in out_dirs:
        for j in range(num_frames):
            name_j = f"{j:03d}"
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                track = np.zeros((N, 4), dtype=np.float32)
                track[:, 0] = points[:, 0]  # x
                track[:, 1] = points[:, 1]  # y
                track[:, 2] = 1.0  # visibility
                track[:, 3] = 1.0  # confidence
                np.save(out_path, track)
        self_track = np.zeros((N, 4), dtype=np.float32)
        self_track[:, 0] = points[:, 0]
        self_track[:, 1] = points[:, 1]
        self_track[:, 2] = 1.0
        self_track[:, 3] = 1.0
        np.save(f"{out_dir}/{name_t}_{name_t}.npy", self_track)

print("Done! Created dummy tracks for all frame pairs.")
