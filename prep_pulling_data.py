import numpy as np
from PIL import Image
import os
import shutil

src = "data/pulling"
dst = "data/nvidia_rodynrf/pulling"
os.makedirs(dst, exist_ok=True)

# 1. Create images_2/ (rename and resize to 2x downscale)
# Original: 640x512 in images/, images_4/ is 160x128 (4x)
# images_2 should be 320x256 (2x)
src_img_dir = os.path.join(src, "images")
dst_img_dir = os.path.join(dst, "images_2")
os.makedirs(dst_img_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith('.png')])
print(f"Found {len(image_files)} images")

for i, fname in enumerate(image_files):
    img = Image.open(os.path.join(src_img_dir, fname)).convert('RGB')
    # Resize to 2x downscale (320x256)
    img_resized = img.resize((320, 256), Image.LANCZOS)
    new_name = f"{i:03d}.png"
    img_resized.save(os.path.join(dst_img_dir, new_name))

print(f"Created images_2/ with {len(image_files)} resized images")

# 2. Create gt/ (symlink/copy for test ground truth)
gt_dir = os.path.join(dst, "gt")
os.makedirs(gt_dir, exist_ok=True)
for i in range(len(image_files)):
    src_path = os.path.join(dst_img_dir, f"{i:03d}.png")
    dst_path = os.path.join(gt_dir, f"v000_t{i:03d}.png")
    shutil.copy2(src_path, dst_path)

print(f"Created gt/ with {len(image_files)} test images")

# 3. Create empty instance_mask/ directories
inst_dir = os.path.join(dst, "instance_mask")
os.makedirs(inst_dir, exist_ok=True)
for i in range(len(image_files)):
    os.makedirs(os.path.join(inst_dir, f"{i:03d}"), exist_ok=True)

print(f"Created instance_mask/ with {len(image_files)} empty dirs")

# 4. Create dummy motion_masks (all zeros / no motion)
mask_dir = os.path.join(dst, "motion_masks")
os.makedirs(mask_dir, exist_ok=True)
H, W = 256, 320  # images_2 resolution
for i in range(len(image_files)):
    mask = np.zeros((H, W), dtype=np.uint8)
    Image.fromarray(mask).save(os.path.join(mask_dir, f"{i:03d}.png"))

print(f"Created motion_masks/ with {len(image_files)} zero masks")

# 5. Create dummy points3D.ply placeholder
from plyfile import PlyData, PlyElement
import numpy as np

ply_path = os.path.join(dst, "dummy_points3D.ply")
xyz = np.zeros((1, 3), dtype=np.float32)
rgb = np.zeros((1, 3), dtype=np.uint8)
times = np.zeros((1, 1), dtype=np.float32)
points = np.zeros(1, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
points['x'] = 0; points['y'] = 0; points['z'] = 0
vertex_element = PlyElement.describe(points, 'vertex')
PlyData([vertex_element]).write(ply_path)

print(f"Created dummy_points3D.ply")
print("Data preparation complete!")
