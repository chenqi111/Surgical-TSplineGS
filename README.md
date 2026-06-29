<div><h2>Surgical-TSplineGS: Topology-Aware Motion-Adaptive Splines for Real-Time Dynamic 3D Reconstruction in Monocular Endoscopy</h2></div>
<br>

**Qi Chen, Beihang University**

![Teaser figure](figure/2.png)

**Surgical-TSplineGS** is a Topology-Aware Motion-Adaptive Splines for Real-Time Dynamic 3D Reconstruction in Monocular Endoscopy. It extends 4D Gaussian Splatting with:

- **Topology-Aware Spline Splitting (TASS)** — detects photometric error spikes, splits continuous splines into child trajectories to model tissue cutting.
- **Mask-Guided Motion-Adaptive Spline (MG-MAS)** — uses tool masks to freeze gradients in occluded regions, decoupling rigid instrument motion from tissue deformation.
- **Cubic Hermite Spline Interpolation** — Smooth trajectory modeling with motion-adaptive control points (K=6 for surgical, K=12 for general).
- **Motion Type Classification** — Automatically classifies Gaussians as static / tissue / instrument for targeted optimization.
- **Dual-Environment Pipeline** — Uses UniDepth (metric depth) and Depth-Anything (disparity) with CoTracker3 (long-range point tracks) for robust geometric initialization.

## ⚙️ Environmental Setups

Clone the repo and install dependencies (two conda environments required):

```sh
git clone https://github.com/chenqi111/Surgical-TSplineGS.git --recursive
cd Surgical-TSplineGS

# === Environment 1: Main training env (Python 3.7, CUDA 11.7, PyTorch 1.13.1) ===
conda create -n surgical_tsplinegs python=3.7
conda activate surgical_tsplinegs
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install nvidia/label/cuda-11.7.0::cuda
conda install nvidia/label/cuda-11.7.0::cuda-nvcc
conda install nvidia/label/cuda-11.7.0::cuda-runtime
conda install nvidia/label/cuda-11.7.0::cuda-cudart

pip install -e submodules/simple-knn
pip install -e submodules/co-tracker
pip install -r requirements.txt

# === Environment 2: Depth estimation env (Python 3.10, CUDA 12.1, PyTorch 2.4) ===
conda deactivate
conda create -n unidepth_surgical_tsplinegs python=3.10
conda activate unidepth_surgical_tsplinegs

pip install -r requirements_unidepth.txt
conda install -c conda-forge ld_impl_linux-64
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda install nvidia/label/cuda-12.1.0::cuda
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
conda install nvidia/label/cuda-12.1.0::cuda-runtime
conda install nvidia/label/cuda-12.1.0::cuda-cudart
conda install nvidia/label/cuda-12.1.0::libcusparse
conda install nvidia/label/cuda-12.1.0::libcublas
cd submodules/UniDepth/unidepth/ops/knn; bash compile.sh; cd ../../../../../
cd submodules/UniDepth/unidepth/ops/extract_patches; bash compile.sh; cd ../../../../../

pip install -e submodules/UniDepth
mkdir -p submodules/mega-sam/Depth-Anything/checkpoints
```

Alternatively, use the automated installation script:
```sh
bash install.sh
```

### Download Depth-Anything Checkpoint

```sh
# Download depth_anything_vitl14.pth and place at:
# submodules/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth \
  -O submodules/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth
```
## 📁 Data Preparations

### EndoNeRF Dataset

1. Download the training images from the [release page](https://github.com/med-air/EndoNeRF?tab=readme-ov-file) 


Each scene should contain:
- `images_2/` — Downscaled (2x) input images
- `instance_masks/` — Per-frame instance masks (surgical instruments)
- `motion_masks/` — Motion masks from [Shape of Motion](https://github.com/vye16/shape-of-motion/) 
- `gt/` — Ground truth images (symlinked from images_2)

2. Generate depth maps and point tracks:
```sh
# Depth estimation (UniDepth + Depth-Anything)
conda activate unidepth_surgical_tsplinegs
bash gen_depth.sh

conda deactivate
conda activate surgical_tsplinegs

# Long-range point tracks (CoTracker3)
bash gen_tracks.sh
```

### Custom Surgical Dataset

1. Prepare your video frames under `data/{dataset_name}/images_2/`.
2. Create instance masks (optional, for instrument masking) in `data/{dataset_name}/instance_masks/` — each frame subdirectory `{:03d}/` containing `000.png`.
3. Generate depth maps:
```sh
conda activate unidepth_surgical_tsplinegs
python gen_depth.py --image_dir data/{dataset_name}/images_2 --out_dir data/{dataset_name}/uni_depth
python gen_depth.py --image_dir data/{dataset_name}/images_2 --out_dir data/{dataset_name}/depth_anything --depth_type disp
```
4. Generate point tracks:
```sh
conda activate surgical_tsplinegs
python gen_tracks.py --image_dir data/{dataset_name}/images_2 --out_dir data/{dataset_name}/bootscotracker_dynamic --grid_size 256
python gen_tracks.py --image_dir data/{dataset_name}/images_2 --out_dir data/{dataset_name}/bootscotracker_static --grid_size 50 --is_static
```
5. Create a configuration file at `arguments/{dataset_name}/{scene}.py` inheriting from `arguments/surgical/default.py`.
6. Create motion masks (optional): use `create_masks.py` for dummy masks or generate using Shape of Motion.
## 🚀 Training

<!-- ### Nvidia Dataset -->

```sh
conda activate surgical_tsplinegs

# Train a single scene (use run_train.sh wrapper to set env)
python train.py -s data/nvidia_rodynrf/pulling/ --expname "Pulling" --configs arguments/nvidia_rodynrf/pulling.py

# Or via the env wrapper:
bash run_train.sh train.py -s data/nvidia_rodynrf/pulling/ --expname "Pulling" --configs arguments/nvidia_rodynrf/pulling.py

# Batch train all 7 Nvidia scenes
bash train.sh
```

### Custom Surgical Dataset

```sh
conda activate surgical_tsplinegs

# Use the surgical training wrapper
bash train_surgical.sh /path/to/scene my_experiment_name

# Or manually:
python train.py -s /path/to/scene/ --expname "my_experiment" \
    --configs arguments/surgical/default.py \
    --dataset_type surgical
```

### Key Training Arguments

| Argument | Description |
|----------|-------------|
| `-s` | Path to scene directory (containing `images_2/`) |
| `--expname` | Experiment name for output directory |
| `--configs` | Path to scene-specific `.py` config file |
| `--dataset_type` | `nvidia` (default) or `surgical` |
| `--depth_type` | `depth` (UniDepth metric) or `disp` (Depth-Anything disparity) |

Training produces checkpoints in `output/{expname}/`.

## 📊 Evaluation

### EndoNeRF Dataset (Full Metrics)

```sh
conda activate surgical_tsplinegs

# Evaluate a single scene
python eval_nvidia.py -s data/nvidia_rodynrf/pulling/ \
    --expname "Pulling" \
    --configs arguments/nvidia_rodynrf/pulling.py \
    --checkpoint output/Pulling/point_cloud/fine_best

# Batch evaluate all scenes
bash eval.sh
```

### Automated Surgical Metrics

```sh
# Comprehensive metric computation (PSNR, SSIM, tOF, g_def, g_split, FAS)
bash run_eval.sh

# MASE (Mean Absolute Scaled Error) metric
python compute_mase.py
```

### Rendered Visualization

```sh
# Render heatmaps (motion magnitude, rigidity, velocity consistency, motion types)
python gen_heatmap.py --checkpoint output/{expname}/point_cloud/fine_best

# Render Gaussian split events from training logs
python gen_split_figures.py --log logs/{expname}.log
```

## 📁 Project Structure

```
Surgical-TSplineGS/
├── arguments/               # Configuration files (scene-specific .py)
│   ├── nvidia_rodynrf/      #   Nvidia dataset configs (18 scenes)
│   └── surgical/            #   Surgical dataset defaults
├── scene/                   # Scene representation
│   ├── gaussian_model.py    #   GaussianModel with densification, control points
│   ├── deformation.py       #   Deformation network, pose network
│   ├── dataset.py           #   PyTorch Dataset (FourDGSdataset)
│   └── cameras.py           #   Camera module with pose & time embedding
├── gaussian_renderer/       # Differentiable rasterization (Hermite interpolation, MG-MAS)
│   ├── __init__.py          #   Render, render_infer, motion classification
├── dycheck_geometry/        # Adobe DyCheck camera geometry utilities
├── submodules/
│   ├── simple-knn/          # KNN distance (Gaussian initialization)
│   ├── co-tracker/          # CoTracker3 (long-range point tracks)
│   ├── mega-sam/            # Mega-SAM + Depth-Anything
│   └── UniDepth/            # Universal monocular depth estimation
├── utils/                   # Utility modules (losses, graphics, params)
├── gen_depth.py             # Depth map generation
├── gen_tracks.py            # CoTracker3 point track generation
├── train.py                 # Main training loop (1496 lines)
├── eval_nvidia.py           # Nvidia dataset evaluation
├── compute_metrics.py       # Automated surgical metric computation
└── compute_mase.py          # MASE metric computation
```

<!-- ## ⭐ Citing Surgical-TSplineGS

If you find our repository useful, please consider giving it a star ⭐ and citing our research papers in your work:
```bibtex
@misc{Chen_2025_Surgical_TSplineGS,
    author    = {Chen, Qi},
    title     = {Surgical-TSplineGS: Topology-Aware Motion-Adaptive Splines for Real-Time Dynamic 3D Reconstruction in Monocular Endoscopy},
    year      = {2026}
}

