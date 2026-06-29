import os
import sys
import re
import numpy as np
import torch
import cv2
from argparse import ArgumentParser
from plyfile import PlyData
import json

LOG_DIR = "/home/chenqi/myworker/Surgical-TSplineGS/output"
OUTPUT_FILE = os.path.join(LOG_DIR, "figures", "eval_metrics.json")

SCENES = {
    'video_4': {
        'display_name': 'Cutting Liver',
        'log_path': os.path.join(LOG_DIR, 'video_4', 'training.log'),
        'config': 'arguments/nvidia_rodynrf/video_4.py',
        'checkpoint': os.path.join(LOG_DIR, 'video_4', 'point_cloud', 'fine_best'),
    },
    'video_3': {
        'display_name': 'Pulling Lung',
        'log_path': os.path.join(LOG_DIR, 'video_3', 'training.log'),
        'config': 'arguments/nvidia_rodynrf/video_3.py',
        'checkpoint': os.path.join(LOG_DIR, 'video_3', 'point_cloud', 'fine_best'),
    },
    'video_2': {
        'display_name': 'Pressing Heart',
        'log_path': os.path.join(LOG_DIR, 'video_2_train.log'),
        'config': 'arguments/nvidia_rodynrf/video_2.py',
        'checkpoint': os.path.join(LOG_DIR, 'video_2', 'point_cloud', 'fine_best'),
    },
}

def parse_split_stats(log_path):
    if not os.path.exists(log_path):
        return {}
    splits = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.search(r'befre clone (\d+)', line)
        if m:
            before = int(m.group(1))
            ac = as_ = None
            for j in range(i+1, min(i+5, len(lines))):
                m2 = re.search(r'after clone (\d+)', lines[j])
                if m2: ac = int(m2.group(1))
                m3 = re.search(r'after split (\d+)', lines[j])
                if m3: as_ = int(m3.group(1))
            if ac is not None and as_ is not None:
                splits.append(as_ - ac)
    total = sum(splits)
    return {
        'g_split_total': total,
        'g_split_events': len(splits),
        'g_split_per_event_avg': total / max(len(splits), 1),
    }

def compute_g_split(scene_key):
    return parse_split_stats(SCENES[scene_key]['log_path'])

def compute_tOF_between_frames(frame1, frame2):
    g1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(np.mean(mag))

def evaluate_scene(scene_key):
    info = SCENES[scene_key]
    print(f"\n{'='*60}")
    print(f"Evaluating {info['display_name']} ({scene_key})...")
    print(f"{'='*60}")

    # Parse args
    parser = ArgumentParser()
    from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    args = parser.parse_args([
        "--checkpoint", info['checkpoint'],
        "--configs", info['config'],
        "-s", f"data/nvidia_rodynrf/{scene_key}",
        "-m", LOG_DIR + f"/{scene_key}",
        "--expname", scene_key,
    ])
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    dataset = lp.extract(args)
    hyper = hp.extract(args)
    opt = op.extract(args)

    from scene.gaussian_model import GaussianModel
    from scene import Scene
    from gaussian_renderer import render_infer, render
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from utils.main_utils import get_pixels

    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())

    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    dyn_gaussians.load_ply(os.path.join(info['checkpoint'], "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(info['checkpoint'], "point_cloud_static.ply"))
    dyn_gaussians.flatten_control_point()
    dyn_gaussians.load_model(info['checkpoint'])
    dyn_gaussians._posenet.eval()

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    my_test_cams = [i for i in test_cams]
    viewpoint_stack = [i for i in train_cams]

    # Update poses
    pixels = get_pixels(
        scene.train_camera.dataset[0].metadata.image_size_x,
        scene.train_camera.dataset[0].metadata.image_size_y,
        use_center=True,
    )
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y
         ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    x = (pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x
         ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    with torch.no_grad():
        for cam in viewpoint_stack:
            time_in = torch.tensor(cam.time).float().cuda()
            pred_R, pred_T = dyn_gaussians._posenet(time_in.view(1, 1))
            R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
            t_ = pred_T.detach().cpu().numpy()
            cam.update_cam(R_[0], t_[0], local_viewdirs, batch_shape,
                           dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy())
    for view_id in range(len(my_test_cams)):
        my_test_cams[view_id].update_cam(
            viewpoint_stack[0].R, viewpoint_stack[0].T,
            local_viewdirs, batch_shape,
            dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
        )

    # ---- Use render_infer for PSNR/SSIM/tOF (fast) ----
    print(f"  Rendering {len(viewpoint_stack)} frames with render_infer...")
    psnr_vals, ssim_vals = [], []
    rendered_frames = []
    n_max = min(50, len(viewpoint_stack))

    with torch.no_grad():
        for idx, viewpoint in enumerate(viewpoint_stack[:n_max]):
            render_pkg = render_infer(viewpoint, stat_gaussians, dyn_gaussians, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_vals.append(psnr(image, gt_image).mean().double().item())
            ssim_vals.append(ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().item())
            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rendered_frames.append(img_np)
            if (idx + 1) % 10 == 0:
                print(f"    {idx+1}/{n_max} done")

    # ---- g_def: Use full render for first 3 frames ----
    print("  Computing g_def with full render (3 frames)...")
    motion_mags = []
    bg_full = torch.tensor([1,1,1,-10] if dataset.white_background else [0,0,0,-10],
                           dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for idx, viewpoint in enumerate(viewpoint_stack[:3]):
            rp = render(viewpoint, stat_gaussians, dyn_gaussians, bg_full, get_dynamic=True)
            nmag = rp.get("norm_motion_magnitude")
            if nmag is not None:
                motion_mags.append(nmag.mean().item())

    # ---- tOF and FAS ----
    print("  Computing tOF and FAS...")
    tOF_vals = []
    for i in range(1, len(rendered_frames)):
        tOF_vals.append(compute_tOF_between_frames(rendered_frames[i-1], rendered_frames[i]))

    if len(rendered_frames) > 1:
        frame_diffs = []
        for i in range(1, len(rendered_frames)):
            diff = np.mean(np.abs(rendered_frames[i].astype(float) - rendered_frames[i-1].astype(float)))
            frame_diffs.append(diff)
        fas = float(np.std(frame_diffs) / (np.mean(frame_diffs) + 1e-8))
    else:
        fas = 0.0

    # ---- Static Gaussian count (for reference) ----
    n_stat = stat_gaussions.get_xyz.shape[0] if hasattr(stat_gaussians, 'get_xyz') else 0
    n_dyn = dyn_gaussians.get_xyz.shape[0] if hasattr(dyn_gaussians, 'get_xyz') else 0

    # ---- g_split from logs ----
    g_split_data = compute_g_split(scene_key)

    results = {
        'scene': scene_key,
        'display_name': info['display_name'],
        'PSNR': float(np.mean(psnr_vals)) if psnr_vals else 0,
        'PSNR_std': float(np.std(psnr_vals)) if len(psnr_vals) > 1 else 0,
        'SSIM': float(np.mean(ssim_vals)) if ssim_vals else 0,
        'SSIM_std': float(np.std(ssim_vals)) if len(ssim_vals) > 1 else 0,
        'tOF': float(np.mean(tOF_vals)) if tOF_vals else 0,
        'g_def': float(np.mean(motion_mags)) if motion_mags else 0,
        'g_split_total': g_split_data.get('g_split_total', 0),
        'g_split_events': g_split_data.get('g_split_events', 0),
        'FAS': fas,
        'n_test_frames': len(viewpoint_stack),
        'n_dyn_gaussians': n_dyn,
        'n_stat_gaussians': n_stat,
    }

    print(f"\n  Results for {info['display_name']}:")
    print(f"    PSNR:   {results['PSNR']:.2f} ± {results['PSNR_std']:.2f}")
    print(f"    SSIM:   {results['SSIM']:.4f} ± {results['SSIM_std']:.4f}")
    print(f"    tOF:    {results['tOF']:.4f}")
    print(f"    g_def:  {results['g_def']:.4f}")
    print(f"    g_split_total: {results['g_split_total']:,}")
    print(f"    FAS:    {results['FAS']:.4f}")

    return results

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_results = {}
    for scene_key in ['video_4', 'video_3', 'video_2']:
        try:
            res = evaluate_scene(scene_key)
            all_results[scene_key] = res
        except Exception as e:
            print(f"\n  ERROR evaluating {scene_key}: {e}")
            import traceback
            traceback.print_exc()
            all_results[scene_key] = {'scene': scene_key, 'error': str(e)}

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Scene':<18} {'PSNR↑':>8} {'SSIM↑':>8} {'tOF↓':>8} {'g_def':>8} {'g_split':>10} {'FAS↓':>8}")
    print("-" * 80)
    for sk in ['video_4', 'video_3', 'video_2']:
        r = all_results.get(sk, {})
        if 'error' in r:
            print(f"{SCENES[sk]['display_name']:<18} {'ERROR':>8}")
        else:
            print(f"{r.get('display_name', sk):<18} {r.get('PSNR', 0):>8.2f} "
                  f"{r.get('SSIM', 0):>8.4f} {r.get('tOF', 0):>8.4f} "
                  f"{r.get('g_def', 0):>8.4f} {r.get('g_split_total', 0):>10,} "
                  f"{r.get('FAS', 0):>8.4f}")
