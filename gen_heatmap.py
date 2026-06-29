import os
import sys
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.main_utils import get_pixels
from utils.graphics_utils import getWorld2View2


def apply_colormap(gray_8u, cmap="3class"):
    if cmap == "gray":
        return cv2.cvtColor(gray_8u, cv2.COLOR_GRAY2BGR)
    elif cmap == "jet":
        return cv2.applyColorMap(gray_8u, cv2.COLORMAP_JET)
    elif cmap == "3class":
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            v = i / 255.0
            if v < 0.15:
                lut[i] = [10, 10, 10]
            elif v < 0.45:
                ratio = (v - 0.15) / 0.30
                r = int(10 + ratio * 20)
                g = int(10 + ratio * 230)
                b = int(10 + ratio * 50)
                lut[i] = [r, g, b]
            elif v < 0.75:
                ratio = (v - 0.45) / 0.30
                r = int(30 + ratio * 220)
                g = int(240 - ratio * 190)
                b = int(60 - ratio * 20)
                lut[i] = [r, g, b]
            else:
                lut[i] = [255, 50, 60]
        return lut[gray_8u]
    elif cmap == "distinct":
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            v = i / 255.0
            if v < 0.15:
                lut[i] = [20, 20, 20]
            elif v < 0.45:
                lut[i] = [30, 220, 70]
            elif v < 0.75:
                lut[i] = [50, 100, 250]
            else:
                lut[i] = [250, 60, 70]
        return lut[gray_8u]
    else:
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 64:
                lut[i] = [0, i * 4, 255]
            elif i < 128:
                lut[i] = [0, 255, 255 - (i - 64) * 4]
            elif i < 192:
                lut[i] = [(i - 128) * 4, 255, 0]
            else:
                lut[i] = [255, 255 - (i - 192) * 4, 0]
        return lut[gray_8u]


def detect_class_contours(class_map, rgb_gray):
    """Detect clear contours at class boundaries with adaptive thresholding."""
    kernel = np.ones((3, 3), np.uint8)

    class_grad_x = cv2.Sobel(class_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    class_grad_y = cv2.Sobel(class_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    class_edges = np.sqrt(class_grad_x ** 2 + class_grad_y ** 2) > 0.01

    rgb_edges = cv2.Canny(rgb_gray, 20, 60)
    rgb_edges = cv2.dilate(rgb_edges, kernel, iterations=1)

    combined = cv2.bitwise_or((class_edges * 255).astype(np.uint8), rgb_edges)
    combined = cv2.dilate(combined, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    return combined


def create_sharp_contour_heatmap(motion_type_values, rgb_bgr, colormap="3class"):
    """
    Create a sharp 3-class heatmap with clear contour boundaries.
    Expects motion_type_values to be label map with values 0.0, 0.5, 1.0
    (or 0.25, 0.5 for dyn-only).
    """
    h, w = motion_type_values.shape

    class_map = np.zeros((h, w), dtype=np.uint8)
    class_map[motion_type_values < 0.15] = 0
    class_map[(motion_type_values >= 0.15) & (motion_type_values < 0.75)] = 1
    class_map[motion_type_values >= 0.75] = 2

    class_map = cv2.medianBlur(class_map.astype(np.float32), 5).astype(np.uint8)

    refined_values = np.zeros_like(motion_type_values)
    refined_values[class_map == 0] = 0.0
    refined_values[class_map == 1] = 0.5
    refined_values[class_map == 2] = 1.0

    refined_8u = (np.clip(refined_values * 255, 0, 255)).astype("uint8")
    colored = apply_colormap(refined_8u, colormap)

    rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    edges = detect_class_contours(class_map, rgb_gray)

    contour_color = [255, 255, 255]
    colored_with_contours = colored.copy()
    colored_with_contours[edges > 0] = contour_color

    return colored_with_contours, refined_values


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate split Gaussian ratio heatmap")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument("--checkpoint", type=str, default="output/video_6/point_cloud/fine_best")
    parser.add_argument("--expname", type=str, default="video_6")
    parser.add_argument("--configs", type=str, default="arguments/nvidia_rodynrf/video_6.py")
    parser.add_argument("--output_dir", type=str, default="output/video_6/heatmap")
    parser.add_argument("--colormap", type=str, default="distinct", choices=["redblue", "jet", "gray", "3class", "distinct"])

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    dataset = lp.extract(args)
    hyper = hp.extract(args)
    opt = op.extract(args)
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)

    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())

    bg_color = [1, 1, 1, -10] if dataset.white_background else [0, 0, 0, -10]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    viewpoint_stack = [i for i in train_cams]
    my_test_cams = [i for i in test_cams]

    dyn_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    dyn_gaussians.load_model(args.checkpoint)
    dyn_gaussians._posenet.eval()

    pixels = get_pixels(
        scene.train_camera.dataset[0].metadata.image_size_x,
        scene.train_camera.dataset[0].metadata.image_size_y,
        use_center=True,
    )
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (
        pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y
    ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    x = (
        pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x
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

    os.makedirs(args.output_dir, exist_ok=True)

    all_cams = [("train", viewpoint_stack), ("test", my_test_cams)]

    with torch.no_grad():
        for split_name, cam_list in all_cams:
            split_dir = os.path.join(args.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for idx, viewpoint in enumerate(cam_list):
                render_pkg = render(
                    viewpoint, stat_gaussians, dyn_gaussians, background,
                    get_dynamic=True,
                )
                image = render_pkg["render"]
                d_alpha = render_pkg["d_alpha"]

                rgb = torch.clamp(image, 0.0, 1.0)
                rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                rgb_img = Image.fromarray(rgb_np)
                rgb_img.save(os.path.join(split_dir, f"render_{idx:04d}.png"))

                rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

                heatmap = d_alpha.squeeze().cpu().numpy()
                heatmap_8u = (np.clip(heatmap * 255, 0, 255)).astype("uint8")
                heatmap_colored = apply_colormap(heatmap_8u, args.colormap)
                cv2.imwrite(
                    os.path.join(split_dir, f"dyn_ratio_{idx:04d}.png"),
                    heatmap_colored,
                )

                heatmap_jet = apply_colormap(heatmap_8u, "jet")
                cv2.imwrite(
                    os.path.join(split_dir, f"dyn_ratio_jet_{idx:04d}.png"),
                    heatmap_jet,
                )

                overlay = cv2.addWeighted(rgb_bgr, 0.6, heatmap_colored, 0.4, 0)
                cv2.imwrite(
                    os.path.join(split_dir, f"overlay_{idx:04d}.png"),
                    overlay,
                )
                overlay_jet = cv2.addWeighted(rgb_bgr, 0.6, heatmap_jet, 0.4, 0)
                cv2.imwrite(
                    os.path.join(split_dir, f"overlay_jet_{idx:04d}.png"),
                    overlay_jet,
                )

                nc = render_pkg["nc_heatmap"].squeeze().cpu().numpy()
                nc_max = max(nc.max(), 1)
                nc_norm = np.clip(nc / nc_max, 0, 1)
                nc_8u = (nc_norm * 255).astype("uint8")
                nc_colored = apply_colormap(nc_8u, args.colormap)
                cv2.imwrite(
                    os.path.join(split_dir, f"nc_{idx:04d}.png"),
                    nc_colored,
                )

                nc_jet = apply_colormap(nc_8u, "jet")
                cv2.imwrite(
                    os.path.join(split_dir, f"nc_jet_{idx:04d}.png"),
                    nc_jet,
                )

                nc_overlay = cv2.addWeighted(rgb_bgr, 0.6, nc_colored, 0.4, 0)
                cv2.imwrite(
                    os.path.join(split_dir, f"nc_overlay_{idx:04d}.png"),
                    nc_overlay,
                )
                nc_overlay_jet = cv2.addWeighted(rgb_bgr, 0.6, nc_jet, 0.4, 0)
                cv2.imwrite(
                    os.path.join(split_dir, f"nc_overlay_jet_{idx:04d}.png"),
                    nc_overlay_jet,
                )

                # ---- Motion type heatmap with sharp contours ----
                motion_type = render_pkg["motion_type"].squeeze().cpu().numpy()
                mt_sharp_colored, mt_refined = create_sharp_contour_heatmap(
                    motion_type, rgb_bgr, colormap=args.colormap
                )
                cv2.imwrite(
                    os.path.join(split_dir, f"motion_type_{idx:04d}.png"),
                    mt_sharp_colored,
                )

                mt_smooth_colored = apply_colormap(
                    (np.clip(motion_type * 255, 0, 255)).astype("uint8"),
                    args.colormap,
                )
                mt_overlay = cv2.addWeighted(rgb_bgr, 0.5, mt_sharp_colored, 0.5, 0)
                cv2.imwrite(
                    os.path.join(split_dir, f"motion_type_overlay_{idx:04d}.png"),
                    mt_overlay,
                )

                # ---- Dynamic-only motion type with sharp contours ----
                dyn_mt = render_pkg["dyn_motion_type"]
                if dyn_mt is not None:
                    dyn_mt_np = dyn_mt.squeeze().cpu().numpy()
                    dmt_sharp_colored, _ = create_sharp_contour_heatmap(
                        dyn_mt_np, rgb_bgr, colormap=args.colormap
                    )
                    cv2.imwrite(
                        os.path.join(split_dir, f"dyn_motion_type_{idx:04d}.png"),
                        dmt_sharp_colored,
                    )
                    dmt_smooth = apply_colormap(
                        (np.clip(dyn_mt_np * 255, 0, 255)).astype("uint8"),
                        args.colormap,
                    )
                    dmt_overlay = cv2.addWeighted(rgb_bgr, 0.6, dmt_smooth, 0.4, 0)
                    cv2.imwrite(
                        os.path.join(split_dir, f"dyn_motion_type_overlay_{idx:04d}.png"),
                        dmt_overlay,
                    )

                print(f"[{split_name}] {idx}/{len(cam_list)} - saved heatmap")

    print(f"Done. Heatmaps saved to {args.output_dir}")
