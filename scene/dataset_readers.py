#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import glob
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import cv2
import dycheck_geometry
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement
from scene.colmap_loader import qvec2rotmat, read_points3D_binary, read_points3D_text
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, getWorld2View2


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float
    max_time: float
    mask: np.array
    sem_mask: Optional[np.array] = None
    metadata: Optional[dycheck_geometry.Camera] = None
    normal: Optional[np.array] = None
    depth: Optional[np.array] = None
    fwd_flow: Optional[np.array] = None
    bwd_flow: Optional[np.array] = None
    fwd_flow_mask: Optional[np.array] = None
    bwd_flow_mask: Optional[np.array] = None
    instance_mask: Optional[np.array] = None
    # tracklet: Optional[np.array] = None
    target_tracks: Optional[np.array] = None
    target_visibility: Optional[np.array] = None
    target_tracks_static: Optional[np.array] = None
    target_visibility_static: Optional[np.array] = None



class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    return {"translate": translate, "radius": radius}


def get_normals(z, camera_metadata):
    pixels = camera_metadata.get_pixels()
    y = (pixels[..., 1] - camera_metadata.principal_point_y) / camera_metadata.scale_factor_y
    x = (
        pixels[..., 0] - camera_metadata.principal_point_x - y * camera_metadata.skew
    ) / camera_metadata.scale_factor_x
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    viewdirs = torch.from_numpy(viewdirs).to(z.device)

    coords = viewdirs[None] * z[..., None]
    coords = coords.permute(0, 3, 1, 2)

    dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
    dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
    dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
    dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
    dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
    dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

    dxdu = torch.nn.functional.pad(dxdu, (0, 1), mode="replicate")
    dydu = torch.nn.functional.pad(dydu, (0, 1), mode="replicate")
    dzdu = torch.nn.functional.pad(dzdu, (0, 1), mode="replicate")

    dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
    dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
    dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

    n_x = dydv * dzdu - dydu * dzdv
    n_y = dzdv * dxdu - dzdu * dxdv
    n_z = dxdv * dydu - dxdu * dydv

    pred_normal = torch.stack([n_x, n_y, n_z], dim=-3)
    pred_normal = torch.nn.functional.normalize(pred_normal, dim=-3)
    return pred_normal


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image, None)
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            time=float(idx / len(cam_extrinsics)),
            mask=None,
        )  # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    times = np.vstack([vertices["t"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)


def storePly(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("t", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def read_flow(flow_path, img_size):
    flow_info = np.load(flow_path)
    flow, mask = flow_info["flow"], flow_info["mask"]

    H, W, _ = flow.shape
    flow[..., 0] = flow[..., 0] / W
    flow[..., 1] = flow[..., 1] / H

    flow = cv2.resize(flow, (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask.astype(np.float32), (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
    return flow, mask


def load_target_tracks(tracks_dir, query_index, target_indices, dim=1, scale=1.0):
    """
    tracks are 2d, occs and uncertainties
    :param dim (int), default 1: dimension to stack the time axis
    return (N, T, 4) if dim=1, (T, N, 4) if dim=0
    """
    q_name = str(query_index).zfill(3)
    all_tracks = []
    for ti in target_indices:
        t_name = str(ti).zfill(3)
        path = f"{tracks_dir}/{q_name}_{t_name}.npy"
        tracks = np.load(path).astype(np.float32)
        tracks[:, :2] = tracks[:, :2] / scale
        all_tracks.append(tracks)
    return torch.from_numpy(np.stack(all_tracks, axis=dim))


def parse_tapir_track_info(occlusions, expected_dist):
    """
    return:
        valid_visible: mask of visible & confident points
        valid_invisible: mask of invisible & confident points
        confidence: clamped confidence scores (all < 0.5 -> 0)
    """
    visiblility = 1 - F.sigmoid(occlusions)
    confidence = 1 - F.sigmoid(expected_dist)
    valid_visible = visiblility * confidence > 0.5
    valid_invisible = (1 - visiblility) * confidence > 0.5
    # set all confidence < 0.5 to 0
    confidence = confidence * (valid_visible | valid_invisible).float()
    return valid_visible, valid_invisible, confidence


def normalize_coords(coords, h, w):
    assert coords.shape[-1] == 2
    return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0

def get_tracks(path, num_frames, is_static=False):
    target_tracks_all = []
    for idx in range(num_frames):
        tracks_dir = os.path.join(path,"bootscotracker_static" if is_static else "bootscotracker_dynamic")
        target_inds_all = torch.from_numpy(np.arange(num_frames))
        target_tracks_all.append(load_target_tracks(tracks_dir, idx, target_inds_all.tolist(), dim=0))
    target_tracks_all = torch.cat(target_tracks_all, dim=1)
    
    if target_tracks_all.size(1) > 30000:
        indices = torch.linspace(0, target_tracks_all.size(1) - 1, 30000).long()
        target_tracks_all = target_tracks_all[:, indices, :]

    target_tracks = target_tracks_all[:, :, :2]
    target_visibility = target_tracks_all[:, :, 2:]

    valid_index = target_visibility[:,:,0].all(dim=0) == True
    valid_target_tracks = target_tracks[:, valid_index, :]
    return valid_target_tracks

def readNvidiaCameras(args):
    path = args.source_path
    train_cam_infos, test_cam_infos = [], []
    image_list = sorted(glob.glob(os.path.join(path, "images_2/*.png")))
    img_0 = cv2.imread(image_list[0])
    sh = img_0.shape[:2]
    focal_length = 500 # dummy
    max_time = len(image_list) - 1
    normal_dir = os.path.join(path, "normal")
    if not os.path.exists(normal_dir):
        os.mkdir(normal_dir)


    # read dynamic track
    num_frames = max_time + 1
    target_tracks = get_tracks(path, num_frames)
    target_tracks_static = get_tracks(path, num_frames, is_static=True)

    depth_list = []
    for idx in range(num_frames):
        frame_name = f"{idx:03d}.png"
        depth_path = os.path.join(path, "uni_depth", frame_name.replace(".png", ".npy"))
        depth = np.load(depth_path)
        depth_list.append(depth)
    mean_depth = np.mean(np.stack(depth_list, 0))
    
    if args.depth_type == "disp":
        disp_list = []
        for idx in range(max_time+1):
            frame_name = f"{idx:03d}.png"
            disp_path = os.path.join(path, "depth_anything", frame_name.replace(".png", ".npy"))
            metric_depth = depth_list[idx]
            da_disp = np.float32(np.load(disp_path))[..., None]
            gt_disp = 1.0 / (metric_depth + 1e-8)
            
            valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
            gt_disp[valid_mask] = 1e-2
            
            sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
            if sky_ratio > 0.5:
                non_sky_mask = da_disp > 0.01
                gt_disp_ms = (
                    gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
                )
                da_disp_ms = (
                    da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
                )
                scale = np.median(gt_disp_ms / da_disp_ms)
                shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
            else:
                gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
                da_disp_ms = da_disp - np.median(da_disp) + 1e-8
                scale = np.median(gt_disp_ms / da_disp_ms)
                shift = np.median(gt_disp - scale * da_disp)

            gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
            da_disp_ms = da_disp - np.median(da_disp) + 1e-8

            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp - scale * da_disp)
            
            aligned_disp = scale * da_disp + shift
            disp_list.append(aligned_disp)   
        mean_disp = np.mean(np.stack(disp_list, 0))

    for idx in range(num_frames):
        frame_name = f"{idx:03d}.png"
        img_path = os.path.join(path, "images_2", frame_name)
        image_name = Path(img_path).stem

        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))

        C2W = np.eye(4).astype(np.float32)
        W2C = np.linalg.inv(C2W)

        R = C2W[:3, :3]
        T = W2C[:3, 3]
        fid = idx / max_time

        FovY = focal2fov(focal_length, curr_img.size[1])
        FovX = focal2fov(focal_length, curr_img.size[0])

        metadata = dycheck_geometry.Camera(
            orientation=W2C[:3, :3],
            position=C2W[:3, 3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1] / 2.0, sh[0] / 2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )

        if args.depth_type == "depth":
            depth = depth_list[idx] / (mean_depth)
        elif args.depth_type == "disp":
            aligned_disp = disp_list[idx] / mean_disp
            depth = 1.0 / (aligned_disp + 1e-8)

        normal_path = os.path.join(path, "normal", frame_name.replace(".png", ".npy"))
        if not os.path.exists(normal_path):
            normal = get_normals(torch.from_numpy(depth).squeeze(-1)[None], metadata)
            normal = normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            np.save(normal_path, normal)
        else:
            normal = np.load(normal_path)
        if np.isnan(normal).any():
            breakpoint()

        # apply avg pooling to normal
        normal = (
            F.avg_pool2d(torch.from_numpy(normal).permute(2, 0, 1)[None], 5, stride=1, padding=2)
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

        # tracklet_path_fwd = os.path.join(path, "forward_tracks_dynamic.npy")
        # tracklet_fwd = np.load(tracklet_path_fwd)
        # tracklet = tracklet_fwd
        
        if idx == 0:
            target_tracks = target_tracks.cpu().numpy()
            # target_visibility = target_visibility.cpu().numpy()
            target_tracks_static = target_tracks_static.cpu().numpy()
            # target_visibility_static = target_visibility_static.cpu().numpy()
        else:
            target_tracks = None
            # target_visibility = None
            target_tracks_static = None
            # target_visibility_static = None

        # read instance mask
        instance_path = os.path.join(path, "instance_mask", frame_name.split(".")[0] + "/*.png")
        instance_mask_list = []
        for mask_path in sorted(glob.glob(instance_path)):
            instance_mask = (
                PILtoTorch(Image.open(mask_path), (int(sh[1]), int(sh[0]))).squeeze(0).unsqueeze(-1).numpy()
            )
            instance_mask[instance_mask > 0] = 1
            instance_mask_list.append(instance_mask)
        instance_mask_list = np.stack(instance_mask_list, 0)

        new_mask = np.zeros_like(instance_mask_list[0])
        for instance in instance_mask_list:
            new_mask = np.maximum(new_mask, instance)

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=curr_img,
            image_path=img_path,
            max_time=max_time,
            image_name=image_name,
            width=curr_img.size[0],
            height=curr_img.size[1],
            time=fid,
            mask=new_mask,
            metadata=metadata,
            normal=normal,
            depth=depth,
            instance_mask=instance_mask_list,
            target_tracks=target_tracks,
            target_tracks_static=target_tracks_static,
        )
        train_cam_infos.append(cam_info)

    for idx in range(num_frames):
        frame_name = f"v000_t{idx:03d}.png"
        img_path = os.path.join(path, "gt", frame_name)
        image_name = Path(img_path).stem

        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))

        C2W = np.eye(4).astype(np.float32)
        W2C = np.linalg.inv(C2W)

        R = C2W[:3, :3]
        T = W2C[:3, 3]
        fid = idx / max_time

        FovY = focal2fov(focal_length, curr_img.size[1])
        FovX = focal2fov(focal_length, curr_img.size[0])

        metadata = dycheck_geometry.Camera(
            orientation=W2C[:3, :3],
            position=C2W[:3, 3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1] / 2.0, sh[0] / 2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=curr_img,
            image_path=img_path,
            max_time=max_time,
            image_name=image_name,
            width=curr_img.size[0],
            height=curr_img.size[1],
            time=fid,
            mask=None,
            metadata=metadata,
            normal=None,
            depth=None,
            sem_mask=None,
        )
        test_cam_infos.append(cam_info)
    return train_cam_infos, test_cam_infos, max_time


def readNvidiaInfo(args):
    print("Reading Nvidia Info")
    path = args.source_path
    train_cam_infos, test_cam_infos, max_time = readNvidiaCameras(args)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "dummy_points3D.ply")

    if not os.path.exists(ply_path) or os.path.getsize(ply_path) < 100:
        totalxyz = []
        totalrgb = []
        totaltime = []

        start_time = 0

        xyz, rgb = np.zeros((1, 3)), np.zeros((1, 3))

        for i in range(start_time, start_time + max_time):
            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / max_time)

        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, totaltime), axis=1)
        storePly(ply_path, xyzt, rgb)
    try:
        pcd = fetchPly(ply_path)

    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        video_cameras=None,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=max_time,
    )
    return scene_info


def readSurgicalInfo(args):
    """
    Dataset reader for surgical endoscopic videos.
    
    Expected structure:
    source_path/
        images/      - RGB frames (000.png, 001.png, ...)
        masks/       - Tool segmentation masks (000.png, 001.png, ...)
        depth_prior/ - Monocular depth priors (000.npy, 001.npy, ...)
    """
    import cv2
    print("Reading Surgical Endoscopic Info")
    path = args.source_path
    train_cam_infos, test_cam_infos = [], []
    
    image_dir = os.path.join(path, "images")
    mask_dir = os.path.join(path, "masks")
    depth_dir = os.path.join(path, "depth_prior")
    
    image_list = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not image_list:
        image_list = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    if not image_list:
        print(f"ERROR: No images found in {image_dir}")
        return None
    
    img_0 = cv2.imread(image_list[0])
    sh = img_0.shape[:2]
    max_time = len(image_list) - 1
    focal_length = 500  # dummy initial, optimized via pose_network
    
    for idx, img_path in enumerate(image_list):
        image_name = Path(img_path).stem
        fid = idx / max_time if max_time > 0 else 0
        
        curr_img = Image.open(img_path).convert("RGB")
        
        # Tool mask
        mask = None
        mask_path = os.path.join(mask_dir, f"{image_name}.png")
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask = (mask_img > 127).astype(np.float32)[..., None]
        
        # Depth prior
        depth_prior = None
        depth_path = os.path.join(depth_dir, f"{image_name}.npy")
        if os.path.exists(depth_path):
            depth_prior = np.load(depth_path)
        else:
            depth_prior = np.ones((*sh, 1), dtype=np.float32) * 0.5
        
        # Identity camera (COLMAP-free, optimized via pose_network)
        C2W = np.eye(4).astype(np.float32)
        W2C = np.linalg.inv(C2W)
        R = C2W[:3, :3]
        T = W2C[:3, 3]
        
        FovY = focal2fov(focal_length, sh[0])
        FovX = focal2fov(focal_length, sh[1])
        
        metadata = dycheck_geometry.Camera(
            orientation=W2C[:3, :3],
            position=C2W[:3, 3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1] / 2.0, sh[0] / 2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )
        
        # Compute normals from depth
        normal = np.zeros((*sh, 3), dtype=np.float32)
        
        cam_info = CameraInfo(
            uid=idx,
            R=R, T=T,
            FovY=FovY, FovX=FovX,
            image=curr_img,
            image_path=img_path,
            image_name=image_name,
            width=sh[1], height=sh[0],
            time=fid, max_time=max_time,
            mask=mask,
            metadata=metadata,
            normal=normal,
            depth=depth_prior,
        )
        train_cam_infos.append(cam_info)
    
    print(f"Loaded {len(train_cam_infos)} surgical frames")
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "dummy_points3D.ply")
    if not os.path.exists(ply_path) or os.path.getsize(ply_path) < 100:
        xyz = np.zeros((1, 3))
        rgb = np.zeros((1, 3))
        times = np.zeros((1, 1))
        xyzt = np.concatenate((xyz, times), axis=1)
        storePly(ply_path, xyzt, rgb)
    
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        video_cameras=train_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=max_time,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "nvidia": readNvidiaInfo,
    "surgical": readSurgicalInfo,
}
