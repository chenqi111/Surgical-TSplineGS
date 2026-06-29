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
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
import kornia
from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render, render_static
from PIL import Image
from scene import GaussianModel, Scene, dataset_readers, deformation
from scene.gaussian_model import controlgaussians
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.graphics_utils import BasicPointCloud, getWorld2View2
from utils.image_utils import psnr
from utils.loss_utils import BinaryDiceLoss, l1_loss, ssim, l2_loss, depth_consistency_loss, trajectory_smoothness_loss
from utils.main_utils import get_gs_mask, get_pixels, get_normals, error_to_prob
from utils.scene_utils import render_training_image
from utils.timer import Timer

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(
    dataset,
    opt,
    hyper,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    dyn_gaussians,
    stat_gaussians,
    scene,
    stage,
    tb_writer,
    train_iter,
    timer,
):
    flag_d = 0
    flag_s = 0
    densify = opt.densify

    BEST_PSNR, BEST_ITER = 0, 0

    first_iter = 0
    dyn_gaussians.training_setup(opt, stage=stage)
    stat_gaussians.training_setup(opt, stage=stage)

    if stage == "fine":
        bg_color = [1, 1, 1, -10] if dataset.white_background else [0, 0, 0, -10]
    else:
        bg_color = [1, 1, 1, 0] if dataset.white_background else [0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log_photo = 0.0
    ema_loss_for_log_reg = 0.0
    ema_loss_for_log_mask = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    viewpoint_stack = None
    viewpoint_stack_ids = []
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    my_test_cams = [i for i in test_cams]  # Large CPU usage
    viewpoint_stack = [i for i in train_cams]  # Large CPU usage

    # Get GT cam to worlds for testing
    gt_train_pose_list = []
    for view_p in viewpoint_stack:
        gt_Rt = getWorld2View2(view_p.R, view_p.T, view_p.trans, view_p.scale)
        gt_C2W = np.linalg.inv(gt_Rt)
        gt_train_pose_list.append(gt_C2W)

    gt_test_pose_list = []
    for view_p in my_test_cams:
        gt_Rt = getWorld2View2(view_p.R, view_p.T, view_p.trans, view_p.scale)
        gt_C2W = np.linalg.inv(gt_Rt)
        gt_test_pose_list.append(gt_C2W)

    batch_size = opt.coarse_batch_size if stage == "warm" else opt.fine_batch_size

    print("data loading done")

    mask_dice_loss = BinaryDiceLoss(from_logits=False)

    if stage == "fine":
        pixels = get_pixels(
            scene.train_camera.dataset[0].metadata.image_size_x,
            scene.train_camera.dataset[0].metadata.image_size_y,
            use_center=True,
        )
        if pixels.shape[-1] != 2:
            raise ValueError("The last dimension of pixels must be 2.")
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

        # update viewpoint_stack
        with torch.no_grad():
            for cam in viewpoint_stack:
                time_in = torch.tensor(cam.time).float().cuda()
                pred_R, pred_T = dyn_gaussians._posenet(time_in.view(1, 1))
                R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
                t_ = pred_T.detach().cpu().numpy()
                cam.update_cam(
                    R_[0],
                    t_[0],
                    local_viewdirs,
                    batch_shape,
                    dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
                )

            for view_id in range(len(my_test_cams)):
                my_test_cams[view_id].update_cam(
                    viewpoint_stack[0].R, viewpoint_stack[0].T, local_viewdirs, batch_shape, viewpoint_stack[0].focal
                )
    else:
        pixels = get_pixels(
            scene.train_camera.dataset[0].metadata.image_size_x,
            scene.train_camera.dataset[0].metadata.image_size_y,
            use_center=True,
        )
        if pixels.shape[-1] != 2:
            raise ValueError("The last dimension of pixels must be 2.")
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

    # Training loop
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()
        dyn_gaussians.update_learning_rate(iteration)
        stat_gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and iteration > 2000:
            dyn_gaussians.oneupSHdegree()
            stat_gaussians.oneupSHdegree()

        # Pick a random Camera
        viewpoint_cams = []
        prev_viewpoint_cams = []
        next_viewpoint_cams = []

        idx = 0
        while idx < batch_size:
            if not viewpoint_stack_ids:
                viewpoint_stack_ids = list(range(len(viewpoint_stack)))

            id = randint(0, len(viewpoint_stack_ids) - 1)
            
            id = viewpoint_stack_ids.pop(id)
            viewpoint_cams.append(viewpoint_stack[id])
            idx += 1

            # Sample 3 views for training (1 target 2 reference)
            all_ids = list(range(len(viewpoint_stack)))
            all_ids.remove(id)
            
            prev_id = all_ids.pop(randint(0, (len(all_ids) - 1) // 2))
            prev_viewpoint_cams.append(viewpoint_stack[prev_id])
            next_id = all_ids.pop(randint((len(all_ids) - 1) // 2, len(all_ids) - 1))
            next_viewpoint_cams.append(viewpoint_stack[next_id])

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        images = []
        s_images = []
        sp_images = []
        sn_images = []
        d_images = []
        gt_images = []
        prev_images = []
        next_images = []

        pred_normals = []
        pred_normals_cvd = []
        gt_normals = []
        gt_pixels = []
        gt_depths = []
        prev_depths = []
        prev_normals = []
        next_depths = []
        next_normals = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        depth_list = []
        s_depth_list = []
        sp_depth_list = []
        sn_depth_list = []
        imgs_p_list = []
        depth_p_list = []

        imgs_n_list = []
        depth_n_list = []

        Ks = []

        motion_masks = []
        d_alphas = []
        d_depths = []

        s_alphas = []

        time = []
        prev_time = []
        next_time = []

        for n_batch in range(len(viewpoint_cams)):
            time.append(torch.tensor(viewpoint_cams[n_batch].time).float().cuda())
            prev_time.append(torch.tensor(prev_viewpoint_cams[n_batch].time).float().cuda())
            next_time.append(torch.tensor(next_viewpoint_cams[n_batch].time).float().cuda())

            gt_image = viewpoint_cams[n_batch].original_image.cuda()
            prev_image = prev_viewpoint_cams[n_batch].original_image.cuda()
            next_image = next_viewpoint_cams[n_batch].original_image.cuda()

            gt_images.append(gt_image.unsqueeze(0))
            prev_images.append(prev_image.unsqueeze(0))
            next_images.append(next_image.unsqueeze(0))

            gt_normals.append(viewpoint_cams[n_batch].normal[None].cuda())
            gt_depths.append(viewpoint_cams[n_batch].depth[None].cuda())
            prev_normals.append(prev_viewpoint_cams[n_batch].normal[None].cuda())
            prev_depths.append(prev_viewpoint_cams[n_batch].depth[None].cuda())
            next_normals.append(next_viewpoint_cams[n_batch].normal[None].cuda())
            next_depths.append(next_viewpoint_cams[n_batch].depth[None].cuda())

            pixels = viewpoint_cams[n_batch].metadata.get_pixels(normalize=True)
            pixels = torch.from_numpy(pixels).cuda()
            gt_pixels.append(pixels)

        alpha_tensor = 1
        gt_image_tensor = torch.cat(gt_images, 0)
        gt_normal_tensor = torch.cat(gt_normals, 0)
        B, C, H, W = gt_image_tensor.shape
        prev_image_tensor = torch.cat(prev_images, 0)
        next_image_tensor = torch.cat(next_images, 0)

        gt_depth_tensor = torch.cat(gt_depths, 0)
        depth_in = gt_depth_tensor.view(-1, 1)

        pgt_depth_tensor = torch.cat(prev_depths, 0)
        pdepth_in = pgt_depth_tensor.view(-1, 1)

        ngt_depth_tensor = torch.cat(next_depths, 0)
        ndepth_in = ngt_depth_tensor.view(-1, 1)

        time_in = torch.stack(time, 0).view(len(viewpoint_cams), 1)
        prev_time_in = torch.stack(prev_time, 0).view(len(viewpoint_cams), 1)
        next_time_in = torch.stack(next_time, 0).view(len(viewpoint_cams), 1)

        pred_R, pred_T, CVD = dyn_gaussians._posenet(time_in, depth=depth_in)
        gt_depth_tensor = CVD.detach()

        p_pred_R, p_pred_T, p_CVD = dyn_gaussians._posenet(prev_time_in, depth=pdepth_in)
        pgt_depth_tensor = p_CVD.detach()

        n_pred_R, n_pred_T, n_CVD = dyn_gaussians._posenet(next_time_in, depth=ndepth_in)
        ngt_depth_tensor = n_CVD.detach()

        w2c_target = torch.cat((pred_R, pred_T[:, :, None]), -1)
        w2c_prev = torch.cat((p_pred_R, p_pred_T[:, :, None]), -1)
        w2c_next = torch.cat((n_pred_R, n_pred_T[:, :, None]), -1)
        no_stat_gs = stat_gaussians.get_xyz.shape[0]
        no_dyn_gs = dyn_gaussians.get_xyz.shape[0]

        with torch.no_grad():
            # R is cam to world
            # t is world to cam
            R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
            t_ = pred_T.detach().cpu().numpy()
            pR_ = torch.transpose(p_pred_R, 2, 1).detach().cpu().numpy()
            pt_ = p_pred_T.detach().cpu().numpy()
            nR_ = torch.transpose(n_pred_R, 2, 1).detach().cpu().numpy()
            nt_ = n_pred_T.detach().cpu().numpy()
            
            # R_ = torch.transpose(pred_R, 2, 1)
            # t_ = pred_T
            # pR_ = torch.transpose(p_pred_R, 2, 1)
            # pt_ = p_pred_T
            # nR_ = torch.transpose(n_pred_R, 2, 1)
            # nt_ = n_pred_T

        for n_batch, viewpoint_cam in enumerate(viewpoint_cams):
            camera_metadata = viewpoint_cam.metadata
            K = torch.zeros(3, 3).type_as(gt_image_tensor)
            K[0, 0] = dyn_gaussians._posenet.focal_bias.exp()
            K[1, 1] = dyn_gaussians._posenet.focal_bias.exp()
            K[0, 2] = float(camera_metadata.principal_point_x)
            K[1, 2] = float(camera_metadata.principal_point_y)
            K[2, 2] = float(1)
            Ks.append(K[None])

            viewpoint_cam.update_cam(
                R_[n_batch],
                t_[n_batch],
                local_viewdirs,
                batch_shape,
                dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
            )
            prev_viewpoint_cams[n_batch].update_cam(
                pR_[n_batch],
                pt_[n_batch],
                local_viewdirs,
                batch_shape,
                dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
            )
            next_viewpoint_cams[n_batch].update_cam(
                nR_[n_batch],
                nt_[n_batch],
                local_viewdirs,
                batch_shape,
                dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
            )

            if stage != "warm":
                # Get reference viewpoints at target time step
                if stage == "fine":
                    with torch.no_grad():
                        render_p_pkg = render(
                            prev_viewpoint_cams[n_batch], stat_gaussians, dyn_gaussians, background, get_static=True
                        )
                        imgs_p_list.append(render_p_pkg["render"].unsqueeze(0))
                        depth_p_list.append(render_p_pkg["depth"].unsqueeze(0))
                        sp_images.append(render_p_pkg["s_render"].unsqueeze(0))
                        sp_depth_list.append(render_p_pkg["s_depth"].unsqueeze(0))

                        render_n_pkg = render(
                            next_viewpoint_cams[n_batch], stat_gaussians, dyn_gaussians, background, get_static=True
                        )
                        imgs_n_list.append(render_n_pkg["render"].unsqueeze(0))
                        depth_n_list.append(render_n_pkg["depth"].unsqueeze(0))
                        sn_images.append(render_n_pkg["s_render"].unsqueeze(0))
                        sn_depth_list.append(render_n_pkg["s_depth"].unsqueeze(0))

                    render_pkg = render(
                        viewpoint_cam, stat_gaussians, dyn_gaussians, background, get_static=True, get_dynamic=True
                    )
                else:
                    with torch.no_grad():
                        render_p_pkg = render_static(
                            prev_viewpoint_cams[n_batch], stat_gaussians, dyn_gaussians, background, get_static=True
                        )
                        imgs_p_list.append(render_p_pkg["render"].unsqueeze(0))
                        depth_p_list.append(render_p_pkg["depth"].unsqueeze(0))
                        sp_images.append(render_p_pkg["s_render"].unsqueeze(0))
                        sp_depth_list.append(render_p_pkg["s_depth"].unsqueeze(0))

                        render_n_pkg = render_static(
                            next_viewpoint_cams[n_batch], stat_gaussians, dyn_gaussians, background, get_static=True
                        )
                        imgs_n_list.append(render_n_pkg["render"].unsqueeze(0))
                        depth_n_list.append(render_n_pkg["depth"].unsqueeze(0))
                        sn_images.append(render_n_pkg["s_render"].unsqueeze(0))
                        sn_depth_list.append(render_n_pkg["s_depth"].unsqueeze(0))

                    render_pkg = render_static(
                        viewpoint_cam, stat_gaussians, dyn_gaussians, background, get_static=True,
                    )
                s_images.append(render_pkg["s_render"].unsqueeze(0))
                s_depth_list.append(render_pkg["s_depth"].unsqueeze(0))
                pred_image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
                visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]
                depth_list.append(render_pkg["depth"].unsqueeze(0))

                radii_list.append(radii)
                visibility_filter_list.append(visibility_filter)
                viewspace_point_tensor_list.append(viewspace_point_tensor)

                images.append(pred_image.unsqueeze(0))
                d_images.append(pred_image.unsqueeze(0))

                pred_normal = get_normals(render_pkg["depth"] + 1e-6, camera_metadata)
                pred_normals.append(pred_normal)
                motion_masks.append(viewpoint_cam.mask.unsqueeze(0).cuda())
                if stage == "fine":
                    d_alphas.append(render_pkg["d_alpha"].unsqueeze(0))
                    d_depths.append(render_pkg["d_depth"].unsqueeze(0))

                s_alphas.append(render_pkg["s_alpha"].unsqueeze(0))

                if torch.isnan(pred_normal).any():
                    print("NaN found in pred normal!")

        # Aggregate into tensor and init vars
        K_tensor = torch.cat(Ks, 0)

        loss = 0
        gs_mask = 1
        gs_mask_0 = 1
        rcon_w = 1.0


        if stage != "warm":
            radii = torch.stack(radii_list, dim=0)
            visibility_filter = torch.stack(visibility_filter_list, dim=0)
            image_tensor = torch.cat(images, 0)
            depth_tensor = torch.cat(depth_list, 0)
            motion_mask_tensor = torch.cat(motion_masks, 0)
            if stage == "fine":
                d_alpha_tensor = torch.cat(d_alphas, 0)
            s_image_tensor = torch.cat(s_images, 0)
            s_depth_tensor = torch.cat(s_depth_list, 0)
            normal_tensor = torch.cat(pred_normals, 0)
            
            sp_depth_tensor = torch.cat(sp_depth_list, 0)
            depth_p_tensor = torch.cat(depth_p_list, 0)
            imgs_p_tensor = torch.cat(imgs_p_list, 0)
            sp_image_tensor = torch.cat(sp_images, 0)

            sn_depth_tensor = torch.cat(sn_depth_list, 0)
            depth_n_tensor = torch.cat(depth_n_list, 0)
            imgs_n_tensor = torch.cat(imgs_n_list, 0)
            sn_image_tensor = torch.cat(sn_images, 0)

            # Main losses (L1 and SSIM) for GS densification
            if stage == "fine":
                # Surgical-TSplineGS: use masked photometric loss (Eq. 8-9)
                if scene.dataset_type == "surgical":
                    mask_l = motion_mask_tensor  # tool mask
                    Ll1 = l1_loss(image_tensor * (1-mask_l), gt_image_tensor[:, :3, :, :] * (1-mask_l))
                    ssim_loss = 0
                    if opt.lambda_dssim != 0:
                        ssim_loss = ssim(image_tensor * (1-mask_l), gt_image_tensor * (1-mask_l))
                else:
                    Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
                    ssim_loss = 0
                    if opt.lambda_dssim != 0:
                        ssim_loss = ssim(image_tensor, gt_image_tensor)
                
                Ll1_ = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
                psnr_ = psnr(image_tensor, gt_image_tensor).detach().mean().double()

                photo_loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)
                photo_loss.backward(retain_graph=True)
            else:
                Ll1 = l1_loss(image_tensor * (1-motion_mask_tensor), gt_image_tensor[:, :3, :, :] * (1-motion_mask_tensor))
                psnr_ = psnr(image_tensor * (1-motion_mask_tensor), gt_image_tensor * (1-motion_mask_tensor)).detach().mean().double()

                ssim_loss = 0
                if opt.lambda_dssim != 0:
                    ssim_loss = ssim(image_tensor * (1-motion_mask_tensor), gt_image_tensor * (1-motion_mask_tensor))

                photo_loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)
                photo_loss.backward(retain_graph=True)
                
            # Split static and dynamic gradients (we know their indices because of cat in render)
            stat_viewspace_point_tensor_grad = torch.zeros(no_stat_gs, 2).cuda()
            stat_radii = radii[..., :no_stat_gs].max(dim=0).values
            stat_visibility_filter = visibility_filter[..., :no_stat_gs].any(dim=0)
            for grad_idx in range(0, len(viewspace_point_tensor_list)):
                # stat_viewspace_point_tensor_grad += viewspace_point_tensor_list[grad_idx].grad.squeeze(0)[0:no_stat_gs]
                stat_viewspace_point_tensor_grad += viewspace_point_tensor_list[grad_idx].absgrad.squeeze(0)[
                    0:no_stat_gs
                ]
            stat_viewspace_point_tensor_grad[:, 0] *= W * 0.5
            stat_viewspace_point_tensor_grad[:, 1] *= H * 0.5

            if stage == "fine":
                dyn_viewspace_point_tensor_grad = torch.zeros(no_dyn_gs, 2).cuda()
                dyn_radii = radii[..., no_stat_gs : no_stat_gs + no_dyn_gs].max(dim=0).values
                dyn_visibility_filter = visibility_filter[..., no_stat_gs : no_stat_gs + no_dyn_gs].any(dim=0)
                for grad_idx in range(0, len(viewspace_point_tensor_list)):
                    # dyn_viewspace_point_tensor_grad += viewspace_point_tensor_list[grad_idx].grad.squeeze(0)[no_stat_gs:no_stat_gs + no_dyn_gs]
                    dyn_viewspace_point_tensor_grad += viewspace_point_tensor_list[grad_idx].absgrad.squeeze(0)[
                        no_stat_gs : no_stat_gs + no_dyn_gs
                    ]
                dyn_viewspace_point_tensor_grad[:, 0] *= W * 0.5
                dyn_viewspace_point_tensor_grad[:, 1] *= H * 0.5

            # Dyn masks from gaussians
            with torch.no_grad():
                min_gs_mask = (iteration / final_iter) * (1e3 - 10) + 10
                min_gs_mask = 1 / min_gs_mask
                gs_mask_c, gs_mask_d = get_gs_mask(s_image_tensor, gt_image_tensor, s_depth_tensor, depth_tensor, CVD)
                gs_mask_0 = gs_mask_c * gs_mask_d
                gs_mask_0[gs_mask_0 < min_gs_mask] = min_gs_mask
                gs_mask_b = torch.bernoulli(gs_mask_0).detach()

                gs_mask = gs_mask_b
                gs_mask = alpha_tensor * gs_mask

                # Do not mask for supervising s_image_tensor if s_image has lower error than image_tensor
                statGS_mask = (
                    torch.mean(torch.abs(s_image_tensor - gt_image_tensor), 1, True)
                    < torch.mean(torch.abs(image_tensor - gt_image_tensor), 1, True)
                ).type_as(image_tensor)
                statGS_mask = (statGS_mask + gs_mask_c) * gs_mask_d * alpha_tensor
                statGS_mask[statGS_mask < min_gs_mask] = min_gs_mask
                statGS_mask[statGS_mask > 1] = 1

                statGS_mask = torch.bernoulli(statGS_mask.detach())
                statGS_mask = gs_mask

                # Get gs masks for reference views (used for pose losses)
                gs_mask_pc, gs_mask_pd = get_gs_mask(
                    sp_image_tensor, prev_image_tensor, depth_p_tensor, sp_depth_tensor, p_CVD
                )
                gs_mask_nc, gs_mask_nd = get_gs_mask(
                    sn_image_tensor, next_image_tensor, depth_n_tensor, sn_depth_tensor, n_CVD
                )
                gs_mask_p = gs_mask_pc * gs_mask_pd
                gs_mask_n = gs_mask_nc * gs_mask_nd
                gs_mask_p[gs_mask_p < min_gs_mask] = min_gs_mask
                gs_mask_n[gs_mask_n < min_gs_mask] = min_gs_mask

            reg_loss = 0
            if stage == "fine":
                reg_loss += l1_loss(image_tensor, gt_image_tensor[:, :3, :, :], mask=motion_mask_tensor)
                # reg_loss += l1_loss(s_image_tensor, gt_image_tensor[:, :3, :, :], mask=1.0 - motion_mask_tensor)

                depth_loss = l1_loss(depth_tensor, gt_depth_tensor)
                reg_loss += opt.w_depth * depth_loss

                mask_loss = opt.w_mask * mask_dice_loss(d_alpha_tensor, motion_mask_tensor)
                reg_loss += mask_loss

                normal_loss = l2_loss(normal_tensor, gt_normal_tensor, mask=motion_mask_tensor)
                loss += opt.w_normal * normal_loss

                loss += reg_loss

        warped_prev, p_grid = deformation.inverse_warp_rt1_rt2(
            prev_image_tensor, CVD, w2c_target, w2c_prev, K_tensor, torch.inverse(K_tensor), ret_grid=True
        )
        warped_next, n_grid = deformation.inverse_warp_rt1_rt2(
            next_image_tensor, CVD, w2c_target, w2c_next, K_tensor, torch.inverse(K_tensor), ret_grid=True
        )
        
        if stage == "warm" and opt.w_track > 0:
            # tracking loss
            tracklet = viewpoint_stack[0].target_tracks_static.cuda()
            _, num_points, _ = tracklet.shape
            current_idx = torch.tensor([viewpoint.uid for viewpoint in viewpoint_cams]).float().cuda() 
            prev_idx = torch.tensor([viewpoint.uid for viewpoint in prev_viewpoint_cams]).float().cuda() 
            next_idx = torch.tensor([viewpoint.uid for viewpoint in next_viewpoint_cams]).float().cuda() 

            current_track = torch.gather(tracklet, 0, current_idx[:,None,None].expand(-1, num_points, 2).long())
            prev_track = torch.gather(tracklet, 0, prev_idx[:,None,None].expand(-1, num_points, 2).long())
            next_track = torch.gather(tracklet, 0, next_idx[:,None,None].expand(-1, num_points, 2).long())
            
            current_track[...,0] = (current_track[...,0] / W)*2 - 1
            current_track[...,1] = (current_track[...,1] / H)*2 - 1
            prev_track[...,0] = (prev_track[...,0] / W)*2 - 1
            prev_track[...,1] = (prev_track[...,1] / H)*2 - 1
            next_track[...,0] = (next_track[...,0] / W)*2 - 1
            next_track[...,1] = (next_track[...,1] / H)*2 - 1
            
            p_grid_track = torch.nn.functional.grid_sample(p_grid.permute(0,3,1,2), current_track[:,None], align_corners=True, mode='bilinear').squeeze(-2).permute(0,2,1)
            n_grid_track = torch.nn.functional.grid_sample(n_grid.permute(0,3,1,2), current_track[:,None], align_corners=True, mode='bilinear').squeeze(-2).permute(0,2,1)
            
            track_loss = torch.mean((p_grid_track - prev_track)**2) + torch.mean((n_grid_track - next_track)**2)
            loss += opt.w_track * track_loss

        with torch.no_grad():
            out_p = alpha_tensor * (torch.sum(warped_prev.detach(), dim=1, keepdim=True) > 0).type_as(warped_prev)
            out_n = alpha_tensor * (torch.sum(warped_next.detach(), dim=1, keepdim=True) > 0).type_as(warped_next)

        if stage != "warm":
            warped_prev_, gs_p_grid = deformation.inverse_warp_rt1_rt2(
                prev_image_tensor,
                depth_tensor.detach(),
                w2c_target,
                w2c_prev,
                K_tensor,
                torch.inverse(K_tensor),
                ret_grid=True,
            )
            warped_next_, gs_n_grid = deformation.inverse_warp_rt1_rt2(
                next_image_tensor,
                depth_tensor.detach(),
                w2c_target,
                w2c_next,
                K_tensor,
                torch.inverse(K_tensor),
                ret_grid=True,
            )

            warped_s_prev = deformation.inverse_warp_rt1_rt2(
                imgs_p_tensor,
                CVD,
                w2c_target.detach(),
                w2c_prev.detach(),
                K_tensor,
                torch.inverse(K_tensor),
                ret_grid=False,
            )
            warped_s_next = deformation.inverse_warp_rt1_rt2(
                imgs_n_tensor,
                CVD,
                w2c_target.detach(),
                w2c_next.detach(),
                K_tensor,
                torch.inverse(K_tensor),
                ret_grid=False,
            )

            with torch.no_grad():
                # Get occlusion masks
                warped_s_prev_ = F.grid_sample(imgs_p_tensor, gs_p_grid, align_corners=True).detach()
                warped_s_next_ = F.grid_sample(imgs_n_tensor, gs_n_grid, align_corners=True).detach()
                gsp_err = torch.mean((warped_s_prev_ - image_tensor) ** 2, dim=1, keepdim=True)
                gsn_err = torch.mean((warped_s_next_ - image_tensor) ** 2, dim=1, keepdim=True)
                occ_mask_p = error_to_prob(gsp_err.detach(), mask=out_p)
                occ_mask_n = error_to_prob(gsn_err.detach(), mask=out_n)
                geo_mask_p = occ_mask_p
                geo_mask_n = occ_mask_n

                # Masks from diff-timestep-same-viewpoint error
                color_mask_p = F.grid_sample(gs_mask_p, p_grid, align_corners=True).detach()
                color_mask_p_gs = F.grid_sample(gs_mask_p, gs_p_grid, align_corners=True).detach()
                color_mask_n = F.grid_sample(gs_mask_n, n_grid, align_corners=True).detach()
                color_mask_n_gs = F.grid_sample(gs_mask_n, gs_n_grid, align_corners=True).detach()

        # Prevent entanglement, detach CVD's and grids (only pose and depths are supervised)
        wc = deformation.points_from_DRTK(CVD, w2c_target, K_tensor).view(gt_image_tensor.shape)
        pwc = deformation.points_from_DRTK(p_CVD, w2c_prev, K_tensor).view(gt_image_tensor.shape)
        nwc = deformation.points_from_DRTK(n_CVD, w2c_next, K_tensor).view(gt_image_tensor.shape)
        warped_pwc = F.grid_sample(pwc, p_grid.detach(), align_corners=True)
        warped_nwc = F.grid_sample(nwc, n_grid.detach(), align_corners=True)

        # Compute photometric errors (using ssim for these mask in warm is no good)
        p_error = torch.mean((gt_image_tensor - warped_prev) ** 2, dim=1, keepdim=True)
        n_error = torch.mean((gt_image_tensor - warped_next) ** 2, dim=1, keepdim=True)
        pose_con_loss = torch.mean((warped_prev - warped_next) ** 2, dim=1, keepdim=True)

        # Geometry errors
        p_wcerr = torch.mean((wc - warped_pwc) ** 2, dim=1, keepdim=True)
        n_wcerr = torch.mean((wc - warped_nwc) ** 2, dim=1, keepdim=True)
        pose_geocon_loss = torch.mean((warped_pwc - warped_nwc) ** 2, dim=1, keepdim=True)

        # Color and Geo masks
        with torch.no_grad():
            if stage == "warm":
                # Color mask
                color_mask_p = error_to_prob(p_error.detach(), mask=out_p)
                color_mask_n = error_to_prob(n_error.detach(), mask=out_n)

                # Geo mask
                geo_mask_p = error_to_prob(p_wcerr.detach(), mask=out_p)
                geo_mask_n = error_to_prob(n_wcerr.detach(), mask=out_n)

            prev_mask = gs_mask_0 * color_mask_p * geo_mask_p * out_p
            next_mask = gs_mask_0 * color_mask_n * geo_mask_n * out_n
            pn_mask = torch.bernoulli(prev_mask * color_mask_n * geo_mask_n * out_n)
            prev_mask = torch.bernoulli(prev_mask)
            next_mask = torch.bernoulli(next_mask)

        # Color error
        color_loss = (
            torch.sum(prev_mask * p_error) / (torch.sum(prev_mask) + 1e-7)
            + torch.sum(next_mask * n_error) / (torch.sum(next_mask) + 1e-7)
            + rcon_w * torch.sum(pn_mask * pose_con_loss) / (torch.sum(pn_mask) + 1e-7)
        )
        # Geometry error
        geo_loss = (
            torch.sum(prev_mask * p_wcerr) / (torch.sum(prev_mask) + 1e-7)
            + torch.sum(next_mask * n_wcerr) / (torch.sum(next_mask) + 1e-7)
            + rcon_w * torch.sum(pn_mask * pose_geocon_loss) / (torch.sum(pn_mask) + 1e-7)
        )
        # Final pose loss
        cvd_pose_loss = color_loss + 1e-3 * geo_loss
        if opt.p_lambda_dssim > 0:
            # SSIM
            pose_ssim_loss = (
                (1 - ssim(warped_prev * prev_mask + gt_image_tensor * (1 - prev_mask), gt_image_tensor))
                + (1 - ssim(warped_next * next_mask + gt_image_tensor * (1 - next_mask), gt_image_tensor))
                + rcon_w
                * (
                    1
                    - ssim(
                        warped_prev * pn_mask + warped_next * (1 - pn_mask),
                        warped_next * pn_mask + warped_prev * (1 - pn_mask),
                    )
                )
            )
            cvd_pose_loss += opt.p_lambda_dssim * pose_ssim_loss

        if stage == "fine":
            # Do the same, but using 3DGS depths (mask are same as ^)
            with torch.no_grad():
                prev_occ = torch.bernoulli(geo_mask_p)
                next_occ = torch.bernoulli(geo_mask_n)

                prev_mask = gs_mask_0 * color_mask_p_gs * geo_mask_p * out_p
                next_mask = gs_mask_0 * color_mask_n_gs * geo_mask_n * out_n
                pn_mask = torch.bernoulli(prev_mask * color_mask_n * geo_mask_n * out_n)
                prev_mask = torch.bernoulli(prev_mask)
                next_mask = torch.bernoulli(next_mask)

            # compute pose loss
            w_gs_pose = 1
            p_error = torch.mean((gt_image_tensor - warped_prev_) ** 2, dim=1, keepdim=True)
            n_error = torch.mean((gt_image_tensor - warped_next_) ** 2, dim=1, keepdim=True)
            # Color
            gs_color_pose_loss = torch.sum(prev_mask * p_error) / (torch.sum(prev_mask) + 1e-7) + torch.sum(
                next_mask * n_error
            ) / (torch.sum(next_mask) + 1e-7)
            cvd_pose_loss += w_gs_pose * gs_color_pose_loss
            # SSIM
            if opt.p_lambda_dssim > 0:
                gs_ssim_loss = (
                    1 - ssim(warped_prev_ * prev_mask + gt_image_tensor * (1 - prev_mask), gt_image_tensor)
                ) + (1 - ssim(warped_next_ * next_mask + gt_image_tensor * (1 - next_mask), gt_image_tensor))
                cvd_pose_loss += w_gs_pose * opt.p_lambda_dssim * gs_ssim_loss

            # projection loss (vde is preserved by using same ray directions)
            p_error = torch.mean((gt_image_tensor - warped_s_prev) ** 2, dim=1, keepdim=True)
            n_error = torch.mean((gt_image_tensor - warped_s_next) ** 2, dim=1, keepdim=True)
            # Color
            gs_color_pose_loss = torch.sum(prev_occ * p_error) / (torch.sum(prev_occ) + 1e-7) + torch.sum(
                next_occ * n_error
            ) / (torch.sum(next_occ) + 1e-7)
            cvd_pose_loss += gs_color_pose_loss
            # SSIM
            if opt.p_lambda_dssim > 0:
                gs_ssim_loss = (
                    1 - ssim(warped_s_prev * prev_occ + gt_image_tensor * (1 - prev_occ), gt_image_tensor)
                ) + (1 - ssim(warped_s_next * next_occ + gt_image_tensor * (1 - next_occ), gt_image_tensor))
                cvd_pose_loss += opt.p_lambda_dssim * gs_ssim_loss

        loss += cvd_pose_loss

        # === Surgical-TSplineGS: additional losses ===
        if stage == "fine" and scene.dataset_type == "surgical":
            # Depth-consistency loss (Eq. 10)
            if hyper.lambda_depth_consistency > 0 and hasattr(viewpoint_cams[0], 'depth') and viewpoint_cams[0].depth is not None:
                depth_gt = gt_depth_tensor  # GT depth from depth prior
                depth_rendered = depth_tensor  # rendered depth
                # Use inverse of mask as confidence (lower confidence in tool regions)
                depth_conf = 1.0 - motion_mask_tensor
                depth_loss_val = depth_consistency_loss(depth_rendered, depth_gt, confidence=depth_conf)
                loss += hyper.lambda_depth_consistency * depth_loss_val
            
            # Trajectory smoothness loss (Eq. 11)
            if hyper.lambda_traj_smoothness > 0:
                smooth_loss_val = trajectory_smoothness_loss(dyn_gaussians, time_grid=10)
                loss += hyper.lambda_traj_smoothness * smooth_loss_val
        
        loss.backward()

        # === Surgical-TSplineGS: MG-MAS gradient attenuation ===
        if stage == "fine" and scene.dataset_type == "surgical" and hyper.mgmas_enabled:
            if dyn_gaussians.control_xyz.grad is not None and render_pkg.get("mg_mas_overlap") is not None:
                mg_overlap = render_pkg["mg_mas_overlap"]
                if mg_overlap is not None:
                    no_dyn = dyn_gaussians.control_xyz.shape[0]
                    if mg_overlap.shape[0] >= no_stat_gs:
                        dyn_overlap = mg_overlap[no_stat_gs:no_stat_gs + no_dyn]
                    else:
                        dyn_overlap = mg_overlap[:no_dyn] if mg_overlap.shape[0] >= no_dyn else mg_overlap
                    
                    if dyn_overlap.shape[0] == no_dyn:
                        dyn_overlap = dyn_overlap.mean(dim=-1, keepdim=True) if dyn_overlap.dim() > 1 else dyn_overlap.view(-1, 1)
                        dyn_gaussians.control_xyz.grad = dyn_gaussians.mg_mas_attenuate_gradients(
                            dyn_gaussians.control_xyz.grad, dyn_overlap
                        )

        # === Surgical-TSplineGS: TASS detection and splitting ===
        if stage == "fine" and scene.dataset_type == "surgical" and hyper.tass_enabled:
            with torch.no_grad():
                if render_pkg.get("mg_mas_overlap") is not None:
                    mg_overlap = render_pkg["mg_mas_overlap"]
                    if mg_overlap is not None:
                        no_dyn = dyn_gaussians.control_xyz.shape[0]
                        # Per-Gaussian photometric error
                        pixel_err = torch.abs(image_tensor - gt_image_tensor[:, :3, :, :]).mean(dim=1, keepdim=False)  # [B, H, W]
                        # Aggregate error per Gaussian using mask overlap as proxy
                        if mg_overlap.shape[0] >= no_stat_gs:
                            dyn_err = mg_overlap[no_stat_gs:no_stat_gs + no_dyn]
                        else:
                            dyn_err = mg_overlap[:no_dyn] if mg_overlap.shape[0] >= no_dyn else mg_overlap
                        dyn_err_flat = dyn_err.view(-1) if dyn_err.dim() > 1 else dyn_err
                        
                        if iteration % 100 == 0 and dyn_err_flat.shape[0] == no_dyn:
                            # Sample per-Gaussian error from the 2D projection
                            if "viewspace_points" in render_pkg:
                                vs_pts = render_pkg["viewspace_points"].squeeze(0)
                                if vs_pts.shape[0] >= no_stat_gs + no_dyn:
                                    dyn_vs = vs_pts[no_stat_gs:no_stat_gs + no_dyn]
                                    # Sample pixel error at Gaussian 2D centers
                                    with torch.no_grad():
                                        H, W = gt_image_tensor.shape[2:]
                                        x_norm = 2.0 * dyn_vs[:, 0].clamp(0, W-1) / (W - 1) - 1.0
                                        y_norm = 2.0 * dyn_vs[:, 1].clamp(0, H-1) / (H - 1) - 1.0
                                        grid = torch.stack([x_norm, y_norm], dim=-1).view(1, 1, -1, 2)
                                        err_sampled = torch.nn.functional.grid_sample(
                                            pixel_err[0:1].unsqueeze(0),
                                            grid, mode='bilinear', padding_mode='border', align_corners=True
                                        ).squeeze()
                                        if err_sampled.dim() > 1:
                                            err_sampled = err_sampled.mean(0)
                                        
                                        # TASS detection
                                        if err_sampled.shape[0] == no_dyn:
                                            split_events = dyn_gaussians.tass_detect_and_split(err_sampled, dyn_err_flat)
                                            if split_events:
                                                print(f"[TASS] Split {len(split_events)} Gaussians at iter {iteration}")
        if torch.isnan(loss).any():
            print("loss is nan,end training, ending program now.")
            exit()

        iter_end.record()

        with torch.no_grad():
            if torch.isnan(loss).any():
                import sys
                print(f"NAN DETECTED! loss={loss.item()}, photo_loss={photo_loss.item() if 'photo_loss' in dir() else 'N/A'}, track_loss={track_loss.item() if 'track_loss' in dir() else 'N/A'}, cvd_pose_loss={cvd_pose_loss.item()}")
                print(f"prev_mask sum={prev_mask.sum().item() if 'prev_mask' in dir() else 'N/A'}, next_mask sum={next_mask.sum().item() if 'next_mask' in dir() else 'N/A'}")
                sys.stdout.flush()
            # Progress bar
            if stage != "warm":
                ema_loss_for_log_photo = 0.4 * photo_loss.detach().item() + 0.6 * ema_loss_for_log_photo
                # ema_loss_for_log_reg = 0.4 * reg_loss.detach().item() + 0.6 * ema_loss_for_log_reg
                # ema_loss_for_log_mask = 0.4 * mask_loss.detach().item() + 0.6 * ema_loss_for_log_mask

                ema_psnr_for_log = 0.4 * psnr_.detach() + 0.6 * ema_psnr_for_log
            else:
                ema_psnr_for_log = 0
            if stage != "warm":
                if iteration % 10 == 0:
                    progress_bar.set_postfix(
                        {
                            "photo loss": f"{ema_loss_for_log_photo:.{6}f}",
                            # "reg loss": f"{ema_loss_for_log_reg:.{6}f}",
                            "psnr": f"{ema_psnr_for_log:.{2}f}",
                            "Pts (static, dynamic)": f"{no_stat_gs}, {no_dyn_gs}",
                            "Focal": f"{viewpoint_stack[0].focal}",
                            "MinCtrl": f"{dyn_gaussians.current_control_num.min().item()}",
                        }
                    )
                    progress_bar.update(10)
            else:
                if iteration % 10 == 0:
                    progress_bar.set_postfix(
                        {
                            "PoseL": f"{cvd_pose_loss.detach().item():.{6}f}",
                            "psnr": f"{ema_psnr_for_log:.{2}f}",
                            "Pts (static, dynamic)": f"{no_stat_gs}, {no_dyn_gs}",
                            "Focal": f"{dyn_gaussians._posenet.focal_bias.exp().detach().item()}",
                        }
                    )
                    progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

        # Log and save
        timer.pause()
        with torch.no_grad():
            if iteration in testing_iterations and stage != "warm":
                print(
                    "Instance scale: ",
                    dyn_gaussians._posenet.instance_scale_list.squeeze()
                    / dyn_gaussians._posenet.instance_scale_list[0].detach(),
                )

                if scene.dataset_type == "nvidia":
                    for view_id in range(len(my_test_cams)):
                        my_test_cams[view_id].update_cam(
                            viewpoint_stack[0].R,
                            viewpoint_stack[0].T,
                            local_viewdirs,
                            batch_shape,
                            viewpoint_stack[0].focal,
                        )
                else:
                    raise NotImplementedError

                test_psnr, cur_iter = training_report(
                    tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    my_test_cams,
                    render,
                    background,
                    stage,
                    scene.dataset_type,
                    final_iter,
                )

                if test_psnr > BEST_PSNR:
                    BEST_PSNR = test_psnr
                    BEST_ITER = cur_iter
                    scene.save_best_psnr(iteration, stage)

            if dataset.render_process:
                if iteration in testing_iterations:
                    if stage != "warm":
                        render_training_image(
                            scene,
                            stat_gaussians,
                            dyn_gaussians,
                            my_test_cams,
                            render,
                            pipe,
                            background,
                            stage,
                            iteration,
                            timer.get_elapsed_time(),
                            scene.dataset_type,
                        )
                    render_training_image(
                        scene,
                        stat_gaussians,
                        dyn_gaussians,
                        viewpoint_stack,
                        render,
                        pipe,
                        background,
                        stage,
                        iteration,
                        timer.get_elapsed_time(),
                        scene.dataset_type,
                        is_train=True,
                    )

            if stage != "warm" and iteration in saving_iterations:
                scene.save(iteration, stage)
                scene.cameras_extent = dataset_readers.getNerfppNorm(viewpoint_stack)["radius"]

            timer.start()

        # Optimizer step
        if iteration < opt.iterations:
            stat_gaussians.optimizer.step()
            stat_gaussians.optimizer.zero_grad(set_to_none=True)

            dyn_gaussians.optimizer.step()
            dyn_gaussians.optimizer.zero_grad(set_to_none=True)

        # Densification
        if stage != "warm":
            with torch.no_grad():
                if iteration < opt.densify_until_iter:
                    if stage == "fine":
                        dyn_gaussians.max_radii2D[dyn_visibility_filter] = torch.max(
                            dyn_gaussians.max_radii2D[dyn_visibility_filter], dyn_radii[dyn_visibility_filter]
                        )
                        dyn_gaussians.add_densification_stats(dyn_viewspace_point_tensor_grad, dyn_visibility_filter)
                        flag_d = controlgaussians(opt, dyn_gaussians, densify, iteration, scene, flag_d, is_dynamic=True)
                        
                        stat_gaussians.max_radii2D[stat_visibility_filter] = torch.max(
                            stat_gaussians.max_radii2D[stat_visibility_filter], stat_radii[stat_visibility_filter]
                        )
                        stat_gaussians.add_densification_stats(stat_viewspace_point_tensor_grad, stat_visibility_filter)
                        flag_s = controlgaussians(opt, stat_gaussians, densify, iteration, scene, 1000) # only prune
                        
                    elif stage == "fine_static":
                        stat_gaussians.max_radii2D[stat_visibility_filter] = torch.max(
                            stat_gaussians.max_radii2D[stat_visibility_filter], stat_radii[stat_visibility_filter]
                        )
                        stat_gaussians.add_densification_stats(stat_viewspace_point_tensor_grad, stat_visibility_filter)
                        flag_s = controlgaussians(opt, stat_gaussians, densify, iteration, scene, flag_s)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if stage == "fine":
                        dyn_gaussians.onedown_control_pts(viewpoint_stack)

    # scene initialization
    if stage == "warm":
        with torch.no_grad():
            # Return point cloud at IDX
            points_list, colors_list = [], []
            for IDX in range(len(viewpoint_stack)):
                image_tensor = viewpoint_stack[IDX].original_image[None].cuda()
                B, C, H, W = image_tensor.shape

                gt_normal = viewpoint_stack[IDX].normal[None].cuda()
                gt_normal.reshape(-1, 3)
                pixels = viewpoint_stack[IDX].metadata.get_pixels(normalize=True)
                pixels = torch.from_numpy(pixels).cuda()
                pixels.reshape(-1, 2)
                gt_depth = viewpoint_stack[IDX].depth[None].cuda()
                depth_in = gt_depth.reshape(-1, 1)
                time_in = torch.tensor(viewpoint_stack[IDX].time).float().cuda()
                time_in = time_in.view(1, 1)

                # Get extrinsics and depth
                pred_R, pred_T, CVD = dyn_gaussians._posenet(time_in, depth=depth_in)

                K_tensor = torch.zeros(1, 3, 3).type_as(image_tensor)
                K_tensor[:, 0, 0] = dyn_gaussians._posenet.focal_bias.exp()
                K_tensor[:, 1, 1] = dyn_gaussians._posenet.focal_bias.exp()
                K_tensor[:, 0, 2] = float(viewpoint_stack[IDX].metadata.principal_point_x)
                K_tensor[:, 1, 2] = float(viewpoint_stack[IDX].metadata.principal_point_y)
                K_tensor[:, 2, 2] = float(1)
                w2c_target = torch.cat((pred_R, pred_T[:, :, None]), -1)

                accum_error = 0
                for cam_id, view_pt in enumerate(viewpoint_stack):
                    ref_image_tensor = view_pt.original_image[None].cuda()
                    ref_normal = view_pt.normal[None].cuda()
                    ref_normal.reshape(-1, 3)
                    ref_depth = view_pt.depth[None].cuda()
                    depth_in = ref_depth.reshape(-1, 1)
                    time_in = torch.tensor(view_pt.time).float().cuda()
                    time_in = time_in.view(1, 1)

                    # Get extrinsics and depth
                    ref_R, ref_T, _ = dyn_gaussians._posenet(time_in, depth=depth_in)
                    w2c_ref = torch.cat((ref_R, ref_T[:, :, None]), -1)

                    warped_ref, grid_ref = deformation.inverse_warp_rt1_rt2(
                        ref_image_tensor, CVD, w2c_target, w2c_ref, K_tensor, torch.inverse(K_tensor), ret_grid=True
                    )

                    out_mask = (torch.sum(warped_ref, dim=1, keepdim=True) > 0).type_as(warped_ref)
                    accum_error += torch.mean(out_mask * torch.abs(warped_ref - image_tensor), dim=1, keepdim=True)

                mean_err = torch.mean(accum_error)
                accum_error = (accum_error > mean_err).type_as(accum_error)
                p_im = accum_error.detach().squeeze().cpu().numpy()
                Image.fromarray(np.rint(255 * p_im).astype(np.uint8))

                points = deformation.points_from_DRTK(CVD, w2c_target, K_tensor)
                points = torch.permute(points, (0, 2, 1))  # B, N, 3

                # Make point cloud
                colors = torch.permute(image_tensor, (0, 2, 3, 1))  # B, H, W, 3

                # error init
                if IDX == 0:
                    colors_list.append(colors[0].detach().cpu().numpy())
                    points_list.append(points[0].view(H, W, 3).detach().cpu().numpy())
                    colors = colors[0].view(-1, 3).detach().cpu().numpy()
                    points = points[0].view(-1, 3).detach().cpu().numpy()
                    coords_2d = get_pixels(W, H)
                else:
                    colors_list.append(colors[0].detach().cpu().numpy())
                    points_list.append(points[0].view(H, W, 3).detach().cpu().numpy())

                if IDX == 0:
                    accum_error = accum_error[0].squeeze(0).detach().cpu().numpy().reshape(-1)
                    motion_mask = viewpoint_stack[IDX].mask.cuda()
                    motion_error = motion_mask.squeeze(0).cpu().numpy().reshape(-1)

                    N_pts = opt.stat_npts
                    stat_colors = colors[(accum_error == 0) & (motion_error == 0), :]
                    stat_points = points[(accum_error == 0) & (motion_error == 0), :]
                    select_inds = random.sample(range(stat_colors.shape[0]), N_pts)

                    stat_time = torch.ones(stat_colors[select_inds].shape[0], 1) * viewpoint_stack[IDX].time
                    stat_color = stat_colors[select_inds]
                    stat_point = stat_points[select_inds]

                    N_pts = opt.dyn_npts
                    dyn_colors = colors[(accum_error == 1) & (motion_error == 1), :]
                    dyn_points = points[(accum_error == 1) & (motion_error == 1), :]
                    dyn_coords_2d = coords_2d.reshape(-1, 2)[(accum_error == 1) & (motion_error == 1), :]
                    if dyn_colors.shape[0] < N_pts:
                        if dyn_colors.shape[0] == 0:
                            dyn_colors = stat_colors.copy()
                            dyn_points = stat_points.copy()
                            dyn_coords_2d = coords_2d.reshape(-1, 2)[(accum_error == 0) & (motion_error == 0), :]
                            select_inds = random.sample(range(dyn_colors.shape[0]), N_pts)
                        else:
                            select_inds = random.choices(range(dyn_colors.shape[0]), k=N_pts)
                    else:
                        select_inds = random.sample(range(dyn_colors.shape[0]), N_pts)

                    dyn_time = torch.ones(dyn_colors[select_inds].shape[0], 1) * viewpoint_stack[IDX].time
                    dyn_color = dyn_colors[select_inds]
                    dyn_point = dyn_points[select_inds]
                    dyn_coord_2d = dyn_coords_2d[select_inds]  # N_pts, 2

            # compute dyn tracker
            tracklet = viewpoint_stack[0].target_tracks
            if isinstance(tracklet, np.ndarray):
                tracklet = torch.from_numpy(tracklet)
            tracklet = tracklet.cuda()
            num_times = tracklet.shape[0]
            start_tracklet = tracklet[0]  # time = 0 (cannonical)
            dyn_tracklet_index = (
                torch.square(torch.from_numpy(dyn_coord_2d[:, None]).cuda() - start_tracklet[None]).sum(-1).argmin(-1)
            )
            dyn_tracklet = torch.gather(
                tracklet[None].expand(dyn_coord_2d.shape[0], -1, -1, -1),
                2,
                dyn_tracklet_index[:, None, None, None].expand(-1, num_times, -1, 2),
            ).squeeze(
                2
            )  # N_dyn_pts, N_time, 2
            points_list = torch.from_numpy(np.stack(points_list, axis=0)).permute(0, 3, 1, 2)
            colors_list = torch.from_numpy(np.stack(colors_list, axis=0)).permute(0, 3, 1, 2)
            dyn_tracklet = dyn_tracklet.permute(1, 0, 2)[:, None]
            dyn_tracklet[..., 0] /= W
            dyn_tracklet[..., 1] /= H
            norm_dyn_tracklet = dyn_tracklet * 2 - 1.0  # norm to -1, 1
            dyn_tracjectory = (
                torch.nn.functional.grid_sample(points_list.cuda(), norm_dyn_tracklet, mode="nearest")
                .squeeze()
                .permute(2, 0, 1)
            )  # N_pts, N_times, 3

            # if viewpoint_stack[0].instance_mask is not None:
            if opt.use_instance_mask:
                total_mask_all_t = None
                total_mask_list = []
                for i in range(dyn_tracjectory.shape[1]):
                    current_traj = dyn_tracjectory[:, i, :]
                    # view_dir = current_traj - viewpoint_stack[i].camera_center.cuda()[None]

                    X_norm = norm_dyn_tracklet[i, 0, :, 0]
                    Y_norm = norm_dyn_tracklet[i, 0, :, 1]

                    pixel_coords = torch.stack([X_norm, Y_norm], dim=-1)
                    src_pixel_coords = pixel_coords[None, None].cuda()

                    total_mask = None
                    for instance_id in range(viewpoint_stack[i].instance_mask.shape[0]):
                        curr_instance_mask = viewpoint_stack[i].instance_mask[instance_id].permute(2, 0, 1)[None].cuda()
                        projected_instance_mask = (
                            torch.nn.functional.grid_sample(
                                curr_instance_mask, src_pixel_coords, mode="nearest", padding_mode="zeros"
                            )
                            .squeeze()
                            .bool()
                        )

                        if projected_instance_mask.any():
                            instance_and_depth_mask = torch.zeros_like(projected_instance_mask).bool()

                            # current_instance_viewdir = view_dir[projected_instance_mask]
                            # z_val = current_instance_viewdir.norm(dim=1)
                            z_val = current_traj[:, 2][projected_instance_mask]
                            mean_z_val = z_val.mean()
                            std_z_val = z_val.std()
                            instance_and_depth_mask[projected_instance_mask] = z_val < (mean_z_val + std_z_val)
                            if total_mask is None:
                                total_mask = instance_and_depth_mask
                            else:
                                total_mask = total_mask | instance_and_depth_mask

                    total_mask_list.append(total_mask)

                    if total_mask_all_t is None:
                        total_mask_all_t = total_mask
                    else:
                        total_mask_all_t = total_mask_all_t & total_mask
            else:
                total_mask_all_t = None
                total_mask_list = []
                for i in range(dyn_tracjectory.shape[1]):
                    current_traj = dyn_tracjectory[:, i, :]
                    X_norm = norm_dyn_tracklet[i, 0, :, 0]
                    Y_norm = norm_dyn_tracklet[i, 0, :, 1]

                    pixel_coords = torch.stack([X_norm, Y_norm], dim=-1)
                    src_pixel_coords = pixel_coords[None, None].cuda()

                    curr_mask = viewpoint_stack[i].mask[None].cuda()
                    erode_mask = kornia.morphology.erosion(curr_mask, kernel=torch.ones(5, 5).cuda())
                    
                    # breakpoint()
                    # _,_,knn = pytorch3d.ops.knn_points()
                    
                    projected_mask = (
                        torch.nn.functional.grid_sample(
                            erode_mask, src_pixel_coords, mode="nearest", padding_mode="zeros"
                        )
                        .squeeze()
                        .bool()
                    )
                    total_mask_list.append(projected_mask > 0)
                    if total_mask_all_t is None:
                        total_mask_all_t = projected_mask > 0
                    else:
                        total_mask_all_t = total_mask_all_t & (projected_mask > 0)

            if total_mask_all_t is None:
                total_mask_all_t = torch.ones(dyn_tracjectory.shape[0], dtype=torch.bool, device='cuda')
            new_dyn_tracjectory = dyn_tracjectory[total_mask_all_t]
            new_dyn_color = dyn_color[total_mask_all_t.cpu().numpy()]
            new_dyn_point = dyn_point[total_mask_all_t.cpu().numpy()]
            new_dyn_time = dyn_time[total_mask_all_t.cpu().numpy()]

            if new_dyn_point.shape[0] == 0:
                print("Warning: no dynamic points, using static points as fallback")
                n_fallback = min(stat_point.shape[0], dyn_tracjectory.shape[0])
                new_dyn_point = stat_point[:n_fallback].copy()
                new_dyn_color = stat_color[:n_fallback].copy()
                new_dyn_time = torch.ones((n_fallback, 1)) * viewpoint_stack[IDX].time
                new_dyn_tracjectory = dyn_tracjectory[:n_fallback]

            stat_pc = BasicPointCloud(colors=stat_color, points=stat_point, normals=None, times=stat_time)
            dyn_pc = BasicPointCloud(colors=new_dyn_color, points=new_dyn_point, normals=None, times=new_dyn_time)
            return stat_pc, dyn_pc, new_dyn_tracjectory
    elif stage == "fine_static":
        return stat_gaussians, dyn_gaussians
    else:
        return BEST_PSNR, BEST_ITER


def training(
    dataset,
    hyper,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    expname,
):
    tb_writer = prepare_output_and_logger(expname)
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)  # large CPU usage
    timer.start()
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())  # pose network with instance scaling
    stat_pc, dyn_pc, dyn_tracjectory = scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        dyn_gaussians,
        stat_gaussians,
        scene,
        "warm",
        tb_writer,
        opt.coarse_iterations,
        timer,
    )
    scene.cameras_extent = 1
    stat_gaussians.create_from_pcd(pcd=stat_pc, spatial_lr_scale=5, time_line=0)
    dyn_gaussians.create_from_pcd_dynamic(pcd=dyn_pc, spatial_lr_scale=5, time_line=0, dyn_tracjectory=dyn_tracjectory)
    
    stat_gaussians, dyn_gaussians = scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        dyn_gaussians,
        stat_gaussians,
        scene,
        "fine_static",
        tb_writer,
        opt.coarse_iterations,
        timer,
    )
    
    BEST_PSNR, BEST_ITER = scene_reconstruction(
        dataset,
        opt,
        hyper,
        pipe,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        debug_from,
        dyn_gaussians,
        stat_gaussians,
        scene,
        "fine",
        tb_writer,
        opt.iterations,
        timer,
    )

    return BEST_PSNR, BEST_ITER


def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    test_cams,
    renderFunc,
    background,
    stage,
    dataset_type,
    final_iter,
):
    if tb_writer:
        tb_writer.add_scalar(f"{stage}/train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(f"{stage}/train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)

    test_psnr = 0.0
    cur_iter = 0.0

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({"name": "test", "cameras": test_cams}, {"name": "train", "cameras": []})

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(viewpoint, scene.stat_gaussians, scene.dyn_gaussians, background)
                    image = render_pkg["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage + "/" + config["name"] + "_view_{}/render".format(viewpoint.image_name),
                                image[None],
                                global_step=iteration,
                            )
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage
                                    + "/"
                                    + config["name"]
                                    + "_view_{}/ground_truth".format(viewpoint.image_name),
                                    gt_image[None],
                                    global_step=iteration,
                                )
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(
                        stage + "/" + config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(stage + "/" + config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

            if config["name"] == "test":
                test_psnr = psnr_test
                cur_iter = iteration

        torch.cuda.empty_cache()
        return test_psnr, cur_iter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100 * i for i in range(1000)])
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[1000, 3000, 4000, 5000, 6000, 7_000, 9000, 10000, 12000, 14000, 15000, 20000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-render_process", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.set_num_threads(16)

    BEST_PSNR, BEST_ITER = training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
    )

    with open(os.path.join(args.model_path, "seed.txt"), 'a') as f:
        f.write("BEST PSNR : " + str(BEST_PSNR) + " SEED : " + str(args.seed) + "\n")

    # All done
    print("\nTraining complete.")
    print("BEST PSNR : ", BEST_PSNR)
    print("BEST ITER : ", BEST_ITER)