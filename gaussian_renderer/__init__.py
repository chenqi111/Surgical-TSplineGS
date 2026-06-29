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

import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization, fully_fused_projection
from scene.gaussian_model import GaussianModel
import cv2


# @torch.jit.script
def interpolate_cubic_hermite(signal, times, N):
    # start.record()
    times_scaled = times * (N - 1)[:, None]
    indices = torch.floor(times_scaled).long()
    # Clamping to avoid out-of-bounds indices

    indices = torch.clamp(
        indices, torch.zeros_like(N)[:, None].expand(-1, 3, -1), (N - 2)[:, None].expand(-1, 3, -1)
    ).long()
    left_indices = torch.clamp(
        indices - 1, torch.zeros_like(N)[:, None].expand(-1, 3, -1), (N - 1)[:, None].expand(-1, 3, -1)
    ).long()
    right_indices = torch.clamp(
        indices + 1, torch.zeros_like(N)[:, None].expand(-1, 3, -1), (N - 1)[:, None].expand(-1, 3, -1)
    ).long()
    right_right_indices = torch.clamp(
        indices + 2, torch.zeros_like(N)[:, None].expand(-1, 3, -1), (N - 1)[:, None].expand(-1, 3, -1)
    ).long()

    t = times_scaled - indices.float()
    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
    # if len(signal.shape) == 3:  # remove extra singleton dimension
    interpolation = interpolation.squeeze(-1)
    # end.record()
    # torch.cuda.synchronize()
    # print('v1:', start.elapsed_time(end))
    return interpolation


# @torch.jit.script
def interpolate_cubic_hermite_infer(signal, times, N, index_offset):
    # start.record()
    times_scaled = times * (N - 1)
    indices = torch.floor(times_scaled).long()

    # Clamping to avoid out-of-bounds indices
    indices = torch.clamp(indices, torch.zeros_like(N).expand(-1, 3), (N - 2).expand(-1, 3)).long()
    left_indices = torch.clamp(indices - 1, torch.zeros_like(N).expand(-1, 3), (N - 1).expand(-1, 3)).long()
    right_indices = torch.clamp(indices + 1, torch.zeros_like(N).expand(-1, 3), (N - 1).expand(-1, 3)).long()
    right_right_indices = torch.clamp(indices + 2, torch.zeros_like(N).expand(-1, 3), (N - 1).expand(-1, 3)).long()

    t = times_scaled - indices.float()
    p0 = torch.gather(signal, 0, left_indices + index_offset)
    p1 = torch.gather(signal, 0, indices + index_offset)
    p2 = torch.gather(signal, 0, right_indices + index_offset)
    p3 = torch.gather(signal, 0, right_right_indices + index_offset)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
    # end.record()
    # torch.cuda.synchronize()
    # print('v2:', start.elapsed_time(end))
    return interpolation


def compute_motion_type(control_xyz, current_control_num=None):
    """
    Compute motion type and magnitude for each dynamic Gaussian (fully vectorized).
    Only uses valid control points for each Gaussian.
    Returns:
        motion_score: in [0, 1], higher = more like rigid-body motion (large, straight displacement)
        motion_magnitude: total path length (sum of chord lengths between control points)
        rigidity_score: straightness * smoothness in [0, 1]
        velocity_consistency: how consistent the velocity magnitude is across segments
    """
    B, N, _ = control_xyz.shape

    if current_control_num is not None:
        ccn = current_control_num.squeeze().long()
        valid_mask = torch.arange(N, device=control_xyz.device)[None, :] < ccn[:, None]
    else:
        valid_mask = torch.ones(B, N, dtype=torch.bool, device=control_xyz.device)

    diffs = control_xyz[:, 1:, :] - control_xyz[:, :-1, :]
    valid_diffs_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    diffs = diffs * valid_diffs_mask.unsqueeze(-1).float()

    chord_lengths = torch.norm(diffs, dim=-1)
    total_path = chord_lengths.sum(dim=-1)

    first_valid = valid_mask.float().argmax(dim=-1)
    last_valid = N - 1 - valid_mask.float().flip(dims=[-1]).argmax(dim=-1)
    batch_idx = torch.arange(B, device=control_xyz.device)
    first_pt = control_xyz[batch_idx, first_valid]
    last_pt = control_xyz[batch_idx, last_valid]
    net_displacement = torch.norm(last_pt - first_pt, dim=-1)

    straightness = net_displacement / (total_path + 1e-8)

    dirs = diffs / (chord_lengths[..., None] + 1e-8)
    dirs = dirs * valid_diffs_mask.unsqueeze(-1).float()

    valid_dir_pairs = valid_diffs_mask[:, :-1] & valid_diffs_mask[:, 1:]
    cos_angles = (dirs[:, :-1, :] * dirs[:, 1:, :]).sum(dim=-1)
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    cos_angles = cos_angles * valid_dir_pairs.float()
    cos_count = valid_dir_pairs.float().sum(dim=-1).clamp(min=1)
    smoothness = cos_angles.sum(dim=-1) / cos_count
    smoothness = (smoothness + 1) / 2

    rigidity_score = straightness * smoothness

    vel_mean = total_path / valid_diffs_mask.float().sum(dim=-1).clamp(min=1)
    vel_var = ((chord_lengths - vel_mean[:, None]) ** 2) * valid_diffs_mask.float()
    vel_std = (vel_var.sum(dim=-1) / valid_diffs_mask.float().sum(dim=-1).clamp(min=1)).sqrt() + 1e-8
    velocity_consistency = 1.0 / (1.0 + vel_std / (vel_mean + 1e-8))
    velocity_consistency[valid_diffs_mask.float().sum(dim=-1) < 2] = 1.0

    bend_angles = torch.acos(torch.clamp(cos_angles, -1.0, 1.0))
    bend_angles = bend_angles * valid_dir_pairs.float()
    bending_energy = bend_angles.sum(dim=-1) / valid_dir_pairs.float().sum(dim=-1).clamp(min=1)

    total_path_min = total_path.min()
    total_path_max = total_path.max()
    path_range = total_path_max - total_path_min
    normalized_magnitude = (total_path - total_path_min) / (path_range + 1e-8)

    normalized_bending = 1.0 / (1.0 + bending_energy)

    motion_score = (
        0.35 * normalized_magnitude +
        0.35 * rigidity_score +
        0.20 * velocity_consistency +
        0.10 * normalized_bending
    )

    return motion_score, total_path, rigidity_score, velocity_consistency


def render(
    viewpoint_camera,
    stat_pc: GaussianModel,
    dyn_pc: GaussianModel,
    bg_color: torch.Tensor,
    get_static=False,
    get_dynamic=False,
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Get dyn variables
    means3D = dyn_pc.get_xyz.detach()
    no_dyn_gs = means3D.shape[0]
    scales = dyn_pc._scaling
    rotations = dyn_pc._rotation
    opacity = dyn_pc.get_opacity

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    pointtimes = (
        torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    )  #

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    K = viewpoint_camera.K.to(viewmat.device)
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    trbfdistanceoffset = viewpoint_camera.time * pointtimes
    tforpoly = trbfdistanceoffset.detach()
    rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    control_xyz = dyn_pc.get_control_xyz.cuda()
    curr_time = torch.tensor(viewpoint_camera.time).cuda()
    deform_means3D = interpolate_cubic_hermite(
        control_xyz.permute(0, 2, 1),
        curr_time[None, None].expand(control_xyz.shape[0], 3, 1),
        N=dyn_pc.current_control_num,
    )
    means3D = deform_means3D * dyn_pc.deform_spatial_scale

    # Apply activations
    scales = dyn_pc.scaling_activation(scales)
    rotations = dyn_pc.rotation_activation(rotations)
    colors_precomp = dyn_pc.get_features(tforpoly)

    smeans3D_final, sscales_final, srotations_final, sopacity_final = (
        stat_means3D,
        stat_scales,
        stat_rotations,
        stat_opacity,
    )
    means3D_final, scales_final, rotations_final, opacity_final = means3D, scales, rotations, opacity
    
    # instance_mask = cv2.imread("/home/nerf/VIC-3DGS/data/nvidia_rodynrf/Balloon2/instance_mask_manual/000/001.png")
    # instance_mask = torch.from_numpy(instance_mask[...,0]).cuda()[None,None] / 255.0
    
    # _, means2D_final, _, _, _ = fully_fused_projection(
    #     means = means3D_final,
    #     covars=None,
    #     quats=rotations_final,
    #     scales=scales_final,
    #     viewmats = viewmat[None],
    #     Ks=K[None],  # [C, 3, 3]
    #     width=int(viewpoint_camera.image_width),
    #     height=int(viewpoint_camera.image_height),
    # ) # B, N, 2

    # means2D_final[..., 0] = means2D_final[...,0] / int(viewpoint_camera.image_width)
    # means2D_final[..., 1] = means2D_final[...,1] / int(viewpoint_camera.image_height)
    # means2D_final = 2*means2D_final[None] - 1.0  
    # pts_mask = torch.nn.functional.grid_sample(
    #     instance_mask,
    #     means2D_final,
    #     mode='nearest',
    #     padding_mode='zeros',
    #     align_corners=False
    # )

    # valid_index = (pts_mask > 0.0).squeeze()    

    d_alpha = None
    if get_dynamic:
        dmeans3D_final = means3D_final
        dscales_final = scales_final
        drotations_final = rotations_final
        dopacity_final = opacity_final
        dcolors_precomp = colors_precomp

        d_img, _, _ = rasterization(
            means=dmeans3D_final,
            quats=drotations_final,
            scales=dscales_final,
            opacities=dopacity_final.squeeze(-1),
            colors=dcolors_precomp,
            backgrounds=bg_color[None],
            viewmats=viewmat[None].detach(),  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )

        d_depth = d_img[..., -1]
        d_img = d_img[..., :-1].permute(0, 3, 1, 2)
        d_image = dyn_pc.rgbdecoder(d_img, viewpoint_camera.cam_ray.cuda())
        d_image = d_image.squeeze(0)

    # Combine stat and dyn gaussians
    means3D_final = torch.cat((smeans3D_final, means3D_final), 0)
    scales_final = torch.cat((sscales_final, scales_final), 0)
    rotations_final = torch.cat((srotations_final, rotations_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)
    
    

    # means3D = dyn_pc.get_xyz.detach()
    # no_dyn_gs = means3D.shape[0]
    # scales = dyn_pc._scaling
    # rotations = dyn_pc._rotation
    # opacity = dyn_pc.get_opacity
    # pointtimes = (
    #     torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    # )  #

    # viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    # K = viewpoint_camera.K.to(viewmat.device)
    # bg_color = bg_color[:3]
    # bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    # trbfcenter = dyn_pc.get_trbfcenter
    # trbfdistanceoffset = 1.0 * pointtimes - trbfcenter
    # tforpoly = trbfdistanceoffset.detach()
    # rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    # control_xyz = dyn_pc.get_control_xyz.cuda()
    # curr_time = torch.tensor(1.0).cuda()
    # deform_means3D = interpolate_cubic_hermite(
    #     control_xyz.permute(0, 2, 1),
    #     curr_time[None, None].expand(control_xyz.shape[0], 3, 1),
    #     N=dyn_pc.current_control_num,
    # )
    # means3D = deform_means3D * dyn_pc.deform_spatial_scale

    # # Apply activations
    # scales = dyn_pc.scaling_activation(scales)
    # rotations = dyn_pc.rotation_activation(rotations)
    # colors_precomp_2 = dyn_pc.get_features(tforpoly)
    # means3D_final_2, scales_final_2, rotations_final_2, opacity_final_2 = means3D, scales, rotations, opacity
    
    # means3D_final_2 = means3D_final_2[valid_index]
    # scales_final_2 = scales_final_2[valid_index]
    # rotations_final_2 = rotations_final_2[valid_index]
    # opacity_final_2 = opacity_final_2[valid_index]
    # colors_precomp_2 = colors_precomp_2[valid_index]
    
    # means3D_final = torch.cat((means3D_final, means3D_final_2), 0)
    # scales_final = torch.cat((scales_final, scales_final_2), 0)
    # rotations_final = torch.cat((rotations_final, rotations_final_2), 0)
    # opacity_final = torch.cat((opacity_final, opacity_final_2), 0)
    # colors_precomp_final = torch.cat((colors_precomp_final, colors_precomp_2), 0)
    
    
    # means3D = dyn_pc.get_xyz.detach()
    # no_dyn_gs = means3D.shape[0]
    # scales = dyn_pc._scaling
    # rotations = dyn_pc._rotation
    # opacity = dyn_pc.get_opacity
    # pointtimes = (
    #     torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    # )  #

    # viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    # K = viewpoint_camera.K.to(viewmat.device)
    # bg_color = bg_color[:3]
    # bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    # trbfcenter = dyn_pc.get_trbfcenter
    # trbfdistanceoffset = (3/11) * pointtimes - trbfcenter
    # tforpoly = trbfdistanceoffset.detach()
    # rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    # control_xyz = dyn_pc.get_control_xyz.cuda()
    # curr_time = torch.tensor((3/11)).cuda()
    # deform_means3D = interpolate_cubic_hermite(
    #     control_xyz.permute(0, 2, 1),
    #     curr_time[None, None].expand(control_xyz.shape[0], 3, 1),
    #     N=dyn_pc.current_control_num,
    # )
    # means3D = deform_means3D * dyn_pc.deform_spatial_scale

    # # Apply activations
    # scales = dyn_pc.scaling_activation(scales)
    # rotations = dyn_pc.rotation_activation(rotations)
    # colors_precomp_2 = dyn_pc.get_features(tforpoly)
    # means3D_final_2, scales_final_2, rotations_final_2, opacity_final_2 = means3D, scales, rotations, opacity
    
    # means3D_final_2 = means3D_final_2[valid_index]
    # scales_final_2 = scales_final_2[valid_index]
    # rotations_final_2 = rotations_final_2[valid_index]
    # opacity_final_2 = opacity_final_2[valid_index]
    # colors_precomp_2 = colors_precomp_2[valid_index]
    
    # means3D_final = torch.cat((means3D_final, means3D_final_2), 0)
    # scales_final = torch.cat((scales_final, scales_final_2), 0)
    # rotations_final = torch.cat((rotations_final, rotations_final_2), 0)
    # opacity_final = torch.cat((opacity_final, opacity_final_2), 0)
    # colors_precomp_final = torch.cat((colors_precomp_final, colors_precomp_2), 0)
    

    rendered_image, _, info = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color[None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB+ED",
        absgrad=True,
    )

    depth = rendered_image[..., -1]
    rendered_image = rendered_image[..., :-1].permute(0, 3, 1, 2)
    radii = info["radii"].squeeze(0)

    # info["means2d"].retain_grad()
    try:
        info["means2d"].retain_grad()
    except:
        pass

    # rendered_image = torch.cat((rendered_image1, rendered_image2, rendered_image3), dim=0)
    rendered_image = dyn_pc.rgbdecoder(rendered_image, viewpoint_camera.cam_ray.cuda())
    rendered_image = rendered_image.squeeze(0)

    # MG-MAS: compute per-Gaussian mask overlap from 2D projection
    mg_mas_overlap = None
    if hasattr(viewpoint_camera, 'mask') and viewpoint_camera.mask is not None:
        with torch.no_grad():
            means2d = info["means2d"].squeeze(0)  # [N, 2]
            valid = (means2d[:, 0] >= 0) & (means2d[:, 0] < viewpoint_camera.image_width) & \
                    (means2d[:, 1] >= 0) & (means2d[:, 1] < viewpoint_camera.image_height)
            
            if valid.any():
                mask_tensor = viewpoint_camera.mask.to(device=means2d.device)  # [1, H, W]
                if mask_tensor.dim() == 3:
                    mask_tensor = mask_tensor.squeeze(0)  # [H, W]
                
                # Sample mask at Gaussian 2D centers
                x_norm = 2.0 * means2d[valid, 0] / (viewpoint_camera.image_width - 1) - 1.0
                y_norm = 2.0 * means2d[valid, 1] / (viewpoint_camera.image_height - 1) - 1.0
                grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N_valid, 2]
                
                sampled_mask = F.grid_sample(
                    mask_tensor.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                    grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                ).view(-1)  # [N_valid]
                
                mg_mas_overlap = torch.zeros(means2d.shape[0], device=means2d.device)
                mg_mas_overlap[valid] = sampled_mask.clamp(0, 1)

    d_alpha, _, _ = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=torch.cat([torch.zeros(no_stat_gs, 1), torch.ones(means3D_final.shape[0] - no_stat_gs, 1)], dim=0).cuda(),
        backgrounds=bg_color[0:1][None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    d_alpha = d_alpha[..., 0]

    # Render Nc (number of control points) heatmap
    nc_colors = torch.cat(
        [torch.zeros(no_stat_gs, 1).cuda(), dyn_pc.current_control_num.float().cuda()], dim=0
    )
    nc_heatmap, _, _ = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=nc_colors,
        backgrounds=bg_color[0:1][None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    nc_heatmap = nc_heatmap[..., 0]

    # Classify dynamic Gaussians into 3 categories: static=0, tissue=0.5, instrument=1.0
    with torch.no_grad():
        motion_score, motion_magnitude, rigidity_score, velocity_consistency = \
            compute_motion_type(dyn_pc.get_control_xyz, dyn_pc.current_control_num)

    ccn = dyn_pc.current_control_num.squeeze()

    mag_min = motion_magnitude.min()
    mag_max = motion_magnitude.max()
    mag_range = mag_max - mag_min
    norm_mag = (motion_magnitude - mag_min) / (mag_range + 1e-8)

    if ccn.max() > ccn.min():
        norm_ccn = (ccn.float() - ccn.min().float()) / (ccn.max().float() - ccn.min().float() + 1e-8)
        ccn_score = 1.0 - norm_ccn
    else:
        ccn_score = torch.ones_like(ccn).float() * 0.5

    # Combined instrument likelihood score
    # Instruments: few control points, high rigidity, high velocity consistency
    instr_likelihood = (
        0.30 * ccn_score +
        0.30 * rigidity_score +
        0.20 * velocity_consistency +
        0.20 * motion_score
    )

    # Adaptive threshold based on distribution
    instr_median = instr_likelihood.median()
    instr_mad = (instr_likelihood - instr_median).abs().median() + 1e-8
    instr_threshold = instr_median + 0.5 * instr_mad
    is_instrument = (norm_mag > 0.05) & (instr_likelihood > instr_threshold)

    # Render instrument contribution per pixel (instrument=1, static+tissue=0)
    instr_colors = torch.zeros((no_stat_gs + no_dyn_gs, 1), device="cuda")
    instr_colors[no_stat_gs:] = is_instrument.float()[:, None]
    instr_alpha_map, _, _ = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=instr_colors,
        backgrounds=bg_color[0:1][None],
        viewmats=viewmat[None].detach(),
        Ks=K[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    instr_alpha_map = instr_alpha_map[..., 0]

    # Also render tissue-only for comparison
    tissue_colors = torch.zeros((no_stat_gs + no_dyn_gs, 1), device="cuda")
    tissue_colors[no_stat_gs:] = (~is_instrument).float()[:, None]
    tissue_alpha_map, _, _ = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=tissue_colors,
        backgrounds=bg_color[0:1][None],
        viewmats=viewmat[None].detach(),
        Ks=K[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    tissue_alpha_map = tissue_alpha_map[..., 0]

    # Per-pixel label via argmax over class contributions
    static_contrib = torch.clamp(1.0 - d_alpha, 0, 1)
    dyn_total = d_alpha
    instr_contrib = instr_alpha_map
    tissue_contrib = torch.clamp(dyn_total - instr_contrib, 0, 1)
    motion_type_label = torch.zeros_like(d_alpha)
    is_tissue = (tissue_contrib > static_contrib) & (tissue_contrib >= instr_contrib)
    is_instr_ = (instr_contrib > static_contrib) & (instr_contrib > tissue_contrib)
    motion_type_label[is_tissue] = 0.5
    motion_type_label[is_instr_] = 1.0

    # Dynamic-only label map (classes 1 and 2)
    dyn_label = torch.zeros_like(d_alpha)
    if get_dynamic:
        inst_dyn = is_instrument.float()[:, None]
        dyn_inst_alpha, _, _ = rasterization(
            means=dmeans3D_final,
            quats=drotations_final,
            scales=dscales_final,
            opacities=dopacity_final.squeeze(-1),
            colors=inst_dyn,
            backgrounds=bg_color[0:1][None],
            viewmats=viewmat[None].detach(),
            Ks=K[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB",
        )
        dyn_inst_alpha = dyn_inst_alpha[..., 0]
        dyn_label = torch.where(dyn_inst_alpha > 0.5, torch.tensor(0.5, device="cuda"), torch.tensor(0.25, device="cuda"))

    s_rendered_image = None
    s_depth = None
    s_alpha = None
    if get_static:
        s_rendered_image, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=stat_colors_precomp,
            backgrounds=bg_color[None],
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )
        s_depth = s_rendered_image[..., -1]
        s_rendered_image = s_rendered_image[..., :-1].permute(0, 3, 1, 2)
        s_rendered_image = dyn_pc.rgbdecoder(s_rendered_image, viewpoint_camera.cam_ray.cuda())
        s_rendered_image = s_rendered_image.squeeze(0)

        s_alpha, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=torch.ones(stat_colors_precomp.shape[0], 1).cuda(),
            backgrounds=bg_color[0:1][None],
            viewmats=viewmat[None].detach(),  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB",
        )
        s_alpha = s_alpha[..., 0]

    return {
        "render": rendered_image,
        "depth": depth,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "s_render": s_rendered_image,
        "s_depth": s_depth,
        "s_alpha": s_alpha,
        "d_render": d_image if get_dynamic else None,
        "d_depth": d_depth if get_dynamic else None,
        "d_alpha": d_alpha,
        "d_means3d": dmeans3D_final if get_dynamic else None,
        "nc_heatmap": nc_heatmap,
        "motion_type": motion_type_label,
        "dyn_motion_type": dyn_label,
        "motion_score": motion_score,
        "rigidity_score": rigidity_score,
        "velocity_consistency": velocity_consistency,
        "instr_likelihood": instr_likelihood,
        "norm_motion_magnitude": norm_mag,
        "mg_mas_overlap": mg_mas_overlap,
    }

def render_static(
    viewpoint_camera,
    stat_pc: GaussianModel,
    dyn_pc: GaussianModel,
    bg_color: torch.Tensor,
    get_static=False,
    get_dynamic=False,
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(stat_means3D.device)
    K = viewpoint_camera.K.to(viewmat.device)
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    smeans3D_final, sscales_final, srotations_final, sopacity_final = (
        stat_means3D,
        stat_scales,
        stat_rotations,
        stat_opacity,
    )

    # Combine stat and dyn gaussians
    means3D_final = smeans3D_final
    scales_final = sscales_final
    rotations_final = srotations_final
    opacity_final = sopacity_final
    colors_precomp_final = stat_colors_precomp

    rendered_image, _, info = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color[None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB+ED",
        absgrad=True,
    )

    depth = rendered_image[..., -1]
    rendered_image = rendered_image[..., :-1].permute(0, 3, 1, 2)
    radii = info["radii"].squeeze(0)

    # info["means2d"].retain_grad()
    try:
        info["means2d"].retain_grad()
    except:
        pass

    # rendered_image = torch.cat((rendered_image1, rendered_image2, rendered_image3), dim=0)
    rendered_image = dyn_pc.rgbdecoder(rendered_image, viewpoint_camera.cam_ray.cuda())
    rendered_image = rendered_image.squeeze(0)

    s_rendered_image = None
    s_depth = None
    s_alpha = None
    if get_static:
        s_rendered_image, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=stat_colors_precomp,
            backgrounds=bg_color[None],
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )
        s_depth = s_rendered_image[..., -1]
        s_rendered_image = s_rendered_image[..., :-1].permute(0, 3, 1, 2)
        s_rendered_image = dyn_pc.rgbdecoder(s_rendered_image, viewpoint_camera.cam_ray.cuda())
        s_rendered_image = s_rendered_image.squeeze(0)

        s_alpha, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=torch.ones(stat_colors_precomp.shape[0], 1).cuda(),
            backgrounds=bg_color[0:1][None],
            viewmats=viewmat[None].detach(),  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB",
        )
        s_alpha = s_alpha[..., 0]

    return {
        "render": rendered_image,
        "depth": depth,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "s_render": s_rendered_image,
        "s_depth": s_depth,
        "s_alpha": s_alpha,
        "d_render": None,
        "d_depth": None,
        "d_alpha":  None,
        "d_means3d": None,
    }


def render_infer(viewpoint_camera,
    stat_pc: GaussianModel,
    dyn_pc: GaussianModel,
    bg_color: torch.Tensor,
    ):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # Get dyn variables
    means3D = dyn_pc.get_xyz.detach()
    no_dyn_gs = means3D.shape[0]
    scales = dyn_pc._scaling
    rotations = dyn_pc._rotation
    opacity = dyn_pc.get_opacity

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    pointtimes = (
        torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    )  #

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    K = viewpoint_camera.K.to(viewmat.device)
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    trbfdistanceoffset = viewpoint_camera.time * pointtimes
    tforpoly = trbfdistanceoffset.detach()
    rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    control_xyz = dyn_pc.flat_control_xyz.cuda()
    # start.record()
    deform_means3D = interpolate_cubic_hermite_infer(control_xyz, torch.tensor(viewpoint_camera.time).cuda()[None].expand(means3D.shape[0], 3), N=dyn_pc.current_control_num, index_offset=dyn_pc.index_offset)
    # end.record()
    # torch.cuda.synchronize()
    # print('spline time:', start.elapsed_time(end) / means3D.shape[0])
    # duration = start.elapsed_time(end) / means3D.shape[0]

    means3D = deform_means3D * 1e-2
    # Apply activations
    scales = dyn_pc.scaling_activation(scales)
    rotations = dyn_pc.rotation_activation(rotations)
    colors_precomp = dyn_pc.get_features(viewpoint_camera.time)

    smeans3D_final, sscales_final, srotations_final, sopacity_final = stat_means3D, stat_scales, stat_rotations, stat_opacity
    means3D_final, scales_final, rotations_final, opacity_final = means3D, scales, rotations, opacity


    # Combine stat and dyn gaussians
    means3D_final = torch.cat((smeans3D_final, means3D_final), 0)
    scales_final = torch.cat((sscales_final, scales_final), 0)
    rotations_final = torch.cat((srotations_final, rotations_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)

    rendered_image, _, _ = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color[None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    
    rendered_image = rendered_image.permute(0, 3, 1, 2)
    rendered_image = dyn_pc.rgbdecoder(rendered_image, viewpoint_camera.cam_ray.cuda())
    rendered_image = rendered_image.squeeze(0)

    return {
        "render": rendered_image,
    }
