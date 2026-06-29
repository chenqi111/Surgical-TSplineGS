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

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scene.deformation import pose_network
from simple_knn._C import distCUDA2
from torch import nn
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud, pts2pixel
from utils.model_utils import getcolormodel
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

def controlgaussians(opt, gaussians, densify, iteration, scene, flag, is_dynamic=False):
    if densify == 1:  # nvidia
        if iteration < opt.densify_until_iter:
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    grad_thres = opt.densify_grad_threshold if not is_dynamic else opt.densify_grad_threshold_dynamic
                    gaussians.densify_pruneclone(grad_thres, opt.opthr, scene.cameras_extent, size_threshold, splitN=2)
                    flag += 1
                else:
                    prune_mask = (gaussians.get_opacity < opt.opthr).squeeze()
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()

            if iteration % opt.opacity_reset_interval == 0:
                gaussians.reset_opacity()

        return flag
    else:
        raise NotImplementedError


def inverse_cubic_hermite(curves, times, N_pts=5, scale=0.8, return_error=False):
    # times = (times - 0.5) * scale + 0.5
    # inverse cubic Hermite splines
    transform_matrix = torch.zeros((times.shape[0], times.shape[1], N_pts), device=curves.device)  # B, T, N_pts
    N = N_pts

    times_scaled = times * (N - 1)
    indices = torch.floor(times_scaled).long()

    # Clamping to avoid out-of-bounds indices
    indices = torch.clamp(indices, 0, N - 2)
    left_indices = torch.clamp(indices - 1, 0, N - 1)
    right_indices = torch.clamp(indices + 1, 0, N - 1)
    right_right_indices = torch.clamp(indices + 2, 0, N - 1)

    t = times_scaled - indices.float()
    # Hermite basis functions
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    p1_coef = h00  # B, T, 1
    p0_coef = torch.zeros_like(h00)
    p2_coef = h01
    p3_coef = torch.zeros_like(h00)

    # One-sided derivatives at the boundaries
    h10_add_p0 = torch.where(left_indices == indices, 0, -h10 / 2)
    h10_add_p1 = torch.where(left_indices == indices, -h10, 0)
    h10_add_p2 = torch.where(left_indices == indices, h10, h10 / 2)

    h11_add_p1 = torch.where(right_right_indices == right_indices, -h11, -h11 / 2)
    h11_add_p2 = torch.where(right_right_indices == right_indices, h11, 0)
    h11_add_p3 = torch.where(right_right_indices == right_indices, 0, h11 / 2)

    p0_coef = p0_coef + h10_add_p0
    p1_coef = p1_coef + h10_add_p1 + h11_add_p1
    p2_coef = p2_coef + h10_add_p2 + h11_add_p2
    p3_coef = p3_coef + h11_add_p3

    # scatter to transform matrix
    transform_matrix = torch.scatter_reduce(input=transform_matrix, dim=-1, index=left_indices, src=p0_coef, reduce="sum")
    transform_matrix = torch.scatter_reduce(input=transform_matrix, dim=-1, index=indices, src=p1_coef, reduce="sum")
    transform_matrix = torch.scatter_reduce(input=transform_matrix, dim=-1, index=right_indices, src=p2_coef, reduce="sum")
    transform_matrix = torch.scatter_reduce(
        input=transform_matrix, dim=-1, index=right_right_indices, src=p3_coef, reduce="sum"
    )

    control_pts = torch.linalg.lstsq(transform_matrix, curves).solution

    if return_error:
        error = torch.dist(control_pts, torch.linalg.pinv(transform_matrix) @ curves)
        return control_pts, error

    return control_pts


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.active_sh_degree = 0
        self.control_num = args.control_num
        self.deform_spatial_scale = args.deform_spatial_scale
        self.error_threshold = args.prune_error_threshold
        self.current_control_num = torch.tensor(self.control_num, device="cuda").repeat(1)
        self.max_sh_degree = args.sh_degree
        self._xyz = torch.empty(0)
        self.control_xyz = torch.empty(0)

        self._posenet = None

        # TASS (Topology-Aware Spline Splitting) state
        self.tass_error_buffer = torch.empty(0)
        self.tass_error_count = torch.empty(0)
        self.tass_split_flags = torch.empty(0)
        self.tass_epsilon = 0.05
        self.tass_gamma = 2.0

        # MG-MAS mask overlap buffer (per Gaussian, updated each iter)
        self.mg_mas_overlap = torch.empty(0)
        # self.grid = TriPlaneGrid()
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.split_depth = torch.empty(0)
        self.setup_functions()
        self._track_split_depth = False

        self._omega = torch.empty(0)
        self.delta_t = None
        self.omegamask = None
        self.maskforems = None
        self.distancetocamera = None
        self.trbfslinit = None
        self.ts = None
        self.trbfoutput = None
        self.preprocesspoints = False
        self.addsphpointsscale = 0.8

        self.maxz, self.minz = 0.0, 0.0
        self.maxy, self.miny = 0.0, 0.0
        self.maxx, self.minx = 0.0, 0.0
        self.raystart = 0.7
        self.computedopacity = None
        self.computedscales = None

        self.rgbdecoder = getcolormodel(args.rgbfuntion)

    def create_pose_network(self, args, train_cams):
        self._posenet = pose_network(args, train_cams=train_cams).to("cuda")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation_stat(self):
        return self.rotation_activation(self._rotation)

    def get_rotation(self, delta_t=None):
        rotation = self._rotation + delta_t * self._omega
        self.delta_t = delta_t
        return self.rotation_activation(rotation)

    def get_rotation_dy(self, rotation, delta_t):
        new_rotation = rotation + delta_t * self._omega
        return new_rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_control_xyz(self):
        return self.control_xyz

    def get_features(self, deltat):
        return torch.cat((self._features_dc, deltat * self._features_t), dim=1)

    @property
    def get_features_static(self):
        return torch.cat((self._features_dc, 0.0 * self._features_t), dim=1)

    @property
    def get_blending(self):
        return self._features_rest[:, -1:, 0]

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def flatten_control_point(self):
        flat_control_point = []
        for i in range(self.control_xyz.shape[0]):
            current_control_xyz = self.control_xyz[i][: self.current_control_num.squeeze().long()[i]]
            flat_control_point.append(current_control_xyz)
        self.flat_control_xyz = torch.cat(flat_control_point, dim=0).contiguous()
        self.index_offset = (
            torch.cat(
                [torch.zeros(1).cuda(), torch.cumsum(self.current_control_num.squeeze()[:-1], dim=0)], dim=0
            ).long()
        )[:, None]

    def add_dummy_control_point(self):
        self.dummy_control_xyz = torch.cat(
            [self.control_xyz, torch.zeros(self.control_xyz.shape[0], 100, 3).cuda()], dim=1
        )

    def onedown_control_pts(self, viewpoints):
        dummy_step = torch.arange(0, self.control_num, 1).cuda().float()[None].repeat(self.control_xyz.shape[0], 1)
        time_step = 1 / (self.current_control_num.squeeze(-1) - 1.0)
        t_step = (dummy_step * time_step[..., None])[..., None]  # t_step corresponding to the current control points
        new_control_num = self.current_control_num - 1
        new_control_num[new_control_num < 4] = 4
        new_control_pts_value = self.inverse_cubic_hermite_for_prune(
            self.control_xyz, t_step, N_pts=new_control_num
        )  # reduce 1 control point
        new_control_pts = self.control_xyz.clone()
        new_control_pts[:, : self.control_num - 1] = new_control_pts_value

        # update current control number and control points: using 2d projection?
        error = self.compute_prune_error(new_control_pts, new_control_num, viewpoints)
        error_threshold = self.error_threshold
        self.current_control_num[error <= error_threshold] = new_control_num[error <= error_threshold]
        self.control_xyz[error <= error_threshold] = new_control_pts[error <= error_threshold]
        print("One down control points: ", (error <= error_threshold).sum().cpu().numpy())

    def compute_prune_error(self, new_control_pts, new_control_num, viewpoints):
        K = torch.zeros(3, 3).type_as(self.control_xyz)
        K[0, 0] = float(self._posenet.focal_bias.exp())
        K[1, 1] = float(self._posenet.focal_bias.exp())
        K[0, 2] = float(viewpoints[0].image_width / 2)
        K[1, 2] = float(viewpoints[0].image_height / 2)
        K[2, 2] = float(1)
        pix_err_list = []
        for idx, viewpoint in enumerate(viewpoints):
            if idx == 0 or idx == len(viewpoints) - 1:
                continue  # skip first and last frame
            deform_means3D = (
                self.interpolate_cubic_hermite(
                    self.control_xyz.permute(0, 2, 1),
                    torch.tensor(viewpoint.time).cuda()[None, None].expand(self.control_xyz.shape[0], 3, 1),
                    N=self.current_control_num,
                )
                * self.deform_spatial_scale
            )
            new_deform_means3D = (
                self.interpolate_cubic_hermite(
                    new_control_pts.permute(0, 2, 1),
                    torch.tensor(viewpoint.time).cuda()[None, None].expand(new_control_pts.shape[0], 3, 1),
                    N=new_control_num,
                )
                * self.deform_spatial_scale
            )
            deform_means2D = pts2pixel(deform_means3D, viewpoint, K)
            new_deform_means2D = pts2pixel(new_deform_means3D, viewpoint, K)
            pix_err_list.append(torch.norm(deform_means2D - new_deform_means2D, dim=-1))
        return torch.stack(pix_err_list, dim=0).mean(0)

    def inverse_cubic_hermite_for_prune(self, curves, times, N_pts):
        transform_matrix = torch.zeros(
            (times.shape[0], self.control_num, self.control_num - 1), device=curves.device
        )  # B, T, N_pts always maximmum entries
        dummy_transform_matrix = torch.diag(torch.ones(self.control_num - 1, device=curves.device), -1)[None].repeat(
            times.shape[0], 1, 1
        )[:, :, :-1]
        # dummy_eq = torch.zeros(self.control_num, device=curves.device)
        # dummy_eq[-1] = 1
        N = N_pts

        times_scaled = times * (N - 1)[:, None]
        indices = torch.floor(times_scaled).long()
        # Clamping to avoid out-of-bounds indices
        indices = torch.clamp(
            indices,
            torch.zeros_like(N)[:, None].expand(-1, self.control_num, -1),
            (N - 2)[:, None].expand(-1, self.control_num, -1),
        ).long()
        left_indices = torch.clamp(
            indices - 1,
            torch.zeros_like(N)[:, None].expand(-1, self.control_num, -1),
            (N - 1)[:, None].expand(-1, self.control_num, -1),
        ).long()
        right_indices = torch.clamp(
            indices + 1,
            torch.zeros_like(N)[:, None].expand(-1, self.control_num, -1),
            (N - 1)[:, None].expand(-1, self.control_num, -1),
        ).long()
        right_right_indices = torch.clamp(
            indices + 2,
            torch.zeros_like(N)[:, None].expand(-1, self.control_num, -1),
            (N - 1)[:, None].expand(-1, self.control_num, -1),
        ).long()

        t = times_scaled - indices.float()
        # Hermite basis functions
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t**2 * (3 - 2 * t)
        h11 = t**2 * (t - 1)

        p1_coef = h00  # B, T, 1
        p0_coef = torch.zeros_like(h00)
        p2_coef = h01
        p3_coef = torch.zeros_like(h00)

        # One-sided derivatives at the boundaries
        h10_add_p0 = torch.where(left_indices == indices, 0, -h10 / 2)
        h10_add_p1 = torch.where(left_indices == indices, -h10, 0)
        h10_add_p2 = torch.where(left_indices == indices, h10, h10 / 2)

        h11_add_p1 = torch.where(right_right_indices == right_indices, -h11, -h11 / 2)
        h11_add_p2 = torch.where(right_right_indices == right_indices, h11, 0)
        h11_add_p3 = torch.where(right_right_indices == right_indices, 0, h11 / 2)

        p0_coef = p0_coef + h10_add_p0
        p1_coef = p1_coef + h10_add_p1 + h11_add_p1
        p2_coef = p2_coef + h10_add_p2 + h11_add_p2
        p3_coef = p3_coef + h11_add_p3

        # scatter to transform matrix
        transform_matrix = torch.scatter_reduce(input=transform_matrix, dim=-1, index=left_indices, src=p0_coef, reduce="sum")
        transform_matrix = torch.scatter_reduce(input=transform_matrix, dim=-1, index=indices, src=p1_coef, reduce="sum")
        transform_matrix = torch.scatter_reduce(
            input=transform_matrix, dim=-1, index=right_indices, src=p2_coef, reduce="sum"
        )
        transform_matrix = torch.scatter_reduce(
            input=transform_matrix, dim=-1, index=right_right_indices, src=p3_coef, reduce="sum"
        )

        valid_mask = torch.ones((times.shape[0], self.control_num), device=curves.device)
        mask_index = self.current_control_num.squeeze() < self.control_num
        valid_mask[mask_index] = torch.scatter(
            input=valid_mask[mask_index],
            dim=-1,
            index=(self.current_control_num)[mask_index],
            src=torch.zeros_like(self.current_control_num).float()[mask_index],
        )
        accum_valid_mask = torch.cumprod(valid_mask, dim=-1)

        mask_curves = curves * accum_valid_mask[..., None]
        mask_transform_matrix = transform_matrix * accum_valid_mask[..., None] + dummy_transform_matrix * (
            1 - accum_valid_mask[..., None]
        )

        # # transform_matrix is not full rank, we need to replace 1 eq with dummy eq
        # valid_mask_2 = torch.ones((times.shape[0], self.control_num), device=curves.device)
        # valid_mask_2 = torch.scatter(input=valid_mask_2, dim=-1, index=(N_pts - 1), src=torch.zeros_like(self.current_control_num).float())
        # mask_transform_matrix_2 = mask_transform_matrix * valid_mask_2[...,None] + dummy_eq[None,None] * (1 - valid_mask_2[...,None])
        control_pts = torch.linalg.lstsq(mask_transform_matrix, mask_curves).solution
        # error = torch.square(control_pts - torch.linalg.pinv(mask_transform_matrix_2) @ curves).sum(-1).mean(-1)
        return control_pts

    def interpolate_cubic_hermite(self, signal, times, N):
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
        if len(signal.shape) == 3:  # remove extra singleton dimension
            interpolation = interpolation.squeeze(-1)

        return interpolation

    def create_from_pcd_dynamic(
        self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int, dyn_tracjectory: torch.tensor
    ):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        times = torch.tensor(np.asarray(pcd.times)).float().cuda()

        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 + 1)).float().cuda()
        )  # NOTE: +1 for blending factor
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        t_step = torch.linspace(0, 1, dyn_tracjectory.shape[1]).cuda().float()
        t_step = t_step[None, :, None].expand(dyn_tracjectory.shape[0], -1, -1)
        init_control_pts = inverse_cubic_hermite(dyn_tracjectory / self.deform_spatial_scale, t_step, N_pts=self.control_num)
        self.control_xyz = nn.Parameter(init_control_pts.requires_grad_(True))
        self.current_control_num = torch.tensor(self.control_num, device="cuda").repeat(fused_point_cloud.shape[0])[
            ..., None
        ]

        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        features9channel = torch.cat((fused_color, fused_color), dim=1)
        self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))
        N, _ = fused_color.shape

        fomega = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        self._features_t = nn.Parameter(fomega.contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))

        nn.init.constant_(self._omega, 0)

        self.split_depth = torch.zeros(fused_point_cloud.shape[0], 1, dtype=torch.long, device="cuda")
        self._track_split_depth = True

        self.rgb_grd = {}

        self.maxz, self.minz = torch.amax(self._xyz[:, 2]), torch.amin(self._xyz[:, 2])
        self.maxy, self.miny = torch.amax(self._xyz[:, 1]), torch.amin(self._xyz[:, 1])
        self.maxx, self.minx = torch.amax(self._xyz[:, 0]), torch.amin(self._xyz[:, 0])
        self.maxz = min((self.maxz, 200.0))  # some outliers in the n4d datasets..

        for name, W in self.rgbdecoder.named_parameters():
            if "weight" in name:
                self.rgb_grd[name] = torch.zeros_like(
                    W, requires_grad=False
                ).cuda()  # self.rgb_grd[name] + W.grad.clone()
            elif "bias" in name:
                print("not implemented")
                quit()

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        # mean_pc = torch.mean(fused_point_cloud, dim=1, keepdim=True)
        # std_pc = 0*torch.std(fused_point_cloud, dim=1, keepdim=True) + 10
        # fused_point_cloud = torch.normal(mean=mean_pc, std=std_pc.expand(fused_point_cloud.shape[0], 3)).cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # fused_color = RGB2SH(torch.randn(fused_point_cloud.shape[0], 3).cuda())

        times = torch.tensor(np.asarray(pcd.times)).float().cuda()

        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 + 1)).float().cuda()
        )  # NOTE: +1 for blending factor
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        mean_x, std_x = torch.mean(fused_point_cloud[..., 0], dim=0), torch.std(fused_point_cloud[..., 0], dim=0)
        mean_y, std_y = torch.mean(fused_point_cloud[..., 1], dim=0), torch.std(fused_point_cloud[..., 1], dim=0)
        mean_z, std_z = torch.mean(fused_point_cloud[..., 2], dim=0), torch.std(fused_point_cloud[..., 2], dim=0)
        std_xyz = torch.tensor([std_x, std_y, std_z], device="cuda")
        mean_xyz = torch.tensor([mean_x, mean_y, mean_z], device="cuda")
        self.control_xyz = nn.Parameter(
            (
                torch.randn(fused_color.shape[0], self.control_num, 3, device="cuda").requires_grad_(True)
                * std_xyz[None, None]
                + mean_xyz[None, None]
            )
        )
        self.current_control_num = torch.tensor(self.control_num, device="cuda").repeat(fused_color.shape[0], 1)
        # self.control_xyz = nn.Parameter((torch.randn(self.control_num, 3, device="cuda").requires_grad_(True) * std_xyz[None] + mean_xyz[None])[None].repeat(fused_color.shape[0], 1, 1))
        # self.grid = self.grid.to("cuda")

        # self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        features9channel = torch.cat((fused_color, fused_color), dim=1)
        self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))
        N, _ = fused_color.shape

        fomega = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        self._features_t = nn.Parameter(fomega.contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))

        nn.init.constant_(self._omega, 0)

        self.rgb_grd = {}

        self.maxz, self.minz = torch.amax(self._xyz[:, 2]), torch.amin(self._xyz[:, 2])
        self.maxy, self.miny = torch.amax(self._xyz[:, 1]), torch.amin(self._xyz[:, 1])
        self.maxx, self.minx = torch.amax(self._xyz[:, 0]), torch.amin(self._xyz[:, 0])
        self.maxz = min((self.maxz, 200.0))  # some outliers in the n4d datasets..

        self.split_depth = torch.zeros(fused_point_cloud.shape[0], 1, dtype=torch.long, device="cuda")
        self._track_split_depth = True

        for name, W in self.rgbdecoder.named_parameters():
            if "weight" in name:
                self.rgb_grd[name] = torch.zeros_like(
                    W, requires_grad=False
                ).cuda()  # self.rgb_grd[name] + W.grad.clone()
            elif "bias" in name:
                print("not implemented")
                quit()

    def init_tass_state(self, n_gaussians):
        """Initialize TASS tracking buffers for n_gaussians."""
        device = self._xyz.device
        self.tass_error_buffer = torch.zeros(n_gaussians, 3, device=device)
        self.tass_error_count = torch.zeros(n_gaussians, dtype=torch.int, device=device)
        self.tass_split_flags = torch.zeros(n_gaussians, dtype=torch.bool, device=device)
        self.mg_mas_overlap = torch.zeros(n_gaussians, 1, device=device)

    def mg_mas_attenuate_gradients(self, control_point_grads, mask_overlap_ratio):
        """
        MG-MAS: attenuate control point gradients based on tool mask overlap (Eq. 4).
        
        control_point_grads: [N, K, 3] - gradients w.r.t. control points
        mask_overlap_ratio: [N, 1] - per-Gaussian occlusion ratio (0=no occlusion, 1=fully occluded)
        
        Returns: attenuated gradients
        """
        gamma = getattr(self, 'tass_gamma', 2.0)
        attenuation = torch.exp(-gamma * mask_overlap_ratio).clamp(min=1e-6)
        return control_point_grads * attenuation.unsqueeze(-1)

    def tass_detect_and_split(self, photometric_error, mask_overlap):
        """
        TASS: detect topological ruptures and split spline trajectories (Eq. 5, 6).
        
        photometric_error: [N] per-Gaussian photometric error at current frame
        mask_overlap: [N] per-Gaussian mask overlap ratio
        
        Returns: list of (parent_idx, child_data) for each split event
        """
        device = self._xyz.device
        n_gaussians = self._xyz.shape[0]
        
        if self.tass_error_buffer.shape[0] != n_gaussians:
            self.init_tass_state(n_gaussians)
        
        epsilon = getattr(self, 'tass_epsilon', 0.05)
        
        # Shift buffer: error[t-2] = error[t-1], error[t-1] = error[t], error[t] = current
        self.tass_error_buffer = torch.roll(self.tass_error_buffer, -1, dims=0)
        self.tass_error_buffer[-1, :] = 0
        self.tass_error_buffer[:, -1] = photometric_error
        
        is_high_error = photometric_error > epsilon
        self.tass_error_count = torch.where(is_high_error, self.tass_error_count + 1, torch.zeros_like(self.tass_error_count))
        
        # Trigger split if: error > epsilon for 3 consecutive frames
        split_mask = (self.tass_error_count >= 3) & ~self.tass_split_flags
        
        split_events = []
        if split_mask.any():
            split_indices = torch.where(split_mask)[0]
            for idx in split_indices:
                split_events.append(self._tass_split_gaussian(idx.item()))
            self.tass_split_flags[split_mask] = True
        
        return split_events

    def _tass_split_gaussian(self, g_idx):
        """
        Split a single Gaussian's trajectory at detected cut point (Eq. 6).
        Duplicates the Gaussian into two separate primitives.
        """
        n, k, _ = self.control_xyz.shape
        
        parent_xyz = self._xyz[g_idx:g_idx+1].detach()
        parent_control = self.control_xyz[g_idx:g_idx+1].detach()
        parent_ccn = self.current_control_num[g_idx:g_idx+1].detach()
        parent_f_dc = self._features_dc[g_idx:g_idx+1].detach()
        parent_f_rest = self._features_rest[g_idx:g_idx+1].detach()
        parent_f_t = self._features_t[g_idx:g_idx+1].detach()
        parent_opacity = self._opacity[g_idx:g_idx+1].detach()
        parent_scaling = self._scaling[g_idx:g_idx+1].detach()
        parent_rotation = self._rotation[g_idx:g_idx+1].detach()
        parent_omega = self._omega[g_idx:g_idx+1].detach()
        parent_split_depth = self.split_depth[g_idx:g_idx+1].detach() if self._track_split_depth else None
        
        # Create child: same position/scale/rotation, new trajectory diverges
        child_control = parent_control.clone() + torch.randn_like(parent_control) * 1e-3
        
        # Both start with identical control points; downstream optimization will diverge them
        self.densification_postfix(
            parent_xyz,
            child_control,
            parent_ccn,
            parent_f_dc,
            parent_f_rest,
            parent_f_t,
            parent_opacity,
            parent_scaling,
            parent_rotation,
            parent_omega,
            new_split_depth=(parent_split_depth + 1) if parent_split_depth is not None else None,
        )
        
        return g_idx

    def get_params(self):
        gs_params = [
            self._xyz,
            self.control_xyz,
            self.current_control_num,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
        ]
        return (
            gs_params
            + list(self._deformation.get_mlp_parameters())
            + list(self._deformation.get_grid_parameters())
            + list(self._posenet.get_mlp_parameters())
        )

    def training_setup(self, training_args, stage="warm"):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.preprocesspoints
        self.rgbdecoder.cuda()

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {
                "params": [self.control_xyz],
                "lr": 10 * training_args.position_lr_init * self.spatial_lr_scale,
                "name": "control_xyz",
            },
            {"params": [self.current_control_num], "lr": 0.0, "name": "current_control_num"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._features_t], "lr": training_args.featuret_lr, "name": "f_t"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self._omega], "lr": training_args.omega_lr, "name": "omega"},
            {"params": list(self.rgbdecoder.parameters()), "lr": training_args.rgb_lr, "name": "decoder"},
        ]

        # Pose is run during warm up, we want a lower starting LR for fine
        if stage != "warm":
            if self._posenet is not None:
                l.append(
                    {
                        "params": list(self._posenet.get_mlp_parameters()),
                        "lr": training_args.pose_lr_init / 10,
                        "name": "posenet",
                    }
                )
                self.pose_scheduler_args = get_expon_lr_func(
                    lr_init=training_args.pose_lr_init / 10,
                    lr_final=training_args.pose_lr_final / 10,
                    # lr_delay_mult=training_args.pose_lr_delay_mult,
                    max_steps=training_args.position_lr_max_steps,
                )
        else:
            if self._posenet is not None:
                # l.append({'params': list(self._posenet.get_all_parameters()),
                #         'lr': training_args.pose_lr_init, "name": "posenet"})
                l.append({"params": list(self._posenet.get_focal_parameters()), "lr": 0.005, "name": "focal"})
                l.append(
                    {
                        "params": list(self._posenet.get_mlp_parameters()),
                        "lr": training_args.pose_lr_init,
                        "name": "posenet",
                    }
                )
                l.append(
                    {
                        "params": list(self._posenet.get_scale_parameters()),
                        "lr": training_args.pose_lr_init,
                        "name": "posenet_cvdscale",
                    }
                )
                self.pose_scheduler_args = get_expon_lr_func(
                    lr_init=training_args.pose_lr_init,
                    lr_final=training_args.pose_lr_final,
                    # lr_delay_mult=training_args.pose_lr_delay_mult,
                    max_steps=training_args.position_lr_max_steps,
                )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            # lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.deformation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deformation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.deformation_lr_final * self.spatial_lr_scale,
            # lr_delay_mult=training_args.deformation_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.grid_scheduler_args = get_expon_lr_func(
            lr_init=training_args.grid_lr_init * self.spatial_lr_scale,
            lr_final=training_args.grid_lr_final * self.spatial_lr_scale,
            # lr_delay_mult=training_args.deformation_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                # return lr
            elif "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group["lr"] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group["lr"] = lr
                # return lr
            elif param_group["name"] == "posenet":
                lr = self.pose_scheduler_args(iteration)
                param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"] 
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        for i in range(self._features_t.shape[1]):
            l.append("f_t_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        for i in range(self._omega.shape[1]):
            l.append("omega_{}".format(i))
        for i in range(self.control_xyz.shape[1]):
            for j in range(self.control_xyz.shape[2]):
                if j == 0:
                    l.append("control_x_{}".format(i))
                elif j == 1:
                    l.append("control_y_{}".format(i))
                elif j == 2:
                    l.append("control_z_{}".format(i))
        l.append("current_control_num")
        return l

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path, "posenet.pth"), map_location="cuda")
        if self._posenet is not None:
            self._posenet.load_state_dict(weight_dict)
            self._posenet = self._posenet.to("cuda")


    def save_deformation(self, path):
        torch.save(self._posenet.state_dict(), os.path.join(path, "posenet.pth"))

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        control_xyz = self.control_xyz.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        current_control_num = self.current_control_num.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy()
        # f_rest = self._features_rest.detach().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        f_t = self._features_t.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        omega = self._omega.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                f_t,
                opacities,
                scale,
                rotation,
                omega,
                control_xyz,
                current_control_num,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pt")
        print(f"Saving model checkpoint to: {model_fname}")
        torch.save(self.rgbdecoder.state_dict(), model_fname)

    def save_ply_compact(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes_compact()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def save_ply_compact_dy(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        # control_xyz = self.control_xyz.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        current_control_num = self.current_control_num.detach().cpu().numpy()

        f_dc = self._features_dc.detach().cpu().numpy()

        f_t = self._features_t.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        omega = self._omega.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes_compact_dy()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((f_dc, f_t, opacities, scale, rotation, omega, current_control_num), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pt")
        print(f"Saving model checkpoint to: {model_fname}")
        torch.save(self.rgbdecoder.state_dict(), model_fname)

        control_fname = path.replace(".ply", ".npy")
        print(f"Saving control points to: {control_fname}")
        np.savez_compressed(
            control_fname.replace("npy", "npz"), flat_control_xyz=self.flat_control_xyz.detach().cpu().numpy()
        )

    def construct_list_of_attributes_compact(self):
        l = ["x", "y", "z"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def construct_list_of_attributes_compact_dy(self):
        l = []
        for i in range(self._features_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_t.shape[1]):
            l.append("f_t_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        for i in range(self._omega.shape[1]):
            l.append("omega_{}".format(i))
        l.append("current_control_num")
        return l

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        if torch.isnan(opacities_new).any():
            print("opacities_new is nan,end training, ending program now.")
            exit()
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        ckpt = torch.load(path.replace(".ply", ".pt"))
        self.rgbdecoder.cuda()
        self.rgbdecoder.load_state_dict(ckpt)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        control_xyz_list = []
        for i in range(self.control_num):
            control_xyz_list.append(
                np.stack(
                    (
                        np.asarray(plydata.elements[0][f"control_x_{i}"]),
                        np.asarray(plydata.elements[0][f"control_y_{i}"]),
                        np.asarray(plydata.elements[0][f"control_z_{i}"]),
                    ),
                    axis=1,
                )
            )
        control_xyz = np.stack(control_xyz_list, axis=1)
        current_control_num = np.asarray(plydata.elements[0]["current_control_num"])[..., np.newaxis]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        fdc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        features_dc = np.zeros((xyz.shape[0], len(fdc_names)))
        for idx, attr_name in enumerate(fdc_names):
            features_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2))

        ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega_")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.control_xyz = nn.Parameter(
            torch.tensor(control_xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.current_control_num = nn.Parameter(
            torch.tensor(current_control_num, dtype=torch.int64, device="cuda"), requires_grad=False
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").squeeze().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_t = nn.Parameter(torch.tensor(ftomegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        n_pts = xyz.shape[0]
        self.split_depth = torch.zeros(n_pts, 1, dtype=torch.long, device="cuda")
        self._track_split_depth = True
        
    def load_ply_compact(self, path):
        # ! only for evaluation now
        plydata = PlyData.read(path)
        self.rgbdecoder.cuda()      

        if 'compact_point_cloud_static' in path:
            xyz = np.stack(
                (
                    np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"]),
                ),
                axis=1,
            )
            
            current_control_num = np.zeros((xyz.shape[0], 1), dtype=np.float32)

            
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

            fdc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
            features_dc = np.zeros((xyz.shape[0], len(fdc_names)))
            for idx, attr_name in enumerate(fdc_names):
                features_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            ftomegas = np.zeros((xyz.shape[0], 3))
            
            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

            rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
            omegas = np.zeros((xyz.shape[0], 3))
        else:
            ckpt = torch.load(path.replace(".ply", ".pt"))
            flattened_control_xyz = np.load(path.replace(".ply", ".npz"))["flat_control_xyz"]
            self.rgbdecoder.load_state_dict(ckpt)
            self.flat_control_xyz = torch.tensor(flattened_control_xyz, dtype=torch.float, device="cuda")
            # self.flat_control_xyz = torch.cat(flat_control_point, dim=0).contiguous()

            current_control_num = np.asarray(plydata.elements[0]["current_control_num"])[..., np.newaxis]
            xyz = np.zeros((current_control_num.shape[0], 3), dtype=np.float32)
            
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

            fdc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
            features_dc = np.zeros((xyz.shape[0], len(fdc_names)))
            for idx, attr_name in enumerate(fdc_names):
                features_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])

            ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
            ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
            for idx, attr_name in enumerate(ft_names):
                ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

            rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

            omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega_")]
            omegas = np.zeros((xyz.shape[0], len(omega_names)))
            for idx, attr_name in enumerate(omega_names):
                omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        # self.control_xyz = nn.Parameter(
        #     torch.tensor(control_xyz, dtype=torch.float, device="cuda")
        # )
        self.current_control_num = nn.Parameter(
            torch.tensor(current_control_num, dtype=torch.float, device="cuda"), requires_grad=False
        )
        
        self.index_offset = (
            torch.cat(
                [torch.zeros(1).cuda(), torch.cumsum(self.current_control_num.squeeze()[:-1], dim=0)], dim=0
            ).long()
        )[:, None]
            
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").squeeze()
        )

        self._features_t = nn.Parameter(torch.tensor(ftomegas, dtype=torch.float, device="cuda"))

        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda"))

        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda"))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or group["name"] == "focal":
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "current_control_num":
                    group["params"][0] = nn.Parameter(group["params"][0][mask], requires_grad=False)
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self.control_xyz = optimizable_tensors["control_xyz"]
        self.current_control_num = optimizable_tensors["current_control_num"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_t = optimizable_tensors["f_t"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._omega = optimizable_tensors["omega"]

        if self._track_split_depth and self.split_depth.numel() > 0:
            self.split_depth = self.split_depth[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or group["name"] == "focal":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "current_control_num":
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0), requires_grad=False
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_control_xyz,
        new_current_control_num,
        new_features_dc,
        new_features_rest,
        new_features_t,
        new_opacities,
        new_scaling,
        new_rotation,
        new_omega,
        new_split_depth=None,
    ):
        d = {
            "xyz": new_xyz,
            "control_xyz": new_control_xyz,
            "current_control_num": new_current_control_num,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "f_t": new_features_t,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "omega": new_omega,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.control_xyz = optimizable_tensors["control_xyz"]
        self.current_control_num = optimizable_tensors["current_control_num"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_t = optimizable_tensors["f_t"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._deformation = optimizable_tensors["deformation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._omega = optimizable_tensors["omega"]

        if new_split_depth is not None:
            self.split_depth = torch.cat([self.split_depth, new_split_depth], dim=0)

    def densify_and_splitv2(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        new_control_xyz = self.control_xyz[selected_pts_mask].repeat(N, 1, 1)
        new_current_control_num = self.current_control_num[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)  # n,1,1 to n1
        new_feature_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_feature_t = self._features_t[selected_pts_mask].repeat(N, 1)

        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_omega = self._omega[selected_pts_mask].repeat(N, 1)

        new_split_depth = (self.split_depth[selected_pts_mask] + 1).repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_control_xyz,
            new_current_control_num,
            new_features_dc,
            new_feature_rest,
            new_feature_t,
            new_opacity,
            new_scaling,
            new_rotation,
            new_omega,
            new_split_depth=new_split_depth,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def densify_pruneclone(self, max_grad, min_opacity, extent, max_screen_size, splitN=2):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        print("befre clone", self._xyz.shape[0])
        self.densify_and_clone(grads, max_grad, extent)
        print("after clone", self._xyz.shape[0])

        self.densify_and_splitv2(grads, max_grad, extent, splitN)
        print("after split", self._xyz.shape[0])

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        torch.cuda.empty_cache()

    def densify_and_clone(
        self,
        grads,
        grad_threshold,
        scene_extent,
        density_threshold=20,
        displacement_scale=20,
        model_path=None,
        iteration=None,
        stage=None,
    ):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(
            grads_accum_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
        new_xyz = self._xyz[selected_pts_mask]
        new_control_xyz = self.control_xyz[selected_pts_mask]
        new_current_control_num = self.current_control_num[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_features_t = self._features_t[selected_pts_mask]

        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_omega = self._omega[selected_pts_mask]

        new_split_depth = self.split_depth[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_control_xyz,
            new_current_control_num,
            new_features_dc,
            new_features_rest,
            new_features_t,
            new_opacities,
            new_scaling,
            new_rotation,
            new_omega,
            new_split_depth=new_split_depth,
        )
