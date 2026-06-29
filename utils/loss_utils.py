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

from math import exp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def trbfunction(x):
    return torch.exp(-1 * x.pow(2))


def compute_tv_loss(pred):
    """
    Args:
        pred: [batch, H, W, 3]

    Returns:
        tv_loss: [batch]
    """
    h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
    w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
    return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))


## som losses
def masked_mse_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_mse_loss(pred, gt, quantile)
    else:
        sum_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (ndim * torch.sum(mask[quantile_mask]) + 1e-8)
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (ndim * torch.sum(mask[quantile_mask]) + 1e-8)
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


def masked_huber_loss(pred, gt, delta, mask=None, normalize=True):
    if mask is None:
        return F.huber_loss(pred, gt, delta=delta)
    else:
        sum_loss = F.huber_loss(pred, gt, delta=delta, reduction="none")
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum(sum_loss * mask) / (ndim * torch.sum(mask) + 1e-8)
        else:
            return torch.mean(sum_loss * mask)


def trimmed_mse_loss(pred, gt, quantile=0.9):
    loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def compute_gradient_loss(pred, gt, mask, quantile=0.98):
    """
    Compute gradient loss
    pred: (batch_size, H, W, D) or (batch_size, H, W)
    gt: (batch_size, H, W, D) or (batch_size, H, W)
    mask: (batch_size, H, W), bool or float
    """
    # NOTE: messy need to be cleaned up
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
    pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
    gt_grad_x = gt[:, :, 1:] - gt[:, :, :-1]
    gt_grad_y = gt[:, 1:, :] - gt[:, :-1, :]
    loss = masked_l1_loss(
        pred_grad_x[mask_x][..., None], gt_grad_x[mask_x][..., None], quantile=quantile
    ) + masked_l1_loss(pred_grad_y[mask_y][..., None], gt_grad_y[mask_y][..., None], quantile=quantile)
    return loss


def get_weights_for_procrustes(clusters, visibilities=None):
    clusters_median = clusters.median(dim=-2, keepdim=True)[0]
    dists2clusters_center = torch.norm(clusters - clusters_median, dim=-1)
    dists2clusters_center /= dists2clusters_center.median(dim=-1, keepdim=True)[0]
    weights = torch.exp(-dists2clusters_center)
    weights /= weights.mean(dim=-1, keepdim=True) + 1e-6
    if visibilities is not None:
        weights *= visibilities.float() + 1e-6
    invalid = dists2clusters_center > np.quantile(dists2clusters_center.cpu().numpy(), 0.9)
    invalid |= torch.isnan(weights)
    weights[invalid] = 0
    return weights


def compute_z_acc_loss(means_ts_nb: torch.Tensor, w2cs: torch.Tensor):
    """
    :param means_ts (G, 3, B, 3)
    :param w2cs (B, 4, 4)
    return (float)
    """
    camera_center_t = torch.linalg.inv(w2cs)[:, :3, 3]  # (B, 3)
    ray_dir = F.normalize(means_ts_nb[:, 1] - camera_center_t, p=2.0, dim=-1)  # [G, B, 3]
    # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, B, 3]
    # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    acc_loss = (((means_ts_nb[:, 1] - means_ts_nb[:, 0]) * ray_dir).sum(dim=-1) ** 2).mean() + (
        ((means_ts_nb[:, 2] - means_ts_nb[:, 1]) * ray_dir).sum(dim=-1) ** 2
    ).mean()
    return acc_loss


def compute_se3_smoothness_loss(
    rots: torch.Tensor,
    transls: torch.Tensor,
    weight_rot: float = 1.0,
    weight_transl: float = 2.0,
):
    """
    central differences
    :param motion_transls (K, T, 3)
    :param motion_rots (K, T, 6)
    """
    r_accel_loss = compute_accel_loss(rots)
    t_accel_loss = compute_accel_loss(transls)
    return r_accel_loss * weight_rot + t_accel_loss * weight_transl


def compute_accel_loss(transls):
    accel = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    loss = accel.norm(dim=-1).mean()
    return loss


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1, img2)
    return loss.mean()


def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        channel = gt.shape[1]
        mask = mask.expand(-1, channel, -1, -1)
        return torch.abs((network_output - gt) * mask).sum() / (mask.sum() + 1e-8)
    else:
        return torch.abs((network_output - gt)).mean()


def depth_consistency_loss(rendered_depth, depth_prior, confidence=None):
    """
    Scale-and-offset invariant depth loss (Eq. 10).
    Aligns rendered depth to monocular depth prior via least-squares scale/shift.
    
    rendered_depth: [B, 1, H, W] - rendered depth map
    depth_prior: [B, 1, H, W] - monocular depth prior
    confidence: [B, 1, H, W] or None - confidence weights (inverse of local gradient)
    """
    B, _, H, W = rendered_depth.shape
    d_rend = rendered_depth.view(B, -1)       # B, N
    d_prior = depth_prior.view(B, -1)         # B, N
    
    if confidence is not None:
        w = confidence.view(B, -1).clamp(min=1e-6)
    else:
        w = torch.ones_like(d_rend)
    
    # Weighted least-squares: solve for s (scale) and o (offset)
    sum_w = w.sum(dim=-1, keepdim=True)       # B, 1
    sum_wd = (w * d_rend).sum(dim=-1, keepdim=True)
    sum_wp = (w * d_prior).sum(dim=-1, keepdim=True)
    sum_wdd = (w * d_rend * d_rend).sum(dim=-1, keepdim=True)
    sum_wdp = (w * d_rend * d_prior).sum(dim=-1, keepdim=True)
    sum_wpp = (w * d_prior * d_prior).sum(dim=-1, keepdim=True)
    
    # s = (sum_w * sum_wdp - sum_wd * sum_wp) / (sum_w * sum_wdd - sum_wd * sum_wd)
    denom_s = sum_w * sum_wdd - sum_wd * sum_wd
    s = (sum_w * sum_wdp - sum_wd * sum_wp) / (denom_s + 1e-8)
    o = (sum_wp - s * sum_wd) / (sum_w + 1e-8)
    
    aligned_prior = s * d_rend + o
    loss = (w * (aligned_prior - d_prior) ** 2).sum(dim=-1) / (sum_w.squeeze(-1) + 1e-8)
    return loss.mean()


def trajectory_smoothness_loss(gaussians, time_grid=10):
    """
    Penalty on second derivative (acceleration) of spline trajectories (Eq. 11).
    
    gaussians: GaussianModel instance with control_xyz and current_control_num
    time_grid: number of samples along the trajectory for numerical integration
    """
    control_xyz = gaussians.control_xyz  # [N, K, 3]
    ccn = gaussians.current_control_num.squeeze().long()  # [N]
    N = control_xyz.shape[0]
    K = control_xyz.shape[1]
    
    total_loss = 0.0
    for g_idx in range(N):
        k = ccn[g_idx]
        if k < 4:
            continue
        pts = control_xyz[g_idx, :k]  # [k, 3]
        # Sample trajectory at regular intervals
        t = torch.linspace(0, 1, time_grid, device=pts.device)
        dt = 1.0 / (time_grid - 1)
        
        # Evaluate spline at sample points
        sampled = torch.zeros(time_grid, 3, device=pts.device)
        for i in range(time_grid):
            # Find segment for this t
            seg_t = t[i] * (k - 1)
            seg_idx = int(seg_t)
            seg_idx = min(max(seg_idx, 0), k - 2)
            local_t = seg_t - seg_idx
            
            idx0 = max(seg_idx - 1, 0)
            idx1 = seg_idx
            idx2 = min(seg_idx + 1, k - 1)
            idx3 = min(seg_idx + 2, k - 1)
            
            p0, p1, p2, p3 = pts[idx0], pts[idx1], pts[idx2], pts[idx3]
            
            # One-sided derivatives
            m0 = (p2 - p0) / 2 if idx0 != idx1 else (p2 - p1)
            m1 = (p3 - p1) / 2 if idx3 != idx2 else (p2 - p1)
            
            h00 = (1 + 2 * local_t) * (1 - local_t) ** 2
            h10 = local_t * (1 - local_t) ** 2
            h01 = local_t ** 2 * (3 - 2 * local_t)
            h11 = local_t ** 2 * (local_t - 1)
            
            sampled[i] = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
        
        # Central difference for acceleration
        accel = sampled[2:] - 2 * sampled[1:-1] + sampled[:-2]
        total_loss += torch.mean(accel ** 2)
    
    return total_loss / max(N, 1)


def l2_loss(network_output, gt, mask=None):
    if mask is not None:
        channel = gt.shape[1]
        mask = mask.expand(-1, channel, -1, -1)
        return torch.square((network_output - gt) * mask).sum() / (mask.sum() + 1e-8)
    else:
        return torch.square((network_output - gt)).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM_tensor(nn.Module):
    """Layer to compute the SSIM loss between a pair of images, returns non-reduced tensor error"""

    def __init__(self):
        super(SSIM_tensor, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class BinaryDiceLoss(nn.Module):
    def __init__(
        self,
        batch_dice: bool = False,
        from_logits: bool = True,
        log_loss: bool = False,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Implementation of Dice loss for binary image segmentation tasks

        Args:
            batch_dice: dice per sample and average or treat batch as a single volumetric sample (default)
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(BinaryDiceLoss, self).__init__()
        self.batch_dice = batch_dice
        self.from_logits = from_logits
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)

        y_true = y_true.view(bs, -1)  # bs x num_elems
        y_pred = y_pred.view(bs, -1)  # bs x num_elems

        if self.batch_dice == True:
            intersection = torch.sum(y_pred * y_true)  # float
            cardinality = torch.sum(y_pred + y_true)  # float
        else:
            intersection = torch.sum(y_pred * y_true, dim=-1)  # bs x float
            cardinality = torch.sum(y_pred + y_true, dim=-1)  # bs x float

        dice_scores = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        if self.log_loss:
            losses = -torch.log(dice_scores.clamp_min(self.eps))
        else:
            losses = 1.0 - dice_scores
        return losses.mean()


def sgt_smoothness(dyn_pc, time, fwd_time, bwd_time):
    pointtimes = (
        torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    )

    # Calculate Polynomial trajectory
    basicfunction = trbfunction
    trbfcenter = dyn_pc.get_trbfcenter
    trbfscale = dyn_pc.get_trbfscale
    trbfdistanceoffset = time * pointtimes - trbfcenter
    trbfdistance = trbfdistanceoffset / torch.exp(trbfscale)
    basicfunction(trbfdistance)

    trbfdistanceoffset_prev = bwd_time * pointtimes - trbfcenter
    trbfdistance_prev = trbfdistanceoffset_prev / torch.exp(trbfscale)
    basicfunction(trbfdistance_prev)

    trbfdistanceoffset_next = fwd_time * pointtimes - trbfcenter
    trbfdistance_next = trbfdistanceoffset_next / torch.exp(trbfscale)
    basicfunction(trbfdistance_next)

    return 0


# class KnnConstraint(nn.Module):
#     """
#     The Normal Consistency Constraint
#     """
#     def __init__(self, neighborhood_size=4, relax_size=16):
#         super().__init__()
#         self.neighborhood_size = neighborhood_size
#         self.relax_size = relax_size

#     def forward(self, xyz, rand_xyz):
#         idx = pointops.knn(xyz, xyz, self.neighborhood_size)[0]
#         with torch.no_grad():
#             max_rand_dist = pointops.knn(rand_xyz, rand_xyz, self.neighborhood_size + self.relax_size)[1][..., -1:]
#             rand_neighborhood = pointops.index_points(rand_xyz, idx)

#         rand_neighborhood_dist = (rand_xyz[...,None,:] - rand_neighborhood).norm(dim=-1)

#         rand_dist_diff = torch.clamp((rand_neighborhood_dist - max_rand_dist), min=0)
#         return rand_dist_diff

# class KnnConstraint(nn.Module):
#     """
#     The Normal Consistency Constraint
#     """
#     def __init__(self, neighborhood_size=4, relax_size=16):
#         super().__init__()
#         self.neighborhood_size = neighborhood_size
#         self.relax_size = relax_size

#     def forward(self, xyz, rand_xyz):
#         with torch.no_grad():
#             idx, dist = pointops.knn(xyz, xyz, self.neighborhood_size)
#             rand_neighborhood = pointops.index_points(rand_xyz, idx)

#         rand_neighborhood_dist = (rand_xyz[...,None,:] - rand_neighborhood).norm(dim=-1)
#         rand_dist_diff = torch.square(dist - rand_neighborhood_dist)
#         return rand_dist_diff


# class KnnConstraint(nn.Module):
#     """
#     The Normal Consistency Constraint
#     """

#     def __init__(self, neighborhood_size=20):
#         super().__init__()
#         self.neighborhood_size = neighborhood_size
#         self.temperature = 0.1

#     def forward(self, xyz, canno_xyz, radius):
#         batch_size, nsample, _ = xyz.shape
#         neighbor_inds = ball_query(xyz, xyz, K=self.neighborhood_size, radius=radius)[1][
#             ..., 1:
#         ]  # remove first element
#         neighbor_inds_mask = neighbor_inds != -1

#         neighbor_inds[~neighbor_inds_mask] = 0
#         neighbor_inds = neighbor_inds.reshape(batch_size, -1).long()

#         # get the neighborhood points
#         neighborhood = torch.gather(xyz, 1, neighbor_inds[:, :, None].expand(-1, -1, 3)).reshape(
#             batch_size, nsample, self.neighborhood_size - 1, 3
#         )  # B, N, K, 3
#         current_dist = (xyz[..., None, :] - neighborhood).norm(dim=-1)  # B, N, K

#         # get cannocal neighborhood points
#         canno_neighborhood = torch.gather(
#             canno_xyz[None].expand(batch_size, -1, -1), 1, neighbor_inds[:, :, None].expand(-1, -1, 3)
#         ).reshape(batch_size, nsample, self.neighborhood_size - 1, 3)
#         canno_dist = (canno_xyz[..., None, :] - canno_neighborhood).norm(dim=-1).detach()  # B, N, K

#         weight = torch.exp(-torch.square(canno_dist) * self.temperature).detach()
#         weight[~neighbor_inds_mask] = 0

#         return weighted_l2_loss_v1(current_dist, canno_dist, weight)
