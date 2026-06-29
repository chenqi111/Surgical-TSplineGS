# Surgical-TSplineGS default config
# Based on paper: Sec IV-A Implementation Details

# Dataset
dataset_type = "surgical"
depth_type = "depth"

# Model - surgical uses K=6 control points (paper Sec IV-A)
control_num = 6
deform_spatial_scale = 1e-2
prune_error_threshold = 0.5
sh_degree = 3

# Pose network (same as base Surgical-TSplineGS)
timebase_pe = 10
timenet_width = 256
timenet_output = 6
pixel_base_pe = 5

# Training
iterations = 30000
coarse_iterations = 2000               # warm-up: 2k iterations
static_iterations = 2000
position_lr_init = 0.00016
position_lr_final = 0.000016
position_lr_max_steps = 30000
pose_lr_init = 0.0005
pose_lr_final = 0.00005
feature_lr = 0.0025
opacity_lr = 0.05
scaling_lr = 0.005
rotation_lr = 0.001
omega_lr = 0.0001
rgb_lr = 0.0001

# Loss weights (paper Eq. 7)
lambda_dssim = 0.2                    # λ_ssim
w_depth = 0.5                         # part of λ_depth
w_mask = 1.0
w_track = 1.0
w_normal = 0

# Surgical-TSplineGS hyperparameters
tass_epsilon = 0.05                   # ε_top: photometric error threshold
tass_gamma = 2.0                      # γ: MG-MAS gradient scaling
mgmas_enabled = True
tass_enabled = True
lambda_depth_consistency = 0.5        # λ_depth (Eq. 10)
lambda_traj_smoothness = 0.1          # λ_smooth (Eq. 11)
lambda_masked_rgb = 1.0               # λ_rgb (Eq. 8)

# Densification
densify_from_iter = 500
densify_until_iter = 15000
densification_interval = 100
densify_grad_threshold = 0.0008
densify_grad_threshold_dynamic = 0.00008
opacity_reset_interval = 3000
densify = 1
desicnt = 6
opthr = 0.005

# Initial points
stat_npts = 30000
dyn_npts = 30000
