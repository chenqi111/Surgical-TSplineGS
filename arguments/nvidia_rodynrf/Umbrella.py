_base_ = "./default.py"

OptimizationParams = dict(
    w_normal=1.0,
    opacity_reset_interval=30_000,
    densify_grad_threshold = 0.0002,
    densify_grad_threshold_dynamic = 0.00002,
)
