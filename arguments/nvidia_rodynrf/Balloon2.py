_base_ = "./default.py"

ModelParams = dict(
    depth_type="disp",
)

OptimizationParams = dict(
    densify_grad_threshold_dynamic = 0.0002,
    densify_grad_threshold = 0.0008,
) 