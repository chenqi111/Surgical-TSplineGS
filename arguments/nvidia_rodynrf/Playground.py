_base_ = "./default.py"

OptimizationParams = dict(
    densify_grad_threshold = 0.0002,
    densify_grad_threshold_dynamic = 0.0002,
    opacity_reset_interval=30_000,
    use_instance_mask=True,
) 
