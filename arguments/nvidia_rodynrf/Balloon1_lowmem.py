_base_ = "./default.py"

ModelParams = dict(
    depth_type="disp",
)

OptimizationParams = dict(
    densify_grad_threshold_dynamic = 0.0002,
    densify_grad_threshold = 0.0008,
    use_instance_mask=True,
    coarse_batch_size=1,
    fine_batch_size=1,
    stat_npts=3000,
    dyn_npts=1000,
)
