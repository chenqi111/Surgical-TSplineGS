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
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.deform_spatial_scale = 1e-2
        self.rgbfuntion = "sandwich"
        self.control_num = 12
        self.prune_error_threshold = 1.0
        
        self._source_path = ""
        self.dataset_type = "nvidia"
        self.depth_type = "depth" # depth or disp
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.render_process = True
        self.debug_process = True
        self.add_points = False
        self.extension = ".png"
        self.llffhold = 8
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.timebase_pe = 10
        self.timenet_width = 256 
        self.timenet_output = 6
        self.pixel_base_pe = 5

        # Surgical-TSplineGS hyperparameters
        self.tass_epsilon = 0.05
        self.tass_gamma = 2.0
        self.mgmas_enabled = True
        self.tass_enabled = True
        self.lambda_depth_consistency = 0.5
        self.lambda_traj_smoothness = 0.1
        self.lambda_masked_rgb = 1.0
        super().__init__(parser, "ModelHiddenParams")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader = False
        self.zerostamp_init = False
        self.custom_sampler = None
        self.iterations = 30_000
        self.coarse_iterations = 1000
        self.static_iterations = 1000

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000

        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01

        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.pose_lr_init = 0.0005
        self.pose_lr_final = 0.00005
        self.pose_lr_delay_mult = 0.01

        self.feature_lr = 0.0025
        self.featuret_lr = 0.001
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.p_lambda_dssim = 0.0
        self.lambda_lpips = 0
        self.weight_constraint_init = 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5_000
        self.opacity_reset_interval = 3_000
        self.densification_interval = 1_00
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.fine_batch_size = 1
        self.coarse_batch_size = 1
        self.add_point = False
        self.use_instance_mask = False

        self.prevpath = "1"
        self.opthr = 0.005
        self.desicnt = 6
        self.densify = 1
        self.densify_grad_threshold = 0.0008
        self.densify_grad_threshold_dynamic = 0.00008
        self.preprocesspoints = 0
        self.addsphpointsscale = 0.8
        self.raystart = 0.7

        self.soft_depth_start = 1000
        self.hard_depth_start = 0
        self.error_tolerance = 0.001

        self.trbfc_lr = 0.0001  #
        self.trbfs_lr = 0.03
        self.trbfslinit = 0.0  #
        self.omega_lr = 0.0001
        self.zeta_lr = 0.0001
        self.movelr = 3.5
        self.rgb_lr = 0.0001

        self.stat_npts = 20000
        self.dyn_npts = 20000

        self.w_depth = 1.0
        self.w_mask = 2.0
        self.w_track = 1.0
        self.w_normal = 0
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
