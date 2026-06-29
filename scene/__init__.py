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

from arguments import ModelParams
from scene.dataset import FourDGSdataset
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration


class Scene:
    dyn_gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        static_gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        load_coarse=False,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.dyn_gaussians = gaussians
        self.stat_gaussians = static_gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        assert args.dataset_type in sceneLoadTypeCallbacks.keys(), "Could not recognize scene type!"

        dataset_type = args.dataset_type
        scene_info = sceneLoadTypeCallbacks[dataset_type](args)
        
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f"Original scene extent {self.cameras_extent}")
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)

        if self.loaded_iter:
            self.dyn_gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            )
            self.dyn_gaussians.load_model(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                )
            )
            self.stat_gaussians.load_ply(
                os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud_static.ply"
                )
            )
        else:
            self.dyn_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
            self.stat_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.dyn_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.dyn_gaussians.save_deformation(point_cloud_path)

        self.stat_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_static.ply"))

    def save_best_psnr(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_best")

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/fine_best")
        self.dyn_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.dyn_gaussians.save_deformation(point_cloud_path)

        self.stat_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_static.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
