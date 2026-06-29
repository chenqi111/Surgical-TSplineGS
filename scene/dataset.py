import torch
from scene.cameras import Camera
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov


class FourDGSdataset(Dataset):
    def __init__(self, dataset, args, dataset_type):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type

    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R, T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask = None
            except:
                caminfo = self.dataset[index]
                image = PILtoTorch(caminfo.image, (caminfo.image.width, caminfo.image.height))
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask

            return Camera(
                colmap_id=index,
                R=R,
                T=T,
                FoVx=FovX,
                FoVy=FovY,
                image=image,
                gt_alpha_mask=None,
                image_name=f"{caminfo.image_name}",
                uid=index,
                data_device=torch.device("cpu"),
                time=time,
                mask=mask,
                metadata=caminfo.metadata,
                normal=caminfo.normal,
                depth=caminfo.depth,
                max_time=caminfo.max_time,
                sem_mask=caminfo.sem_mask,
                fwd_flow=caminfo.fwd_flow,
                bwd_flow=caminfo.bwd_flow,
                fwd_flow_mask=caminfo.fwd_flow_mask,
                bwd_flow_mask=caminfo.bwd_flow_mask,
                instance_mask=caminfo.instance_mask,
                target_tracks=caminfo.target_tracks,
                target_visibility=caminfo.target_visibility,
                target_tracks_static=caminfo.target_tracks_static,
                target_visibility_static=caminfo.target_visibility_static,
            )
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
