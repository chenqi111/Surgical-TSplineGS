import argparse
import numpy as np
import torch
from PIL import Image

import glob
import os
import sys

def gen_depth(model, args):
    from submodules.UniDepth.unidepth.utils import colorize
    
    focal = 500 # default 500 for all datasets

    images_list = sorted(glob.glob(os.path.join(args.image_dir, '*.png')))
    os.makedirs(args.out_dir, exist_ok=True)
    for image in images_list:
        rgb = np.array(Image.open(image).convert('RGB'))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
        
        H, W = rgb_torch.shape[1], rgb_torch.shape[2]
        intrinsics_torch = torch.from_numpy(np.array([[focal, 0, H/2],
                                                    [0, focal, W/2],
                                                    [0, 0 ,1]])).float()
        # predict
        predictions = model.infer(rgb_torch, intrinsics_torch)
        depth_pred = predictions["depth"].squeeze().cpu().numpy()[..., None]
        
        fname = os.path.basename(image)
        np.save(os.path.join(args.out_dir, fname.replace('png', 'npy')), depth_pred)

        # # colorize
        depth_pred_col = colorize(depth_pred)
        Image.fromarray(depth_pred_col).save(os.path.join(args.out_dir, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--depth_type", type=str, help="depth type disp or depth", default="depth")  
    parser.add_argument("--depth_model", type=str, help="unidepth model", default="v2old")
    args = parser.parse_args()
    
    print("Torch version:", torch.__version__)
    
    if args.depth_type == "disp":  
        os.makedirs(args.out_dir, exist_ok=True)     
        cmd = f"{sys.executable} submodules/mega-sam/Depth-Anything/run_videos.py --encoder vitl \
        --load-from submodules/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth \
        --localhub \
        --img-path {args.image_dir} \
        --outdir {args.out_dir}"
        os.system(cmd)
    elif args.depth_type == "depth":
        import sys as _sys
        _sys.path.insert(0, 'submodules/UniDepth')
        from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
        type_ = "l"  # available types: s, b, l
        name = f"unidepth-{args.depth_model}-vit{type_}14"
        if args.depth_model == "v2":
            model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
            model.interpolation_mode = "bilinear"
        elif args.depth_model == "v2old":
            model = UniDepthV2old.from_pretrained(f"lpiccinelli/{name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        gen_depth(model, args)
    else:
        raise ValueError("depth_type must be either 'disp' or 'depth'")

