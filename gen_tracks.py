
""" Code borrowed from 
https://github.com/vye16/shape-of-motion/blob/main/preproc/compute_tracks_torch.py
"""
import argparse
import glob
import os

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from cotracker.utils.visualizer import Visualizer

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.concatenate([np.array(Image.open(frame_path)).transpose(2, 0, 1)[None, None] for frame_path in frame_paths], axis=1)
    video = torch.from_numpy(video).float()
    return video

def read_mask(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.concatenate([np.array(Image.open(frame_path))[None, None] for frame_path in frame_paths], axis=1)
    video = torch.from_numpy(video).float()
    return video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--mask_dir", type=str, required=True, help="mask dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--is_static", action="store_true")
    parser.add_argument("--grid_size", type=int, default=100, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    args = parser.parse_args()

    folder_path = args.image_dir
    mask_dir = args.mask_dir
    frame_names = [
        os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))
    ]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)

    done = True
    for t in range(len(frame_names)):
        for j in range(len(frame_names)):
            name_t = os.path.splitext(frame_names[t])[0]
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                done = False
                break
    print(f"{done}")
    if done:
        print("Already done")
        return

    ## Load model
    from cotracker.predictor import CoTrackerPredictor
    model = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False).to(DEFAULT_DEVICE)
    checkpoint_url = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    model.model.load_state_dict(state_dict)
    video = read_video(folder_path)  # keep on CPU to save GPU memory
    
    masks = read_mask(mask_dir).to(DEFAULT_DEVICE)
    
    masks[masks>0] = 1.
    if args.is_static:
        masks = 1.0 - masks
    
    _, num_frames,_, height, width = video.shape
    vis = Visualizer(save_dir=os.path.join(out_dir, "vis"), pad_value=120, linewidth=3)

    for t in tqdm(range(num_frames), desc="query frames"):
        name_t = os.path.splitext(frame_names[t])[0]
        file_matches = glob.glob(f"{out_dir}/{name_t}_*.npy")
        if len(file_matches) == num_frames:
            print(f"Already computed tracks with query {t} {name_t}")
            continue

        current_mask = masks[:,t].unsqueeze(1)
        start_pred = None
        
        for j in range(num_frames):
            if j > t:
                current_video = video[:,t:j+1].to(DEFAULT_DEVICE)
            elif j < t:
                current_video = torch.flip(video[:,j:t+1], dims=(1,)).to(DEFAULT_DEVICE) # reverse
            else:
                continue
            
            pred_tracks, pred_visibility = model(
                current_video,
                grid_size=args.grid_size,
                grid_query_frame=0,
                backward_tracking=False,
                segm_mask=current_mask
            )
            del current_video
            torch.cuda.empty_cache()
            

            pred = torch.cat([pred_tracks, pred_visibility.unsqueeze(-1)], dim=-1)
            current_pred = pred[0,-1]
            start_pred = pred[0,0]

            # save
            name_j = os.path.splitext(frame_names[j])[0]
            np.save(f"{out_dir}/{name_t}_{name_j}.npy", current_pred.cpu().numpy())
            
        np.save(f"{out_dir}/{name_t}_{name_t}.npy", start_pred.cpu().numpy())



if __name__ == "__main__":
    main()
