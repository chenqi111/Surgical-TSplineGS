import argparse
import glob
import os
from PIL import Image
import numpy as np
import torch
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor

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
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--is_static", action="store_true")
    parser.add_argument("--grid_size", type=int, default=100)
    parser.add_argument("--query_frame", type=int, required=True, help="Single query frame index to process")
    args = parser.parse_args()

    folder_path = args.image_dir
    mask_dir = args.mask_dir
    frame_names = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))]
    out_dir = args.out_dir
    num_frames = len(frame_names)

    from cotracker.predictor import CoTrackerPredictor
    model = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False).to("cuda")
    checkpoint_url = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    model.model.load_state_dict(state_dict)
    
    video = read_video(folder_path)
    masks = read_mask(mask_dir).to("cuda")
    masks[masks>0] = 1.
    if args.is_static:
        masks = 1.0 - masks

    t = args.query_frame
    name_t = os.path.splitext(frame_names[t])[0]
    file_matches = glob.glob(f"{out_dir}/{name_t}_*.npy")
    if len(file_matches) == num_frames:
        print(f"Already completed, skipping frame {t}")
        return

    current_mask = masks[:,t].unsqueeze(1)
    for j in range(num_frames):
        name_j = os.path.splitext(frame_names[j])[0]
        out_path = f"{out_dir}/{name_t}_{name_j}.npy"
        if os.path.exists(out_path):
            print(f"Skipping existing {out_path}")
            continue
            
        if j > t:
            current_video = video[:,t:j+1].to("cuda")
        elif j < t:
            current_video = torch.flip(video[:,j:t+1], dims=(1,)).to("cuda")
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
        if j > t:
            current_pred = pred[0,-1]
            start_pred = pred[0,0]
        else:
            current_pred = pred[0,0]
            start_pred = pred[0,-1]

        np.save(f"{out_dir}/{name_t}_{name_j}.npy", current_pred.cpu().numpy())
        torch.cuda.empty_cache()
    
    # Save self-track
    if not os.path.exists(f"{out_dir}/{name_t}_{name_t}.npy"):
        np.save(f"{out_dir}/{name_t}_{name_t}.npy", start_pred.cpu().numpy())
    print(f"Completed frame {t}")

if __name__ == "__main__":
    main()
