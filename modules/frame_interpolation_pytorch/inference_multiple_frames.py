import bisect
import sys
import os
from tqdm import tqdm
import torch
import numpy as np
import cv2

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

sys.path = [MODULE_ROOT] + sys.path

from frame_interpolation_pytorch.util import load_image
from frame_interpolation_pytorch.interpolator import Interpolator

def inference_multiple_frames(model_path=None,
                              input_dir_path=None,
                              input_file_prefix="frame",
                              save_path=None, 
                              gpu=True, 
                              interpolation_frames=3, 
                              output_fps=25, 
                              half=False):

    model = Interpolator()
    sd = torch.load(model_path)
    model.load_state_dict(sd, strict=True)
    model.to("cuda")
    model.eval()

    if not half:
        model.float()

    if gpu and torch.cuda.is_available():
        if half:
            model = model.half()
        else:
            model.float()
        model = model.cuda()

    file_list = list()
    files = os.listdir(input_dir_path)
    for f in files:
        p = os.path.join(input_dir_path, f)
        if (p.endswith(".png") or (".jpg")) and f.startswith(input_file_prefix):
            file_list.append(p)
    file_list = sorted(file_list)
    num_files = len(file_list)
    print(f"Number of frames found: {num_files}")
    combined_frames = list()
    for i in tqdm(range(1, num_files), 'Generating in-between frames'):

        frames = _interpolate_two_frames(
            file_list[i-1],
            file_list[i],
            interpolation_frames,
            gpu,
            half,
            model,
            skip_first=i!=1)
        combined_frames.append(frames)
    frames = np.concatenate(combined_frames, axis=0)
    print(frames.shape)
    
    video_folder = os.path.split(save_path)[0]
    if video_folder and os.path.exists(video_folder) is False:
        os.makedirs(video_folder)

    w, h = frames[0].shape[1::-1]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(save_path, fourcc, output_fps, (w, h))

    # frame 0, frame 1, frame 2, frame 3, frame 4, frame 5
    for i, frame in enumerate(frames):
        writer.write(frame)

    writer.release()


def _interpolate_two_frames(img1, img2, inter_frames, gpu, half, model, skip_first):
    """
    Args:
        skip_first (bool): Do not return the first frame.
    """
    img_batch_1, crop_region_1 = load_image(img1)
    img_batch_2, crop_region_2 = load_image(img2)

    img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
    img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

    results = [
        img_batch_1,
        img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i]
        x1 = results[end_i]

        if gpu and torch.cuda.is_available():
            if half:
                x0 = x0.half()
                x1 = x1.half()
            x0 = x0.cuda()
            x1 = x1.cuda()

        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
        del remains[step]

    y1, x1, y2, x2 = crop_region_1
    frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]
    
    if skip_first:
        frames = frames[1:]
    return frames


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test frame interpolator model')

    parser.add_argument('model_path', type=str, help='Path to the TorchScript model')
    parser.add_argument('input_dir_path', type=str, help='Path to the directory containing frames')
    
    parser.add_argument('--save_path', type=str, default='img1 folder', help='Path to save the interpolated frames')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--fp16', action='store_true', help='Use FP16')
    parser.add_argument('--frames', type=int, default=18, help='Number of frames to interpolate')
    parser.add_argument('--fps', type=int, default=25, help='FPS of the output video')

    # Input: 25 frames
    # 24 intervals
    # For each interval, 3 interpolated frames
    # 24 x 3 = 72 interpolated frames
    # Total frames = 25 + 72 = 97
    # So the time is roughly 4 times as long as the original
    # 1 sec video turns into 4 sec video
    args = parser.parse_args([
        "/home/pup/data/programs/python/cremage/models/film/film.safe.pt",
        "/home/pup/data/programs/python/cremage_resources/example_frames",
        "--save_path", "/home/pup/data/programs/python/cremage_resources/tmp/tmp2.mp4",
        "--frames", "3",  
        "--gpu",
        "--fps", "25"
    ])

    inference_multiple_frames(
        model_path=args.model_path,
        input_dir_path=args.input_dir_path,
        input_file_prefix="frame",
        save_path=args.save_path,
        gpu=args.gpu,
        interpolation_frames=args.frames,
        output_fps=args.fps,
        half=args.fp16)
