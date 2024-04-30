"""
GFPGAN face enhancement.

Based on the code from
https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
Licensed under Apache 2
For details, see LICENSE file at the project root directory.

Python package (PIP) dependencies:
* gfpgan
* realesrgan
* basicsr
"""
# TODO. Fix the model saving directory.
#   Currently, detection_Resnet50_Fina.pth and parsing_parsenet.pth are saved
#   under the project root/gfpgan/weights
import os
import sys
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import torch

import sys

# Patch torchvision as this module is deprecated but basicsr still calls it
import torchvision.transforms.functional as torchvision_transforms_functional
sys.modules['torchvision.transforms.functional_tensor'] = torchvision_transforms_functional
from basicsr.utils import imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gfpgan")

@dataclass
class GfpModelInfo:
    arch: str
    channel_multiplier: int
    model_name: str
    url: str


def model_selector(model_version="1.4"):
    if model_version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif model_version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif model_version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif model_version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif model_version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {model_version}.')

    return GfpModelInfo(arch, channel_multiplier, model_name, url)


def gfp_wrapper(args):
    if args.bg_upsampler == 'realesrgan':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=args.bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available())  # need to set False in CPU mode
    else:
        bg_upsampler = None
   
    model_info = model_selector(args.version)
    arch = model_info.arch
    channel_multiplier = model_info.channel_multiplier
    model_name = model_info.model_name
    url = model_info.url
    
    # Determine model paths
    model_path = os.path.join(MODEL_DIR, model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = MODEL_DIR
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    input_img = cv2.imread(args.input, cv2.IMREAD_COLOR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=args.weight)

    imwrite(restored_img, args.output)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    """
    Inference using GFPGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Input image')
    parser.add_argument(
        '-v', '--version', type=str, default='1.4', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    parser.add_argument(
        '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    parser.add_argument('-o', '--output', type=str, default="tmpgfp.png", help='Output image path')
    
    # FIXME: Change test image data paths
    args = parser.parse_args(["--input", "human_couple.png",
                              "--output", "testgfpgan.png"])  

    gfp_wrapper(args)
