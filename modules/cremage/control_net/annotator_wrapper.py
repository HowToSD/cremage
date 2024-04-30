"""
Based on ControlNet 1.0 annotator code. See LICENSE for licensing information.
"""
import os
import sys
from typing import Dict, Any

import numpy as np
import PIL
from PIL import Image
import cv2

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..") 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
RESOURCE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "resources", "images")

sys.path = [MODULE_ROOT] + sys.path

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.openpose import OpenposeDetector
from annotator.hed import HEDdetector, nms
from annotator.mlsd import MLSDdetector
from annotator.uniformer import UniformerDetector


def _generate_one_arg(detector: Any, pil_image: Image,
                   image_resolution:int=512) -> Image:
    input_image = np.asarray(pil_image)[:,:,::-1]  # RGB to BGR
    img = resize_image(HWC3(input_image), image_resolution)
    detected_map = detector(img)  # This returns rank 2 for depth map, rank 3 for open pose
    detected_map = Image.fromarray(detected_map)
    return detected_map


def _generate_three_args(detector: Any, 
                         pil_image: Image,    
                         arg2,
                         arg3,                   
                         image_resolution:int=512) -> Image:
    input_image = np.asarray(pil_image)[:,:,::-1]  # RGB to BGR
    img = resize_image(HWC3(input_image), image_resolution)
    detected_map = detector(img, arg2, arg3)
    detected_map = Image.fromarray(detected_map)
    return detected_map


def _generate_one_arg_two_tuples_former(detector: Any, pil_image: Image,
                   image_resolution:int=512) -> Image:
    input_image = np.asarray(pil_image)[:,:,::-1]  # RGB to BGR
    img = resize_image(HWC3(input_image), image_resolution)
    detected_map, _ = detector(img)  # This returns rank 2 for depth map, rank 3 for open pose
    detected_map = Image.fromarray(detected_map)
    return detected_map

def _generate_two_args_two_tuples_latter(detector: Any,
                                  pil_image: Image,
                                  arg2: Any,
                                  image_resolution:int=512) -> Image:
    input_image = np.asarray(pil_image)[:,:,::-1]  # RGB to BGR
    img = resize_image(HWC3(input_image), image_resolution)
    _, detected_map = detector(img, arg2)
    detected_map = Image.fromarray(detected_map)
    return detected_map


def generate_canny(pil_image: Image,
                   low_threshold:int=100,
                   high_threshold:int=200,
                   image_resolution:int=512) -> Image:
    """
    Generates a canny edge map for the input image.

    Args:
        pil_image (Image): The input PIL image
        low_threshold (int): Low threshold for Canny detection
        high_threshold (int): High threshold for Canny detection
        image_resolution (int): Value used in resizing.
    """
    return _generate_three_args(CannyDetector(),
                                pil_image,
                                low_threshold,
                                high_threshold,
                                image_resolution=image_resolution)


def generate_depth_map(pil_image: Image,
                   image_resolution:int=512) -> Image:
    """
    Generates a depth map for the input image.

    Args:
        pil_image (Image): The input PIL image
        image_resolution (int): Value used in resizing.  If you use the minimum
            of width and height, then height or width won't be changed
            in the first part of the Canny wrapper code.
            Specifically, this is the formula used in the first part of the Canny
            wrapper code:

            k = float(resolution) / min(H, W)  # k will be 1 if resolution == min(H, W)
            H *= k
            W *= k
    """
    return _generate_one_arg_two_tuples_former(MidasDetector(),
                                 pil_image=pil_image,
                                 image_resolution=image_resolution)



def generate_normal_map(pil_image: Image,
                        bg_threshold:float=0.4, 
                        image_resolution:int=512) -> Image:
    """
    Generates a normal map for the input image.

    Args:
        pil_image (Image): The input PIL image
        bg_threshold: Background threshold. minimum:0.0, maximum:1.0, defaut:0.4
        image_resolution (int): Value used in resizing.
    """
    return _generate_two_args_two_tuples_latter(
                MidasDetector(),
                pil_image,
                bg_threshold,
                image_resolution=image_resolution)


def generate_open_pose(pil_image: Image,
                   image_resolution:int=512) -> Image:
    """
    Generates an open pose map for the input image.

    Args:
        pil_image (Image): The input PIL image
        image_resolution (int): Value used in resizing.
    """
    return _generate_one_arg_two_tuples_former(OpenposeDetector(),
                                 pil_image=pil_image,
                                 image_resolution=image_resolution)

def generate_fake_scribble(pil_image: Image,
                   image_resolution:int=512) -> Image:
    """
    Generates a fake scribble map for the input image.

    Args:
        pil_image (Image): The input PIL image
        image_resolution (int): Value used in resizing.
    """
    apply_hed = HEDdetector()
    input_image = np.asarray(pil_image)[:,:,::-1]  # RGB to BGR
    input_image = HWC3(input_image)
    detected_map = apply_hed(resize_image(input_image, image_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, _ = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    detected_map = nms(detected_map, 127, 3.0)
    detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
    detected_map[detected_map > 4] = 255
    detected_map[detected_map < 255] = 0
    detected_map = Image.fromarray(detected_map)
    return detected_map


def generate_scribble(pil_image: Image,
                   image_resolution:int=512) -> Image:
    """
    Generates a binary map to be fed to the scribble model from the input image.

    Args:
        pil_image (Image): The input PIL image
        image_resolution (int): Value used in resizing.
    """

    input_image = np.asarray(pil_image)[:,:,::-1]  # RGB to BGR
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, _ = img.shape

    detected_map = np.zeros_like(img, dtype=np.uint8)  # Fill with black
    detected_map[np.min(img, axis=2) < 127] = 255  # if < 127, set to white

    detected_map = Image.fromarray(detected_map)
    return detected_map


def generate_hed(pil_image: Image,
                   image_resolution:int=512) -> Image:
    """
    Generates a HED map for the input image.

    Args:
        pil_image (Image): The input PIL image
        image_resolution (int): Value used in resizing.
    """
    return _generate_one_arg(HEDdetector(),
                             pil_image=pil_image,
                             image_resolution=image_resolution)


def generate_mlsd(pil_image: Image,
                  value_threshold: float,
                  distance_threshold: float,
                  image_resolution:int=512) -> Image:
    """
    Generates a Hough map for the input image.

    Args:
        pil_image (Image): The input PIL image
        image_resolution (int): Value used in resizing.
    """
    return _generate_three_args(MLSDdetector(),
                                pil_image,
                                value_threshold,
                                distance_threshold,
                                image_resolution=image_resolution)

def generate_seg(pil_image: Image,
                  image_resolution:int=512) -> Image:
    """
    Generates a segmentation map for the input image.

    Args:
        pil_image (Image): The input PIL image
        image_resolution (int): Value used in resizing.
    """
    return _generate_one_arg(UniformerDetector(),
                                pil_image,
                                image_resolution=image_resolution)


if __name__ == "__main__":
    input_image = Image.open(os.path.join(RESOURCE_IMAGES_DIR, "turtle_line.png"))
    w, h = input_image.size
    resolution = min(w, h)

    detected_map = generate_canny(input_image,
                   low_threshold=100,
                   high_threshold=200,
                   image_resolution=resolution)
    