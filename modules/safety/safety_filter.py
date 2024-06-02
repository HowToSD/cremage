# safety_filter
import os
import logging
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from einops import rearrange

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True
logger.info(f"AutoFeatureExtractor and StableDiffusionSafetyChecker connection to internet disabled : {local_files_only_value}")


def numpy_to_pil(input_images:np.ndarray) -> List[Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.

    Args:
        images (numpy.ndarray): Numpy array in either B, H, W, C or H, W, C format.
          Values are in [0, 1].
    Returns:
        List of PIL-format images.
    """
    if input_images.ndim == 3:  # if numpy array has H, W, C, then add B to make it B, H, W, C
        images = input_images[None, ...]
    else:
        images = input_images
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def load_replacement(x):
    """
    Loads a censored image (black image with a message to indicate replacement).

    Args:
        x (np.ndarray): An np.ndarray image in H, W, C format. Values are in [0, 1].

    Returns:
        An np.ndarray image in H, W, C format. Values are in [0, 1].
    """
    hwc = x.shape
    y = Image.new('RGB', (hwc[1], hwc[0]), "black")
    y_np = (np.array(y)/255.0).astype(x.dtype)
    try:
        # Create a draw object
        draw = ImageDraw.Draw(y)

        # Set the text to be added
        text = "Safety check filtered potentially sensitive image.\nTo change settings, go to File | Preferenses."
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            logger.info("arial.ttf not found. Using default font.")
            font = ImageFont.load_default()

        bbox = draw.multiline_textbbox((0, 0), text, font=font, align="left")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (hwc[1] - text_width) / 2
        text_y = (hwc[0] - text_height) / 2
        draw.multiline_text((text_x, text_y), text, font=font, fill="white", align="center")
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return y_np  # black image


def check_safety(x_image):
    """
    Filters out NSFW images.

    This code is based in the original SD 1.5 code. Check the license file
    in the project root.

    Args:
        x_image (numpy.ndarray): Numpy array in either B, H, W, C or H, W, C format.
          Values are in [0, 1].
    Returns:
        Tuple of filtered image array and a list. The first array contains filtered images. The second list contains
          the boolean flag to indicate if the image with the index is NSFW (True) or safe (False).
    """
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id, local_files_only=local_files_only_value)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id, local_files_only=local_files_only_value)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

class SafetyFilter():

    def __init__(self):
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id, local_files_only=local_files_only_value)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id, local_files_only=local_files_only_value)

    def __call__(self,
                 numpy_images=None,
                 torch_images=None):
        """
        Filter out unsafe images.

        Args:
            numpy_images (numpy.ndarray): Numpy array in B, H, W, C. Values are in [0, 1]
            torch_images (torch.tensor): torch tensor in B, C, H, W. Values are in [0, 1]

        Returns:
            Filtered image (np.ndarray): Images in B, H, W, C format. Values are in [0, 1]
        """
        device = None
        if numpy_images is not None:
            # Cremage note: safety_checker method below alters input "x_image"
            # so make sure that you keep a copy if you want to refer to it after the call.
            x_image = numpy_images.copy()
        elif torch_images is not None:
            device = torch_images.device
            x_image = rearrange(torch_images, "b c h w -> b h w c").detach().cpu().numpy()
        else:
            raise ValueError("Invalid input data")
        safety_checker_input = self.safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)

        # If there are three images in the array, and if the second image is NSFW,
        # then concept array contains [False, True, False]
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = load_replacement(x_checked_image[i])

        if torch_images is not None:
            x_checked_image = torch.tensor(x_checked_image)
            x_checked_image = rearrange(x_checked_image, "b h w c -> b c h w").to(device)
        return x_checked_image, has_nsfw_concept

