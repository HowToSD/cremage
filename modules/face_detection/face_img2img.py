"""
Copyright (C) 2024, Hideyuki Inada. All rights reserved.

Face detection code for OpenCV is based on the code downloaded from
https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detect.py
Licensed under Apache License 2.0
https://github.com/opencv/opencv/blob/4.x/LICENSE

OpenCV face detection model: face_detection_yunet_2023mar.onnx
The model was downloaded from 
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
Licensed under the MIT license.

See the license in the project root directory.
"""
import os
import sys
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

import logging
import argparse
from io import BytesIO
from typing import List

import numpy as np
import cv2 as cv
import PIL
from PIL import Image
import tempfile
import threading
import shutil
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from transformers import ViTImageProcessor, ViTForImageClassification  # For face classification

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.misc_utils import get_tmp_dir
from cremage.utils.misc_utils import generate_lora_params
from sd.options import parse_options as sd15_parse_options
from sd.img2img import generate as sd15_img2img_generate
from sd.inpaint import generate as sd15_inpaint_generate

FACE_FIX_TMP_DIR = os.path.join(get_tmp_dir(), "face_fix.tmp")
OPENCV_FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_ROOT, "opencv", "face_detection_yunet_2023mar.onnx")
from cremage.const.const import GMT_SD_1_5, GMT_SDXL

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def face_image_to_image(input_image=None,
                        meta_prompt=None,
                        output_dir=FACE_FIX_TMP_DIR,
                        generator_model_type = GMT_SD_1_5,
                        positive_prompt = "",
                        negative_prompt = "",
                        model_path=None,
                        lora_models = None,
                        lora_weights = None,
                        embedding_path = None,
                        sampling_steps = None,
                        seed = -1,
                        vae_path = None,
                        sampler = None,
                        target_edge_len = None,
                        denoising_strength = None,
                        enable_face_id = False,
                        face_input_image_path = None,
                        face_model_full_path = None,
                        discretization = None,
                        discretization_sigma_min = None,
                        discretization_sigma_max = None,
                        discretization_rho = None,
                        guider = None,
                        linear_prediction_guider_min_scale = None,
                        linear_prediction_guider_max_scale = None,
                        triangle_prediction_guider_min_scale = None,
                        triangle_prediction_guider_max_scale = None,
                        sampler_s_churn = None,
                        sampler_s_tmin = None,
                        sampler_s_tmax = None,
                        sampler_s_noise = None,
                        sampler_eta = None,
                        sampler_order = None,
                        clip_skip = None
                        ):
    """
    Event handler for the Generation button click

    Args:
        meta_prompt (str): Gender string of the face detected by the gender ML model
    """
    logger.info("face_image_to_image")
    is_sdxl = True if generator_model_type == GMT_SDXL else False

    if positive_prompt:
        positive_prompt = positive_prompt
    else:  # use blank
        positive_prompt = ""

    # Prepend meta_prompt
    if meta_prompt:
        positive_prompt = "face of " + meta_prompt + ", " + positive_prompt

    if negative_prompt:
        negative_prompt = negative_prompt
    else:  # use blank
        negative_prompt = ""

    if os.path.exists(output_dir) and output_dir.find("face_fix") >= 0:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    clip_skip = str(clip_skip)

    sampling_steps = str(sampling_steps)
    args_list = ["--prompt", positive_prompt,
                    "--negative_prompt", negative_prompt,
                    "--H", str(target_edge_len),
                    "--W", str(target_edge_len),
                    "--sampler", sampler,
                    "--sampling_steps", sampling_steps,
                    "--clip_skip", clip_skip,
                    "--seed", str(seed),
                    "--n_samples", str(1),
                    "--n_iter",str(1),
                    "--ckpt", model_path,
                    "--embedding_path", embedding_path,
                    "--vae_ckpt", vae_path,
                    "--lora_models", lora_models,
                    "--lora_weights", lora_weights,
                    "--outdir", output_dir]

    input_image_path = os.path.join(get_tmp_dir(), "input_image.png")
    input_image.save(input_image_path)

    args_list += [
        "--init-img", input_image_path,
        "--strength", str(denoising_strength)
    ]

    # FaceID
    if enable_face_id:
        args_list += [
            "--face_input_img", face_input_image_path,
            "--face_model", face_model_full_path
        ]

    # multithreading seems to cause memory leak when you move model between
    # cpu and gpu.
    use_thread = False
    
    if is_sdxl:
        from sdxl.sdxl_pipeline.options import parse_options as sdxl_parse_options
        from sdxl.sdxl_pipeline.sdxl_image_generator import generate as sdxl_generate

        args_list += [
            "--refiner_strength", "0", # str(refiner_strength),
            "--discretization", discretization,
            "--discretization_sigma_min", str(discretization_sigma_min),
            "--discretization_sigma_max", str(discretization_sigma_max),
            "--discretization_rho", str(discretization_rho),
            "--guider", guider,
            "--linear_prediction_guider_min_scale", str(linear_prediction_guider_min_scale),
            "--linear_prediction_guider_max_scale", str(linear_prediction_guider_max_scale),
            "--triangle_prediction_guider_min_scale", str(triangle_prediction_guider_min_scale),
            "--triangle_prediction_guider_max_scale", str(triangle_prediction_guider_max_scale),
            "--sampler_s_churn", str(sampler_s_churn),
            "--sampler_s_tmin", str(sampler_s_tmin),
            "--sampler_s_tmax", str(sampler_s_tmax),
            "--sampler_s_noise", str(sampler_s_noise),
            "--sampler_eta", str(sampler_eta),
            "--sampler_order", str(sampler_order)
        ]
        options = sdxl_parse_options(args_list)
        setattr(options, "disable_seed", True)
        generate_func=sdxl_generate

        # Start the image generation thread
        if use_thread:
            thread = threading.Thread(
                target=generate_func,
                kwargs={'options': options,
                        'generation_type': "img2img",
                        'ui_thread_instance': None})  # FIXME
        else:
            generate_func(options=options,
                         generation_type="img2img")
    else:  # SD15
        options = sd15_parse_options(args_list)
        generate_func=sd15_img2img_generate
        setattr(options, "disable_seed", True)

        if use_thread:
            # Start the image generation thread
            thread = threading.Thread(
                target=generate_func,
                kwargs={'options': options,
                        'ui_thread_instance': None})  # FIXME
        else:
            generate_func(options=options)
            print("*****")
            print("DEBUG POINT 1")

        # app.ui_to_ml_queue.put({
        #     "type": MP_LDM,
        #     "mode": app.generation_mode,
        #     "command":{'options': options,
        #             'ui_thread_instance': True}
        # })

    if use_thread:
        thread.start()

        thread.join()  # Wait until img2img is done.

    # Get the name of the output image
    files = os.listdir(output_dir)
    assert len(files) == 1
    file_name = os.path.join(output_dir, files[0])
    if os.path.exists(file_name):
        image = Image.open(file_name)
        image2 = image.copy()
        image.close()
        os.remove(file_name)  # Remove the temporary image
        image = image2
        return image
    else:
        raise ValueError(f"Invalid output file from img2img {file_name}")

