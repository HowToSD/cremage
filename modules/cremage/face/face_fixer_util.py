"""
Copyright (C) 2024, Hideyuki Inada. All rights reserved.
"""
import os
import sys
import platform
import logging
import argparse
import json
from io import BytesIO
import tempfile
import threading
import shutil
from typing import Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import cv2 as cv
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk
import cairo


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
OPENCV_FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_ROOT, "opencv", "face_detection_yunet_2023mar.onnx")

from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.gtk_utils import text_view_get_text, create_combo_box_typeahead
from cremage.utils.misc_utils import generate_lora_params
from cremage.utils.misc_utils import get_tmp_dir
from cremage.ui.model_path_update_handler import update_ldm_model_name_value_from_ldm_model_dir
from cremage.ui.model_path_update_handler import update_sdxl_ldm_model_name_value_from_sdxl_ldm_model_dir
from cremage.const.const import GMT_SD_1_5, GMT_SDXL
from face_detection.face_detector_engine import mark_face_with_insight_face
from face_detection.face_detector_engine import mark_face_with_opencv
from face_detection.face_detector_engine import parse_face_data
from face_detection.face_detector_engine import process_face

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

copy_keys = [
    "embedding_path",
    "sampling_steps",
    "denoising_strength",
    "seed",
    "clip_skip",
    "discretization",
    "discretization_sigma_min",
    "discretization_sigma_max",
    "discretization_rho",
    "guider",
    "linear_prediction_guider_min_scale",
    "linear_prediction_guider_max_scale",
    "triangle_prediction_guider_min_scale",
    "triangle_prediction_guider_max_scale",
    "sampler_s_churn",
    "sampler_s_tmin",
    "sampler_s_tmax",
    "sampler_s_noise",
    "sampler_eta",
    "sampler_order"
]


def build_face_process_params_from_preferences(
              preferences: Dict[str, Any],
              generator_model_type=GMT_SD_1_5,
              sd15_model_name=None,
              sdxl_model_name=None,
              enable_face_id=False,
              face_input_image_path=None,
              face_model_full_path=None):

    if generator_model_type == GMT_SDXL:
        is_sdxl = True
    else:
        is_sdxl = False

    # Set up parameters
    if is_sdxl:
        target_edge_len = 1024
        model_dir = preferences["sdxl_ldm_model_path"]
        model_name = sdxl_model_name # Use the model on UI
        vae_path = "None" if preferences["sdxl_vae_model"] == "None" \
            else os.path.join(
                    preferences["sdxl_vae_model_path"],
                    preferences["sdxl_vae_model"])
        sampler = preferences["sdxl_sampler"]+"Sampler"
    else:  # SD1.5
        target_edge_len = 512
        model_dir = preferences["ldm_model_path"]
        model_name = sd15_model_name # Use the model on UI
        vae_path = "None" if preferences["vae_model"] == "None" \
            else os.path.join(
                    preferences["vae_model_path"],
                    preferences["vae_model"])
        sampler = preferences["sampler"]

    model_path = os.path.join(model_dir, model_name)
    lora_models, lora_weights = generate_lora_params(preferences, sdxl=is_sdxl)

    retval = dict()
    retval["generator_model_type"] = generator_model_type
    retval["target_edge_len"] = target_edge_len
    retval["model_path"] = model_path
    retval["vae_path"] = vae_path
    retval["lora_models"] = lora_models
    retval["lora_weights"] = lora_weights
    retval["sampler"] = sampler
    retval["enable_face_id"] = enable_face_id
    retval["face_input_image_path"] = face_input_image_path
    retval["face_model_full_path"] = face_model_full_path
    for k in copy_keys:
        retval[k] = preferences[k]
    return retval
