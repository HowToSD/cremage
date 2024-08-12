
"""
SDXL inpainting

Copyright (c) 2024 Hideyuki Inada

Note that this code uses Diffuser pipeline unlike txt2img or img2img.
LoRA and other advanced features are not supported for this flow.

References
[1] https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
"""
import os
import sys
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True
import logging
import gc
import random
import time
import json
import torch
from typing import List, Optional, Tuple

from diffusers import AutoPipelineForInpainting
from PIL.PngImagePlugin import PngInfo
import numpy as np
import PIL
from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
sys.path = [MODULE_ROOT] + sys.path
from cremage.ui.update_image_handler import update_image
from cremage.configs.preferences import load_user_config
from cremage.utils.random_utils import safe_random_int

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def generate(
        checkpoint:str=None,  # Not used for now
        out_dir:str=None,
        positive_prompt: str=None,
        negative_prompt: str=None,
        steps: int = 50,  # [2] sets default to 100
        guidance_scale: float = 4.0,  # See [2]
        height: int = 1024,
        width: int = 1024,
        number_of_batches = 1,
        batch_size = 1,
        ui_thread_instance=None,
        seed=-1,
        input_image=None,  # PIL image
        mask_image=None,
        denoising_strength=0.2,
        auto_face_fix=False,
        face_fix_options = None,
        safety_check=True,
        watermark=False,
        status_queue=None):
    """
    Generates an image based on the provided prompts, steps, and guidance scale.
    """

    if seed == -1:
        seed = safe_random_int()

    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=local_files_only_value).to("cuda")

    if status_queue:
        status_queue.put("Diffusers pipeline created")

    file_number_base = len(os.listdir(out_dir))

    if safety_check:
        from safety.safety_filter import SafetyFilter
        safety_filter = SafetyFilter()

    if watermark:
        from imwatermark import WatermarkEncoder
        wm = "Cremage"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    for batch_index in range(number_of_batches):
        new_seed_group_index = batch_size * batch_index
        random_number_generator = [torch.Generator(device=os.environ.get("GPU_DEVICE", "cpu")).manual_seed(seed + new_seed_group_index + i) for i in range(batch_size)]

        if status_queue:
            status_queue.put("Generating images ...")

        images = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            image=input_image.resize((width, height), resample=PIL.Image.LANCZOS),
            mask_image=mask_image,
            guidance_scale=guidance_scale, # 8.0
            num_inference_steps=steps, # 20. HF example recommends 15 and 30.
            strength=0.99,  # `strength` needs to be below 1.0
            generator=random_number_generator,
        )

        for i, image in enumerate(images.images):
            
            # Extra processing start
            if auto_face_fix:
                logger.debug("Applying face fix")
                status_queue.put("Applying face fix")
                from face_detection.face_detector_engine import face_fix
                image = face_fix(image, **face_fix_options)

            file_name = str(time.time())

            generation_parameters = {
                "time": time.time(),
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "sampling_iterations": steps,
                "image_height": height,
                "image_width": width,
                "cfg": guidance_scale,
                "seed": seed + new_seed_group_index + i,
                "safety_check": safety_check,
                "auto_face_fix": auto_face_fix,
                "denoising_strength": denoising_strength,
                "generator_model_type": "SDXL inpainting"
            }

            if auto_face_fix:
                generation_parameters["auto_face_fix_strength"] = face_fix_options["denoising_strength"]
                generation_parameters["auto_face_fix_prompt"] = face_fix_options["positive_prompt"]
                generation_parameters["auto_face_fix_face_detection_method"] = face_fix_options["detection_method"]

            file_number = file_number_base + new_seed_group_index + i
            time_str = time.time()
            file_name =  os.path.join(out_dir, f"{file_number:05}_{time_str}.png")
            str_generation_params = json.dumps(generation_parameters)
            metadata = PngInfo()
            metadata.add_text("generation_data", str_generation_params)

            # Safety check
            if safety_check:
                checked_np_images, _ = safety_filter(np.expand_dims(np.asarray(image), 0) / 255.0)
                image = Image.fromarray((checked_np_images * 255.0).astype(np.uint8)[0])
            image.save(file_name, pnginfo=metadata)

            if watermark:
                from cremage.utils.image_utils import put_watermark
                image = put_watermark(image, wm_encoder)

            str_generation_params = json.dumps(generation_parameters)
            # Pass img (PIL Image) to the main thread here!
            if ui_thread_instance:
                update_image(ui_thread_instance,
                                image,
                                generation_parameters=str_generation_params)

            # end single batch
    gc.collect()
    # end batch
    if status_queue:
        status_queue.put("Completed")
