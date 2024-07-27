
"""
sd3 txt2img

Copyright (c) 2024 Hideyuki Inada
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

from diffusers import StableDiffusion3Pipeline
from PIL.PngImagePlugin import PngInfo
import numpy as np
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
        checkpoint:str=None,
        out_dir:str=None,
        positive_prompt: str=None,
        negative_prompt: str=None,
        steps: int = 28,
        guidance_scale: float = 7.0,
        height: int = 1024,
        width: int = 1024,
        number_of_batches = 1,
        batch_size = 1,
        quantize_t5 = False,
        ui_thread_instance=None,
        seed=-1,
        auto_face_fix=False,
        auto_face_fix_strength=0.3,
        auto_face_fix_face_detection_method="OpenCV",
        auto_face_fix_prompt="",
        safety_check=True,
        watermark=False,
        status_queue=None):
    """
    Generates an image based on the provided prompts, steps, and guidance scale.

    References
    https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3

    Args:
        positive_prompt (str): The positive prompt for image generation.
        negative_prompt (str): The negative prompt for image generation.
        steps (int): The number of inference steps. Default is 28.
        guidance_scale (float): The guidance scale for image generation. Default is 7.0.

    Returns:
        Tuple: A tuple containing the generated image, file information, and metadata.
    """

    if seed == -1:
        seed = safe_random_int()

    if quantize_t5:
        from transformers import T5EncoderModel, BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_id = checkpoint
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_3",
            quantization_config=quantization_config,
            local_files_only=local_files_only_value
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            text_encoder_3=text_encoder,
            device_map="balanced",
            torch_dtype=torch.float16,
            local_files_only=local_files_only_value)
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            checkpoint, torch_dtype=torch.float16,
            local_files_only=local_files_only_value)
        pipe.enable_model_cpu_offload()
        # pipe = pipe.to("cuda")

    if status_queue:
        status_queue.put("Diffusers pipeline created")

    # Note that only one scheduler seems to be officially supported at this point:
    # Below returns only 1.
    # See:
    # https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers?schedulers=LMSDiscreteScheduler#compare-schedulers
    # [<class 'diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler'>]
    # compatibles = pipe.scheduler.compatibles

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
        random_number_generator = [torch.Generator(device="cuda").manual_seed(seed + new_seed_group_index + i) for i in range(batch_size)]

        if status_queue:
            status_queue.put("Generating images ...")

        images = pipe(
            positive_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            generator=random_number_generator
        )

        for i, image in enumerate(images.images):
            
            # Extra processing start
            if auto_face_fix:
                from face_fixer import FaceFixer
                logger.debug("Applying face fix")
                status_queue.put("Applying face fix")
                app = load_user_config()
                if auto_face_fix_prompt:
                    auto_face_fix_prompt = auto_face_fix_prompt
                else:
                    auto_face_fix_prompt = positive_prompt
                face_fixer = FaceFixer(
                    preferences=app,
                    positive_prompt=auto_face_fix_prompt,
                    negative_prompt=negative_prompt,
                    denoising_strength=auto_face_fix_strength,
                    procedural=True,
                    status_queue=status_queue)

                if auto_face_fix_face_detection_method == "InsightFace":
                    image = face_fixer.fix_with_insight_face(image)
                elif auto_face_fix_face_detection_method == "OpenCV":
                    image = face_fixer.fix_with_opencv(image)
                else:
                    logger.info(f"Ignoring unsupported face detection method: {auto_face_fix_face_detection_method}")

            file_name = str(time.time())

            generation_parameters = {
                "time": time.time(),
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "sd3_ldm_model_path": checkpoint,
                "sampling_iterations": steps,
                "image_height": height,
                "image_width": width,
                "cfg": guidance_scale,
                "seed": seed + new_seed_group_index + i,
                "safety_check": safety_check,
                "auto_face_fix": auto_face_fix,
                "generator_model_type": "SD 3"
            }

            if auto_face_fix:
                generation_parameters["auto_face_fix_strength"] = auto_face_fix_strength
                generation_parameters["auto_face_fix_prompt"] = auto_face_fix_prompt
                generation_parameters["auto_face_fix_face_detection_method"] = auto_face_fix_face_detection_method

            # generation_parameters = {
            #         # "ldm_model": os.path.basename(opt.ckpt),
            #         # "vae_model": os.path.basename(opt.vae_ckpt),
            #         # "lora_models": opt.lora_models,
            #         # "lora_weights": opt.lora_weights,
            #         # "sampler": opt.sampler.replace("Sampler", ""),
            #         # "sampling_iterations": opt.sampling_steps,
            #         # #  "clip_skip": opt.clip_skip,  # Cremage. Commenting out as CLIP skip for SDXL creates confusion
            #         # "watermark": False,
            #         # "safety_check": False
            #     }

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
