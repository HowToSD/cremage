
"""
Stable Cascade txt2img

Copyright (c) 2024 Hideyuki Inada

Code to interact with Stable Cascade is based on [1], [2] and [3] below and is covered under Apache 2.0 license.
Refer to the license document at the root of this project.

References
[1] https://huggingface.co/docs/diffusers/v0.29.2/api/pipelines/pixart_sigma
[2] https://github.com/huggingface/diffusers/blob/v0.29.2/docs/source/en/api/pipelines/pixart_sigma.md
[3] https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_cascade
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

from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from PIL.PngImagePlugin import PngInfo
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
sys.path = [MODULE_ROOT] + sys.path
from cremage.ui.update_image_handler import update_image
from cremage.configs.preferences import load_user_config
from cremage.const.const import GMT_STABLE_CASCADE
from cremage.utils.random_utils import safe_random_int

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(
        model_id=None,
        checkpoint:str=None,  # Not used for now
        out_dir:str=None,
        positive_prompt: str=None,
        negative_prompt: str=None,
        steps: int = 20,
        guidance_scale: float = 4.0,
        height: int = 1024,
        width: int = 1024,
        number_of_batches = 1,
        batch_size = 1,
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
    """
    start_time = time.perf_counter()

    if batch_size != 1:  # Specifying bs > 1 results in the similar/same image being generated for the same batch
        logger.warn("Currently only 1 is supported for the batch size. Number of batches were adjusted to generate the specified number of images.")
        number_of_batches *= batch_size
        batch_size = 1

    if seed == -1:
        seed = safe_random_int()

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

    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior",
                                                        variant="bf16",
                                                        torch_dtype=torch.bfloat16,
                                                        local_files_only=local_files_only_value)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",
                                                           variant="bf16",
                                                           torch_dtype=torch.float16,
                                                           local_files_only=local_files_only_value)

    for batch_index in range(number_of_batches):
        new_seed_group_index = batch_size * batch_index
        random_number_generator = [torch.Generator(device=os.environ.get("GPU_DEVICE", "cpu")).manual_seed(seed + new_seed_group_index + i) for i in range(batch_size)]

        if status_queue:
            status_queue.put("Generating images ...")

        # If you specify 1024x1024 for output
        # For bs=1, prior generates image embedding in shape (1, 16, 24, 24)
        prior.enable_model_cpu_offload()
        prior_output = prior(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,  # default 4
            num_images_per_prompt=batch_size,  # 1,  # FIXME
            num_inference_steps=steps,  # default 20
            generator=random_number_generator,
        )
        decoder.enable_model_cpu_offload()

        image_list = []
        for i in range(batch_size):
            with torch.no_grad():
                outputs = decoder(
                        image_embeddings=prior_output.image_embeddings.to(torch.float16),
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=0.0,
                        output_type="pil",
                        num_inference_steps=10,   # FIXME. 10 is the default
                        generator=random_number_generator[i],
                        num_images_per_prompt=1,
                    )
                image = outputs.images[0]
            image_list.append(image)
        images = image_list

        for i, image in enumerate(images):
            
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

            if checkpoint and os.path.exists(checkpoint):
                ldm_model = os.path.basename(checkpoint)
            else:
                ldm_model = "None"
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
                "model_id": model_id,
                "ldm_model": ldm_model,
                "generator_model_type": GMT_STABLE_CASCADE
            }
            if auto_face_fix:
                generation_parameters["auto_face_fix_strength"] = auto_face_fix_strength
                generation_parameters["auto_face_fix_prompt"] = auto_face_fix_prompt
                generation_parameters["auto_face_fix_face_detection_method"] = auto_face_fix_face_detection_method

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
    
    flush()
    # end batch

    end_time = time.perf_counter()
    logger.info(f"Completed. Time elapsed: {end_time - start_time}")

    if status_queue:
        status_queue.put(f"Completed. Time elapsed: {end_time - start_time:0.1f} seconds")
