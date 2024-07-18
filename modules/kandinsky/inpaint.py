
"""
Kandinsky inpainting

Copyright (c) 2024 Hideyuki Inada

References
[1] https://huggingface.co/docs/diffusers/en/using-diffusers/kandinsky?text-to-image=Kandinsky+2.2&image-to-image=Kandinsky+2.2
https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder
[2] https://huggingface.co/docs/diffusers/en/api/pipelines/kandinsky_v22#diffusers.KandinskyV22InpaintCombinedPipeline
"""
import os
import sys
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
        safety_check=True,
        watermark=False,
        status_queue=None):
    """
    Generates an image based on the provided prompts, steps, and guidance scale.
    """

    if seed == -1:
        seed = random.getrandbits(32)

    pipe = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

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
        # Note that generator list is created by doubling the size of batch size
        # This is specific to Kandinsky and to work around the exception specific to an issue found.
        random_number_generator = [torch.Generator(device="cuda").manual_seed(seed + new_seed_group_index + i) for i in range(batch_size*2)]

        if status_queue:
            status_queue.put("Generating images ...")

        images = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            prior_guidance_scale=1.0,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            generator=random_number_generator,
            image = input_image.resize((width, height), resample=PIL.Image.LANCZOS),
            mask_image = mask_image.resize((width, height), resample=PIL.Image.LANCZOS)
        )

        for i, image in enumerate(images.images):
            
            # Extra processing start
            if auto_face_fix:
                from face_fixer import FaceFixer
                logger.debug("Applying face fix")
                status_queue.put("Applying face fix")
                app = load_user_config()
                face_fixer = FaceFixer(
                    preferences=app,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    procedural=True,
                    status_queue=status_queue)
                
                image = face_fixer.fix_with_insight_face(image)

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
                "generator_model_type": "Kandinsky 2.2"
            }

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
