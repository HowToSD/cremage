
"""
PixArt-Σ txt2img

Copyright (c) 2024 Hideyuki Inada

Code to interact with PixArt-Σ is based on [1] and [2] below and is covered under Apache 2.0 license.
Refer to the license document at the root of this project.

See pixart_sigma_utils.py for the list of model IDs.

References
[1] https://huggingface.co/docs/diffusers/v0.29.2/api/pipelines/pixart_sigma
[2] https://github.com/huggingface/diffusers/blob/v0.29.2/docs/source/en/api/pipelines/pixart_sigma.md
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

from transformers import T5EncoderModel
from diffusers import PixArtSigmaPipeline
from PIL.PngImagePlugin import PngInfo
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
sys.path = [MODULE_ROOT] + sys.path
from cremage.ui.update_image_handler import update_image
from cremage.configs.preferences import load_user_config
from cremage.utils.pixart_sigma_utils import update_pixart_sigma_model_with_custom_model
from cremage.utils.pixart_sigma_utils import DEFAULT_MODEL_ID, MODEL_ID_LIST
from cremage.utils.random_utils import safe_random_int

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def generate(
        model_id=None,
        checkpoint:str=None,  # Not used for now
        out_dir:str=None,
        positive_prompt: str=None,
        negative_prompt: str=None,
        steps: int = 20,
        guidance_scale: float = 4.5,
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

    if batch_size != 1:
        logger.warn("Currently only 1 is supported for the batch size. Number of batches were adjusted to generate the specified number of images.")
        number_of_batches *= batch_size
        batch_size = 1

    if model_id not in MODEL_ID_LIST:
        model_id = DEFAULT_MODEL_ID

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

    for batch_index in range(number_of_batches):
        new_seed_group_index = batch_size * batch_index
        # TODO: Fix the issue where if you specify random number generator when bs > 1, the same image is generated for all images within
        # the batch.
        random_number_generator = [torch.Generator(device="cuda").manual_seed(seed + new_seed_group_index + i) for i in range(batch_size**2)]

        if status_queue:
            status_queue.put("Generating images ...")

        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            load_in_8bit=True,
            device_map="auto",
            local_files_only=local_files_only_value
        )
        pipe = PixArtSigmaPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            transformer=None,
            device_map="balanced",
            local_files_only=local_files_only_value
        )

        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = \
                pipe.encode_prompt(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=batch_size)

        del text_encoder
        del pipe
        flush()

        pipe = PixArtSigmaPipeline.from_pretrained(
            model_id,
            text_encoder=None,
            torch_dtype=torch.float16,
            local_files_only=local_files_only_value
        )

        if model_id == DEFAULT_MODEL_ID and checkpoint and os.path.exists(checkpoint):
            pipe.transformer = update_pixart_sigma_model_with_custom_model(pipe.transformer, checkpoint)
        else:
            checkpoint = "None"

        # Compile
        # Disabling torch.compile for now as compiling takes significant amount of time
        # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
        pipe.to("cuda")

        latents = pipe(
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_images_per_prompt=batch_size,
            height=height,
            width=width,
            num_inference_steps=steps,  # default 20
            guidance_scale=guidance_scale,  # default 4.5
            generator=random_number_generator,
            output_type="latent",
        ).images

        del pipe.transformer
        flush()

        image_list = []
        for _ in range(batch_size):
            with torch.no_grad():
                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pil")[0]
            image_list.append(image)
        images = image_list
        del pipe
        flush()

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
                "generator_model_type": "Pixart Sigma"
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
    gc.collect()
    # end batch

    end_time = time.perf_counter()
    logger.info(f"Completed. Time elapsed: {end_time - start_time}")

    if status_queue:
        status_queue.put(f"Completed. Time elapsed: {end_time - start_time:0.1f} seconds")
