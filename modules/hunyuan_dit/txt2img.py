
"""
Hunyuan-DiT txt2img

Copyright (c) 2024 Hideyuki Inada

Code to interact with Hunyuan-Dit is based on PixArt-Sigma interaction code [1] and [2] as well as Hunyuan documentation [3] below and is covered under Apache 2.0 license.
Refer to the license document at the root of this project.
In addition, portion of the code was taken from [4].

References
[1] https://huggingface.co/docs/diffusers/v0.29.2/api/pipelines/pixart_sigma
[2] https://github.com/huggingface/diffusers/blob/v0.29.2/docs/source/en/api/pipelines/pixart_sigma.md
[3] https://huggingface.co/docs/diffusers/en/api/pipelines/hunyuandit#diffusers.HunyuanDiTPipeline
[4] https://gist.github.com/sayakpaul/3154605f6af05b98a41081aaba5ca43e
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
from diffusers import HunyuanDiTPipeline
from PIL.PngImagePlugin import PngInfo
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
sys.path = [MODULE_ROOT] + sys.path
from cremage.ui.update_image_handler import update_image
from cremage.configs.preferences import load_user_config
from cremage.const.const import GMT_HUNYUAN_DIT

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def generate(
        model_id="Tencent-Hunyuan/HunyuanDiT-Diffusers",
        checkpoint:str=None,  # Not used for now
        out_dir:str=None,
        positive_prompt: str=None,
        negative_prompt: str=None,
        steps: int = 50,
        guidance_scale: float = 5.0,
        height: int = 1024,
        width: int = 1024,
        number_of_batches = 1,
        batch_size = 1,
        ui_thread_instance=None,
        seed=-1,
        auto_face_fix=False,
        safety_check=True,
        watermark=False,
        status_queue=None):
    """
    Generates an image based on the provided prompts, steps, and guidance scale.
    """

    if seed == -1:
        seed = random.getrandbits(32)

    if status_queue:
        status_queue.put("Diffusers pipeline created")

    file_number_base = len(os.listdir(out_dir))

    if batch_size != 1:
        logger.warn("Currently only 1 is supported for the batch size. Number of batches were adjusted to generate the specified number of images.")
        number_of_batches *= batch_size
        batch_size = 1

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

        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            load_in_8bit=True,
            device_map="auto",
            local_files_only=local_files_only_value
        )
        pipe = HunyuanDiTPipeline.from_pretrained(
            model_id,
            text_encoder_2=text_encoder,
            transformer=None,
            vae=None,
            device_map="balanced",
            local_files_only=local_files_only_value
        )

        with torch.no_grad():
            prompt_embeds, negative_embeds, prompt_attention_mask, negative_prompt_attention_mask = \
                pipe.encode_prompt(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=batch_size)

            prompt_embeds_2, negative_prompt_embeds_2, prompt_attention_mask_2, negative_prompt_attention_mask_2 = \
                pipe.encode_prompt(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    prompt_attention_mask=None,
                    negative_prompt_attention_mask=None,
                    max_sequence_length=256,
                    text_encoder_index=1,
                )
        del text_encoder
        del pipe
        flush()

        pipe = HunyuanDiTPipeline.from_pretrained(
            model_id,
            text_encoder=None,
            text_encoder_2=None,
            vae=None,
            torch_dtype=torch.float16,
            local_files_only=local_files_only_value
        )
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
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=random_number_generator,
            output_type="latent",
            prompt_embeds_2=prompt_embeds_2,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            prompt_attention_mask_2=prompt_attention_mask_2,
            negative_prompt_attention_mask_2=negative_prompt_attention_mask_2,
        ).images

        del pipe.transformer
        flush()

        # Instantiate VAE
        pipe = HunyuanDiTPipeline.from_pretrained(
            model_id,
            text_encoder=None,
            text_encoder_2=None,
            transformer=None,
            torch_dtype=torch.float16,
            local_files_only=local_files_only_value
        )
        pipe.to("cuda")

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
                face_fixer = FaceFixer(
                    preferences=app,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    procedural=True,
                    status_queue=status_queue)
                
                image = face_fixer.fix_with_insight_face(image)

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
                "generator_model_type": GMT_HUNYUAN_DIT
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
