
"""
Copyright (c) 2024 Hideyuki Inada

# Credit
* Code to interact with the Flux model is based on a github user's code. Link to the code is available below:
  https://www.reddit.com/r/StableDiffusion/comments/1ehl4as/how_to_run_flux_8bit_quantized_locally_on_your_16/

* Further memory footprint reduction idea is from [3]

# References
[1] black-forest-labs/FLUX.1-schnell. https://huggingface.co/black-forest-labs/FLUX.1-schnell
[2] Sayak Paul, David Corvoysier. Memory-efficient Diffusion Transformers with Quanto and Diffusers. https://huggingface.co/blog/quanto-diffusers
[3] https://huggingface.co/black-forest-labs/FLUX.1-schnell/discussions/5
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
import torch
import optimum.quanto
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
from PIL.PngImagePlugin import PngInfo
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
sys.path = [MODULE_ROOT] + sys.path
from cremage.ui.update_image_handler import update_image
from cremage.configs.preferences import load_user_config
from cremage.utils.random_utils import safe_random_int

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_REVISION = "refs/pr/1"
TEXT_MODEL_ID = "openai/clip-vit-large-patch14"
MODEL_DATA_TYPE = torch.bfloat16


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def quantize_and_freeze(model:torch.nn.Module,
                        weights:optimum.quanto.tensor=qfloat8) -> torch.nn.Module:
    """
    Quantizes and freezes the model to reduce memory footprint.

    Args:
        model (torch.nn.Module): Model to quantize and freeze.
        weights (optimum.quanto.tensor.qtype, optional): Target data type. Defaults to quanto_tensor.qfloat8.

    Returns:
        torch.nn.Module: The quantized and frozen model.
    """
    quantize(model, weights)
    freeze(model)
    return model


def generate(
        checkpoint:str=None,  # Not used for now
        out_dir:str=None,
        positive_prompt: str=None,
        negative_prompt: str=None,
        steps: int = 4,
        guidance_scale: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        low_mem=True,
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

    if seed == -1:
        seed = safe_random_int()

    pipe = None

    # Initialize required models
    # Text encoders
    # 1
    logger.info("Instantiating CLIP text tokenizer and model")
    tokenizer = CLIPTokenizer.from_pretrained(
        TEXT_MODEL_ID , torch_dtype=MODEL_DATA_TYPE)
    text_encoder = CLIPTextModel.from_pretrained(
        TEXT_MODEL_ID , torch_dtype=MODEL_DATA_TYPE)

    # 2
    logger.info("Instantiating T5 text tokenizer and model")
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        MODEL_ID, subfolder="tokenizer_2", torch_dtype=MODEL_DATA_TYPE,
        revision=MODEL_REVISION)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder_2", torch_dtype=MODEL_DATA_TYPE,
        revision=MODEL_REVISION)

    # Transformers
    logger.info("Instantiating scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler", revision=MODEL_REVISION)
    logger.info("Instantiating transformer")
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=MODEL_DATA_TYPE,
        revision=MODEL_REVISION)

    # VAE
    logger.info("Instantiating VAE")
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=MODEL_DATA_TYPE,
        revision=MODEL_REVISION)

    if low_mem is False:
        logger.info("Quantizing T5 to 8 bits")
        quantize_and_freeze(text_encoder_2)

        logger.info("Quantizing transformer to 8 bits")
        quantize_and_freeze(transformer)

    # Create a pipeline without T5 and transformer
    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None
    )

    # Set to the quantized version
    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer = transformer

    if low_mem:
        logging.info("Using low memory mode")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.enable_sequential_cpu_offload()
    else:
        logging.info("Using standard memory mode")
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
        random_number_generator = [torch.Generator(device=os.environ.get("GPU_DEVICE", "cpu")).manual_seed(seed + new_seed_group_index + i) for i in range(batch_size)]

        if status_queue:
            status_queue.put("Generating images ...")

        images = pipe(
            prompt=positive_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
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
                "sampling_iterations": steps,
                "image_height": height,
                "image_width": width,
                "cfg": guidance_scale,
                "seed": seed + new_seed_group_index + i,
                "safety_check": safety_check,
                "auto_face_fix": auto_face_fix,
                "generator_model_type": "Flux-1.schnell"
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
    if status_queue:
        status_queue.put("Completed")


if __name__ == "__main__":
     generate(
        checkpoint=None,  # Not used for now
        out_dir="/var/tmp/flux",  # FIXME
        positive_prompt="puppy",
        negative_prompt="low quality",
        steps = 20,
        guidance_scale = 4.0,
        height = 1024,
        width = 1024,
        number_of_batches = 1,
        batch_size = 1,
        ui_thread_instance=None,
        seed=-1,
        auto_face_fix=False,
        auto_face_fix_strength=0.3,
        auto_face_fix_face_detection_method="OpenCV",
        auto_face_fix_prompt="",
        safety_check=False,
        watermark=False,
        status_queue=None)
