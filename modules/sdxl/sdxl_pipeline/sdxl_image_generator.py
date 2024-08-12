"""
Cremage note:
Code refactored from sampling.py.
Sampling via options.
"""
import os
import sys
import gc
import json
import logging
import random
import time
import copy  # to deepcopy an object

from tqdm import tqdm, trange
import numpy as np
import torch
from pytorch_lightning import seed_everything
from PIL import Image
import cv2

SDXL_MODULE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.realpath(os.path.join(SDXL_MODULE_ROOT, "configs"))
CHECKPOINTS_DIR = "/media/pup/ssd2/recoverable_data/sd_models/Stable-diffusion"  # FIXME
VAE_CHECKPOINTS_DIR = "/media/pup/ssd2/recoverable_data/sd_models/VAE_sdxl"  # FIXME
PROJECT_ROOT = os.path.realpath(os.path.join(SDXL_MODULE_ROOT, "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
sys.path = [SDXL_MODULE_ROOT, MODULE_ROOT] + sys.path

from sdxl_pipeline.vram_mode import set_lowvram_mode
from sdxl_pipeline.sdxl_image_generator_utils import *
from sdxl_pipeline.options import parse_options
from cremage.utils.image_utils import save_torch_tensor_as_image_with_watermark
from cremage.utils.image_utils import tensor_to_pil_image, pil_image_to_tensor
from cremage.ui.update_image_handler import update_image
from cremage.utils.generation_status_updater import StatusUpdater
from cremage.utils.hires_fix_upscaler_utils import hires_fix_upscaler_name_list
from cremage.utils.ml_utils import scale_pytorch_images
from cremage.utils.cuda_utils import gpu_memory_info
from cremage.configs.preferences import load_user_config
from cremage.utils.wildcards import resolve_wildcards
from cremage.utils.misc_utils import strip_directory_from_path_list_str
from cremage.utils.random_utils import safe_random_int

# SD_XL_BASE_RATIOS = {
#     "0.5": (704, 1408),
#     "0.52": (704, 1344),
#     "0.57": (768, 1344),
#     "0.6": (768, 1280),
#     "0.68": (832, 1216),
#     "0.72": (832, 1152),
#     "0.78": (896, 1152),
#     "0.82": (896, 1088),
#     "0.88": (960, 1088),
#     "0.94": (960, 1024),
#     "1.0": (1024, 1024),
#     "1.07": (1024, 960),
#     "1.13": (1088, 960),
#     "1.21": (1088, 896),
#     "1.29": (1152, 896),
#     "1.38": (1152, 832),
#     "1.46": (1216, 832),
#     "1.67": (1280, 768),
#     "1.75": (1344, 768),
#     "1.91": (1344, 704),
#     "2.0": (1408, 704),
#     "2.09": (1472, 704),
#     "2.4": (1536, 640),
#     "2.5": (1600, 640),
#     "2.89": (1664, 576),
#     "3.0": (1728, 576),
# }

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": f"{CONFIG_DIR}/inference/sd_xl_base.yaml",
        #"ckpt": f"{CHECKPOINTS_DIR}/sd_xl_base_1.0.safetensors",
        "ckpt": f"{CHECKPOINTS_DIR}/juggernautXL_juggernautX.safetensors"
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": f"{CONFIG_DIR}/inference/sd_xl_refiner.yaml",
        "ckpt": f"{CHECKPOINTS_DIR}/sd_xl_refiner_1.0.safetensors",
        # below throws an error, so refiner is not a regular sdxl model
        # "ckpt": f"{CHECKPOINTS_DIR}/juggernautXL_juggernautX.safetensors"
    },
}


def load_img(image_path, key=None, device=os.environ.get("GPU_DEVICE", "cpu")):
    """
    Loads image from the path.

    Returns:
      Float32 PyTorch tensor in b, c, h, w format. Values are in [-1, 1].
    """
    image = Image.open(image_path)

    if image is None:
        return None

    w, h = image.size
    logger.debug(f"loaded input image of size ({w}, {h})")
    width, height = map(
        lambda x: x - x % 64, (w, h)
    )  # resize to integer multiple of 64
    image = image.resize((width, height))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image.to(device)


def run_txt2img(
    positive_prompt,
    negative_prompt,
    state,
    version,
    version_dict,
    opt=None,
    is_legacy=False,
    return_latents=False,
    safety_filter=None,
    stage2strength=None,
):
    """
    Generates images from the input conditions (e.g. prompts).

    Returns
        images (Tensor) : b, c, h, n in [0, 1]
    """
    logger.debug("Starting run_txt2img: GPU:. GPU: " + gpu_memory_info())

    W, H = opt.W, opt.H # FIXME st.selectbox("Resolution:", list(SD_XL_BASE_RATIOS.values()), 10)

    C = version_dict["C"]
    F = version_dict["f"]

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
        init_dict,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
    )

    sampler = init_sampling(
        opt=opt,
        stage2strength=stage2strength)
    num_samples = opt.n_samples

    if opt.hires_fix_upscaler and opt.hires_fix_upscaler.lower() == "latent":
        return_latents = True

    out = do_sample(
        state["model"],
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        opt=opt,
        # Cremage change
        # force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        force_uc_zero_embeddings=[],  # Cremage: Do not use zero embedding for uc
        return_latents=return_latents,
        safety_filter=safety_filter,
    )
    hires_fix_upscale_factor = opt.hires_fix_scale_factor
    
    if opt.hires_fix_upscaler and opt.hires_fix_upscaler != "None":
        if isinstance(out, (tuple, list)):
            samples, samples_z = out
        else:
            samples = out
            samples_z = None

        out_list = list()
        for i, sample in enumerate(samples):
            if opt.hires_fix_upscaler.lower() == "latent":
                skip_encode=True  # skip VAE
                input_image = torch.nn.functional.interpolate(samples_z, scale_factor=hires_fix_upscale_factor, mode='bilinear', align_corners=False)
                input_image = torch.unsqueeze(input_image[i], dim=0)
            else:
                skip_encode=False
                # image is c, h, w and [0, 1] at this point
                sample_with_batch_axis = torch.unsqueeze(sample, dim=0)   # to b, c, h, w
                scaled_samples = scale_pytorch_images(  # This takes b, c, h, w
                        sample_with_batch_axis,
                        width=int(opt.W * hires_fix_upscale_factor),
                        height=int(opt.H * hires_fix_upscale_factor),
                        interpolation=cv2.INTER_LANCZOS4)
                # No need as the image is already in [0, 1]
                # scaled_samples = scaled_samples * 2.0 - 1.0  # [0, 1] to [0, 2] to [-1, -1]
                input_image = scaled_samples

            opt2 = copy.deepcopy(opt)
            opt2.n_samples = 1
            out_single = run_img2img(
                positive_prompt,
                negative_prompt,
                image_path=None,
                state=state,
                version_dict=version_dict,
                opt=opt2,  # denoising strength is set here (opt.strength)
                is_legacy=is_legacy,
                return_latents=False,
                safety_filter=safety_filter,
                stage2strength=opt2.refiner_strength,
                skip_encode=skip_encode,
                input_image=input_image)
            out_list.append(out_single)
        out = torch.concat(out_list)
    return out


def run_img2img(
    positive_prompt,
    negative_prompt,
    image_path,
    state,
    version_dict,
    opt=None,
    is_legacy=False,
    return_latents=False,
    safety_filter=None,
    stage2strength=None,
    skip_encode=False,  # Skip VAE
    input_image=None
):
    """
    
    Args:
        skip_encode (bool): If True, do not use VAE to encode. This assumes
          that the image is already in the latent space.
          If False, use VAE to encode.
        input_image (Tensor): Images in the b, c, h, w format.
          Values are in [0, 1]. Internally, this gets converted to [-1, 1].
    """
    if input_image is None:
        img = load_img(image_path)  # b, c, h, w in [-1, 1]
    else:
        if skip_encode is False:  # pixel space hires fix
            img = input_image * 2 - 1.0  # [0, 1] to [0, 2] to [-1, 1]
        else:  # latent hires fix
            img = input_image
    if img is None:
        return None
    H, W = img.shape[2], img.shape[3]

    # FIXME. Make the resolution behavior the same as SD 1.5.
    opt.H = H
    opt.W = W
    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
        init_dict,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
    )
    strength = opt.strength  # Denoising strengh e.g. 0.75 [0, 1]
    sampler = init_sampling(
        opt=opt,
        img2img_strength=strength,
        stage2strength=stage2strength,
    )
    num_samples = opt.n_samples

    out = do_img2img(
        repeat(img, "1 ... -> n ...", n=num_samples),
        state["model"],
        sampler,
        value_dict,
        num_samples,
        opt=opt,
        # Cremage change
        # force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        force_uc_zero_embeddings=[],
        return_latents=return_latents,
        safety_filter=safety_filter,
        skip_encode=skip_encode
    )
    return out


def apply_refiner(
    input,
    state,
    sampler,
    num_samples,
    prompt,
    negative_prompt,
    opt=None,
    safety_filter=None,
    finish_denoising=False,
):
    init_dict = {
        "orig_width": input.shape[3] * 8,
        "orig_height": input.shape[2] * 8,
        "target_width": input.shape[3] * 8,
        "target_height": input.shape[2] * 8,
    }

    value_dict = init_dict
    value_dict["prompt"] = prompt
    value_dict["negative_prompt"] = negative_prompt

    value_dict["crop_coords_top"] = 0
    value_dict["crop_coords_left"] = 0

    value_dict["aesthetic_score"] = 6.0
    value_dict["negative_aesthetic_score"] = 2.5

    logger.debug(f"refiner input shape: {input.shape}")
    samples = do_img2img(
        input,
        state["model"],
        sampler,
        value_dict,
        num_samples,
        opt=opt,
        skip_encode=True,
        safety_filter=safety_filter,
        add_noise=not finish_denoising,
    )

    return samples

persistent_state = None
prev_ckpt = None
prev_lora_models = None
prev_lora_weights = None
persistent_state_2 = None
prev_ckpt_2 = None
prev_lora_models_2 = None
prev_lora_weights_2 = None

def generate(options=None,
             generation_type="txt2img",
             ui_thread_instance=None,
             status_queue=None):
    global persistent_state
    global prev_ckpt
    global prev_lora_models
    global prev_lora_weights
    global persistent_state_2
    global prev_ckpt_2
    global prev_lora_models_2
    global prev_lora_weights_2

    start_time = time.perf_counter()
    opt = options
    use_refiner = False

    # Hires fix
    hires_fix_upscale_factor = opt.hires_fix_scale_factor
    if opt.hires_fix_upscaler and \
       opt.hires_fix_upscaler.lower() != "none" and \
       opt.hires_fix_upscaler in hires_fix_upscaler_name_list:
        use_hires_fix = True
    else:
        use_hires_fix = False

    # Parse some parameters
    positive_prompt = opt.prompt
    negative_prompt = opt.negative_prompt
    # Make a copy of prompts. This is needed as the original
    # contains unresolved wildcards which we
    # need to use for each batch
    positive_prompt_original = positive_prompt
    negative_prompt_original = negative_prompt

    base_count = len(os.listdir(opt.outdir))
    wm_encoder = None
    if opt.watermark:
        from imwatermark import WatermarkEncoder
        wm = "Cremage"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    version = "SDXL-base-1.0"
    version_dict = VERSION2SPECS[version]
    version_dict["ckpt"] = opt.ckpt
    logger.debug(f"Check point is set to {opt.ckpt}")

    mode = generation_type

    set_lowvram_mode(True)

    # Initialize status updater object. This object contains a method
    # which is invoked as a callback for each sampling step.
    status_update_func = None
    if status_queue:
        steps_to_track = opt.sampling_steps
        
        status_updater = StatusUpdater(steps_to_track, int(opt.n_iter), status_queue)
        status_update_func = status_updater.status_update

    if opt.refiner_strength == 0:
        logger.debug("Refiner is turned off for this generation")
        add_pipeline = False
    else:
        add_pipeline = True

    # Seed
    if opt.seed == -1:
        seed = safe_random_int()
    else:
        seed = opt.seed
    seed_everything(seed)
    
    # When LoRA model changes, a model has to be instantiated
    # as weights will be different.
    if persistent_state is None or prev_lora_models != opt.lora_models or \
        prev_lora_weights != opt.lora_weights:

        if prev_lora_models != opt.lora_models or \
            prev_lora_weights != opt.lora_weights:
            logger.debug("Reinit for LoRA change")       
            if status_queue:
                status_queue.put("Initializing for LoRA")
        else:
            logger.debug("Reinitializing the state.")
            if status_queue:
                status_queue.put("Initializing ...")

        if opt.safety_check:
            logger.debug("Enabling safety filter")
        else:
            logger.debug("Disabling safety filter")

        state = init_st(version_dict,
                        lora_models=opt.lora_models,
                        lora_weights=opt.lora_weights)
        if status_queue:
            status_queue.put("Model instantiated")
        persistent_state = state

    else:
        logger.debug("Reusing already initialized state")
        state = persistent_state

    if prev_ckpt != opt.ckpt or prev_lora_models != opt.lora_models or \
        prev_lora_weights != opt.lora_weights:
        # Load state dict
        state["model"] = load_state_dict_into_model(
            state, 
            opt.ckpt,
            vae_ckpt=opt.vae_ckpt,
            verbose=True)
        prev_ckpt = opt.ckpt
        prev_lora_models = opt.lora_models
        prev_lora_weights = opt.lora_weights
        if status_queue:
            status_queue.put("Weights loaded")

    model = state["model"]

    if opt.safety_check:
        # state["filter"] = DeepFloydDataFiltering(verbose=False)
        from safety.safety_filter import SafetyFilter
        state["filter"] = SafetyFilter()
    else:
        if "filter" in state:
            del state["filter"]
        assert "filter" not in state

    is_legacy = version_dict["is_legacy"]

    stage2strength = None
    finish_denoising = False

    if add_pipeline:  # Set up refiner

        if os.path.basename(opt.refiner_sdxl_ckpt).startswith("sd_xl_refiner"):
            refiner = True
            version2 = "SDXL-refiner-1.0"  # "SDXL-refiner-1.0" or "SDXL-refiner-0.9"
        else:  # Use a regular SDXL model as a refiner
            refiner = False
            version2 = "SDXL-base-1.0"

        logger.warning(
            f"Running with {version2} as the second stage model. Make sure to provide (V)RAM :) "
        )
        logger.debug("**Refiner Options:**")

        version_dict2 = VERSION2SPECS[version2]

        if persistent_state_2 is None or \
            prev_ckpt_2 != opt.refiner_sdxl_ckpt or \
            prev_lora_models_2 != opt.refiner_sdxl_lora_models or \
            prev_lora_weights_2 != opt.refiner_sdxl_lora_weights:

            if prev_lora_models_2 != opt.refiner_sdxl_lora_models or \
                prev_lora_weights_2 != opt.refiner_sdxl_lora_weights:
                logger.debug("Reinit for refiner LoRA change")       
                if status_queue:
                    status_queue.put("Initializing refiner for LoRA")
            else:
                logger.debug("Reinitializing the refiner state.")
                if status_queue:
                    status_queue.put("Initializing refiner ...")

            state2 = init_st(version_dict2,
                            lora_models=opt.refiner_sdxl_lora_models,
                            lora_weights=opt.refiner_sdxl_lora_weights)
            persistent_state_2 = state2
        else:
            logger.debug("Reusing already refiner initialized state")
            state2 = persistent_state_2

        logger.debug(state2["msg"])

        if prev_ckpt_2 != opt.refiner_sdxl_ckpt or \
            prev_lora_models_2 != opt.refiner_sdxl_lora_models or \
            prev_lora_weights_2 != opt.refiner_sdxl_lora_weights:
            # Load state dict
            state2["model"] = load_state_dict_into_model(
                state2, 
                opt.refiner_sdxl_ckpt,
                vae_ckpt=opt.refiner_sdxl_vae_ckpt,
                verbose=True,
                refiner=refiner)
            prev_ckpt_2 = opt.refiner_sdxl_ckpt
            prev_lora_models_2 = opt.refiner_sdxl_lora_models
            prev_lora_weights_2 = opt.refiner_sdxl_lora_weights
            if status_queue:
                status_queue.put("Refiner weights loaded")

            model2 = state2["model"]

        stage2strength = opt.refiner_strength  # 0.15 [0, 1]
        logger.debug(f"Refiner strength: {stage2strength}")

        # Note stage2strengh parameter is set to None which is the default
        sampler2 = init_sampling(
            opt=opt,
            key=2,
            img2img_strength=stage2strength
        )

        finish_denoising = True
        # Cremage note:
        # If finish_denoising is None or 0,
        # then set stage2strengh to None so that txt2img will do
        # full denoising.  Read this as left-over denoising.
        if not finish_denoising:
            stage2strength = None

    tic = time.time()
    for n in trange(opt.n_iter, desc="Number of batches"):
        if status_queue:
            status_queue.put(f"Generating images (Batch {n+1}/{opt.n_iter})")

        # Restore prompts with unresolved wildcards
        positive_prompt = positive_prompt_original
        negative_prompt = negative_prompt_original

        # Resolve wildcards
        positive_prompt = resolve_wildcards(positive_prompt, wildcards_dir=opt.wildcards_path)
        negative_prompt = resolve_wildcards(negative_prompt, wildcards_dir=opt.wildcards_path)

        if mode == "txt2img":
            out = run_txt2img(
                positive_prompt,
                negative_prompt,
                state,
                version,
                version_dict,
                opt=opt,
                is_legacy=is_legacy,
                return_latents=add_pipeline,
                safety_filter=state.get("filter"),
                stage2strength=stage2strength,
            )
        elif mode == "img2img":
            out = run_img2img(
                positive_prompt,
                negative_prompt,
                opt.init_img,
                state,
                version_dict,
                opt=opt,
                is_legacy=is_legacy,
                return_latents=add_pipeline,
                safety_filter=state.get("filter"),
                stage2strength=stage2strength,
            )
        else:
            raise ValueError(f"unknown mode {mode}")
        if isinstance(out, (tuple, list)):
            samples, samples_z = out
        else:
            samples = out
            samples_z = None

        if add_pipeline and samples_z is not None:
            use_refiner = True
            logger.debug("**Running Refinement Stage**")
            status_queue.put("Running refiner ...")
            samples = apply_refiner(
                samples_z,
                state2,
                sampler2,
                samples_z.shape[0],
                prompt=positive_prompt,
                negative_prompt=negative_prompt if is_legacy else "",
                opt=opt,
                safety_filter=state.get("filter"),
                finish_denoising=finish_denoising,
            )

        # Save
        for i, sample in enumerate(samples):

            # Extra processing start
            if opt.auto_face_fix:
                logger.debug("Applying face fix")
                status_queue.put("Applying face fix")

                if opt.auto_face_fix_prompt:
                    auto_face_fix_prompt = opt.auto_face_fix_prompt
                else:
                    auto_face_fix_prompt = positive_prompt
                pil_img, tensor_device = tensor_to_pil_image(sample)
                face_fix_options = {
                        "positive_prompt" : auto_face_fix_prompt,
                        "negative_prompt" : opt.negative_prompt,
                        "clip_skip": opt.clip_skip,
                        "denoising_strength": opt.auto_face_fix_strength,
                        "sampler": opt.face_fix_sampler,
                        "sampling_steps": opt.face_fix_sampling_steps,
                        "model_path": opt.face_fix_ckpt,
                        "vae_path": opt.face_fix_vae_ckpt,
                        "lora_models": opt.face_fix_lora_models,
                        "lora_weights": opt.face_fix_lora_weights,
                        "sampler": opt.face_fix_sampler,
                        "seed": "0",
                        "generator_model_type": opt.face_fix_generator_model_type
                        # "status_queue": status_queue
                }

                if opt.auto_face_fix_face_detection_method == "InsightFace":
                    from face_detection.face_detector_engine import fix_with_insight_face
                    pil_img = fix_with_insight_face(pil_img, **face_fix_options)
                elif opt.auto_face_fix_face_detection_method == "OpenCV":
                    from face_detection.face_detector_engine import fix_with_opencv
                    pil_img = fix_with_opencv(pil_img, **face_fix_options)
                else:
                    logger.info(f"Ignoring unsupported face detection method: {opt.auto_face_fix_face_detection_method}")

                sample = pil_image_to_tensor(pil_img,
                                             half=sample.dtype==torch.float16,
                                             device=tensor_device)

            generation_parameters = {
                "time": time.time(),
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "ldm_model": os.path.basename(opt.ckpt),
                "vae_model": os.path.basename(opt.vae_ckpt),
                "lora_models": strip_directory_from_path_list_str(opt.lora_models),
                "lora_weights": opt.lora_weights,
                "sampler": opt.sampler.replace("Sampler", ""),
                "sampling_iterations": opt.sampling_steps,
                "cfg": opt.scale,
                "image_height": opt.H,
                "image_width": opt.W,
                #  "clip_skip": opt.clip_skip,  # Cremage. Commenting out as CLIP skip for SDXL creates confusion
                "seed": seed + i,
                "watermark": opt.watermark,
                "safety_check": opt.safety_check
            }

            # Add face fix
            if opt.auto_face_fix:
                generation_parameters["auto_face_fix"] = True
                generation_parameters["auto_face_fix_strength"] = opt.auto_face_fix_strength
                generation_parameters["auto_face_fix_prompt"] = auto_face_fix_prompt
                generation_parameters["auto_face_fix_face_detection_method"] = opt.auto_face_fix_face_detection_method

            if use_hires_fix:
                generation_parameters["hires_fix_upscaler"] = opt.hires_fix_upscaler
                generation_parameters["hires_fix_scale_factor"] = opt.hires_fix_scale_factor
                generation_parameters["upscale_width"] = opt.W * hires_fix_upscale_factor
                generation_parameters["upscale_height"] = opt.H * hires_fix_upscale_factor

            if use_refiner:
                generation_parameters["refiner_ldm_model"] = os.path.basename(opt.refiner_sdxl_ckpt)
                generation_parameters["refiner_vae_model"] = os.path.basename(opt.refiner_sdxl_vae_ckpt)
                generation_parameters["refiner_lora_models"] = strip_directory_from_path_list_str(opt.refiner_sdxl_lora_models)
                generation_parameters["refiner_lora_weights"] = opt.refiner_sdxl_lora_weights
                generation_parameters["refiner_strength"] = opt.refiner_strength

            img2 = save_torch_tensor_as_image_with_watermark(
                opt,
                opt.outdir,
                sample,
                generation_parameters,
                file_number=base_count,
                wm_encoder=wm_encoder)

            str_generation_params = json.dumps(generation_parameters)
            # Pass img (PIL Image) to the main thread here!
            if ui_thread_instance:
                update_image(ui_thread_instance,
                                img2,
                                generation_parameters=str_generation_params)
                
            base_count += 1

        # end single batch
        gc.collect()
    # end batch

    end_time = time.perf_counter()
    logger.info(f"Completed. Time elapsed: {end_time - start_time}")

    if status_queue:
        status_queue.put(f"Completed. Time elapsed: {end_time - start_time:0.1f} seconds")
