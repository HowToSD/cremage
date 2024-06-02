# Cremage note: This is the engine file that I need to tweak.
# Refactored streamlit_helpers.py
import os
import sys
import logging
import copy
import math
import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Union
import contextlib
from contextlib import contextmanager, nullcontext

from imwatermark import WatermarkEncoder
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TT
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors

SDXL_MODULE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = os.path.realpath(os.path.join(SDXL_MODULE_ROOT, "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
sys.path = [SDXL_MODULE_ROOT, MODULE_ROOT] + sys.path

from cremage.utils.image_utils import save_torch_tensor_as_image
from cremage.utils.ml_utils import load_model as ml_utils_load_model
from cremage.utils.lora_loader import load_loras, load_loras_state_dict_into_custom_model_state_dict
from cremage.utils.cuda_utils import gpu_memory_info

from sdxl_pipeline.vram_mode import *
from scripts.demo.discretization import (
    Img2ImgDiscretizationWrapper,
    Txt2NoisyDiscretizationWrapper,
)
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.modules.diffusionmodules.guiders import (
    LinearPredictionGuider,
    TrianglePredictionGuider,
    VanillaCFG,
)
from sgm.modules.diffusionmodules.sampling import (
    DPMPP2MSampler,
    DPMPP2SAncestralSampler,
    EulerAncestralSampler,
    EulerEDMSampler,
    HeunEDMSampler,
    LinearMultistepSampler,
)
from sgm.util import append_dims, default, instantiate_from_config
from torch import autocast
from torchvision import transforms
from torchvision.utils import make_grid, save_image

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def init_st(version_dict, lora_models=None, lora_weights=None):
# def init_st(version_dict, load_ckpt=True, load_filter=True):
    """
    Instantiates the model, but does not load state dict.
    Caller should access the instantiated model via state["model"]
    """

    # load lora
    # You need to load lora weights first as that will
    # dictate the main model weight shapes
    if lora_models:
        loras, lora_ranks, lora_weights = load_loras(lora_models, lora_weights, model_type="SDXL")
    else:
        loras = []
        lora_ranks = []
        lora_weights = []
    # end load lora

    state = dict()
    if not "model" in state:
        config = version_dict["config"]
        ckpt = version_dict["ckpt"]

        config = OmegaConf.load(config)

        config["model"]["params"]["network_config"]["params"]["lora_ranks"] = lora_ranks
        config["model"]["params"]["network_config"]["params"]["lora_weights"] = lora_weights
        config["model"]["params"]["conditioner_config"]["params"]["lora_ranks"] = lora_ranks
        config["model"]["params"]["conditioner_config"]["params"]["lora_weights"] = lora_weights

        # Instantiate the main model (LDM) using the config
        # Note this contains VAE, CLIP and UNet
        # Any weight that may be loaded here will be overridden later.
        logger.debug("Instantiating the main model (LDM).")
        logger.debug(config["model"]["params"]["network_config"]["params"])
        import time
        t_start = time.perf_counter()
        model = instantiate_model_from_config(config)
        t_end = time.perf_counter()
        logger.debug(f"Base model instantiation took {t_end-t_start} seconds")
        state["msg"] = "Model instantiated (state dict is not loaded yet)"
        state["model"] = model
        state["ckpt"] = None
        state["config"] = config
        state["loras"] = loras  # actual sd
        state["lora_ranks"] = lora_ranks
        state["lora_weights"] = lora_weights

    return state


def instantiate_model_from_config(config, ckpt=None, verbose=True):
    """
    Instantiates a SDXL model from config.
    This does not load state dict, and that is done later.

    Args:
        config: The model configuration.

    Returns:
        The instantiated model instance.
    """
    model = instantiate_from_config(config.model)  # DiffusionEngine
    return model

def load_state_dict_into_model(state,
                               ckpt,
                               vae_ckpt:str=None,
                               verbose=True,
                               refiner=False):
    """
    Loads state dict from a checkpoint into the model instance.

    Args:
        ckpt (str): The full path of the main model.
        vae_ckpt (str): The full path of the VAE model, "None" or None.
    """
    model = state["model"]
    if ckpt is None or os.path.exists(ckpt) is False:
        raise ValueError(f"Invalid checkpoint: {ckpt}")

    logger.debug(f"Loading model from {ckpt}")
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            global_step = pl_sd["global_step"]
            logger.debug(f"loaded ckpt from global step {global_step}")
            logger.debug(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError

    # Cremage: Patch open_clip for MHA for LoRA
    # This is only applicable for open_clip model weight (conditioner.embedders.1)
    import re
    updates = list()
    updates_del = list()
    for k, v in sd.items():
        if refiner:  # Cremage note: Refiner uses open_clip as primary. See config
            match = re.search(r"conditioner.embedders.0.*attn.in_proj", k)
        else:
            match = re.search(r"conditioner.embedders.1.*attn.in_proj", k)
        if match:
            new_key = k.replace("attn.", "attn.multihead_attn.")
            updates.append((new_key, v.clone()))
            updates_del.append(k)

        if refiner:
            match = re.search(r"conditioner.embedders.0.*attn.out_proj", k)
        else:
            match = re.search(r"conditioner.embedders.1.*attn.out_proj", k)
        if match:
            new_key = k.replace("attn.", "attn.multihead_attn.")
            updates.append((new_key, v.clone()))
            updates_del.append(k)
    # Apply the updates
    for new_key, v in updates:
        sd[new_key] = v

    for old_key in updates_del:
        del sd[old_key]
    # end LoRA support

    # LoRA statedict loading start
    logger.debug("Loading lora into custom checkpoint's statedict")

    if state["loras"] and len(state["loras"]) > 0:
        sd = load_loras_state_dict_into_custom_model_state_dict(
            state["loras"], # list of sds for LoRA
            sd # custom sd (e.g. juggernaut)
        )
        # with open("tmp_lora_weight_name.txt", "w") as f:
        #     model_sd = model.state_dict()
        #     for k, v in model_sd.items():
        #         if "lora" in k:
        #             f.write(f"{k}\n")

    # LoRA statedict loading end

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logger.debug("Missing keys:")
        logger.debug(m)
    if len(u) > 0 and verbose:
        logger.debug("Unexpected keys:")
        logger.debug(u)

    if vae_ckpt and vae_ckpt != None:
        if os.path.exists(vae_ckpt) is False:
            logger.debug(f"{vae_ckpt} is not found. Ignoring.")
        else:
            vae_model = ml_utils_load_model(vae_ckpt)
            if isinstance(sd, dict) is False:
                logger.warning("Invalid vae model format.")
            else:
                for unwanted_key in ["model_ema.decay", "model_ema.num_updates"]:
                    if unwanted_key in vae_model:
                        del vae_model[unwanted_key]
                        logger.debug(f"Removed {unwanted_key} from the custom vae model before updating the main VAE")

                model.first_stage_model.load_state_dict(vae_model)
                logger.debug(f"Overrode VAE model parameters from the custom model {os.path.basename(vae_ckpt)}")

    model = initial_model_load(model)
    model.eval()
    return model


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    # Hardcoded demo settings; might undergo some changes in the future

    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = "A professional photograph of an astronaut riding a pig"
            if negative_prompt is None:
                negative_prompt = ""

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            orig_width = init_dict["orig_width"]  # 1024
            orig_height = init_dict["orig_height"]  # 1024
            
            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0
            
            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

        if key in ["fps_id", "fps"]:
            fps = 6  # FIXME
            
            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1

        if key == "motion_bucket_id":
            mb_id = 0  # FIXME st.number_input("motion bucket id", 0, 511, value=127)
            value_dict["motion_bucket_id"] = mb_id

        if key == "pool_image":
            image = load_img(
                key="pool_image_input",
                size=224,
                center_crop=True,
            )
            if image is None:
                logger.debug("Need an image here")
                image = torch.zeros(1, 3, 224, 224)
            value_dict["pool_image"] = image

    return value_dict


def get_guider(options, key, opt):
    """
    Cremage added Docstring.

    Args:
        opt_scale: CFG scale value specified in the option object.
    """
    guider = opt.guider  # Default is set to "VanillaCFG"
    # f"Discretization #{key}",
    # [
    #     "VanillaCFG",
    #     "IdentityGuider",
    #     "LinearPredictionGuider",
    #     "TrianglePredictionGuider",
    # ],

    additional_guider_kwargs = options.pop("additional_guider_kwargs", {})

    if guider == "IdentityGuider":
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        scale = opt.scale  # 5.0 
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale": scale,
                **additional_guider_kwargs,
            },
        }
    elif guider == "LinearPredictionGuider":
        max_scale = options.get("cfg", opt.linear_prediction_guider_max_scale)  # 1.5
        min_scale = opt.linear_prediction_guider_min_scale  # 1.0  [1.0, 10.0]
        
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.LinearPredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": options["num_frames"],
                **additional_guider_kwargs,
            },
        }
    elif guider == "TrianglePredictionGuider":
        max_scale = opt.triangle_prediction_guider_max_scale  # 2.5  # [1.0, 10.0]
        min_scale = opt.triangle_prediction_guider_min_scale  # 1.0  # [1.0, 10.0]
        
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.TrianglePredictionGuider",
            "params": {
                "max_scale": max_scale,
                "min_scale": min_scale,
                "num_frames": options["num_frames"],
                **additional_guider_kwargs,
            },
        }
    else:
        raise NotImplementedError
    return guider_config


def init_sampling(
    opt=None,
    key=1,
    img2img_strength: Optional[float] = None,
    # specify_num_samples: bool = True,
    stage2strength: Optional[float] = None,
    options: Optional[Dict[str, int]] = None,
):
    """
    Docstring added by Cremage:

    txt2img:
      img2img_strength: None
      stage2strength: stage2strengh variable value (opt.refiner_strength)

    img2img:
      img2img_strength: opt.denoising_strength
      stage2strength: stage2strengh variable value (opt.refiner_strength)

    refiner:
      img2img_strength: stage2strengh variable value (opt.refiner_strength)
      stage2strength: None
    
    stage2strength: How much to denoise. 0: No denoise, 1: Full denoise
    img2img_strengh: Add noise by img2img_strength
    """

    options = {} if options is None else options
    steps = opt.sampling_steps  # Original value 50
    sampler = opt.sampler # "EulerEDMSampler"

    # ["LegacyDDPMDiscretization", "EDMDiscretization"]
    discretization = opt.discretization  # "LegacyDDPMDiscretization"

    discretization_config = get_discretization(
        opt=opt,  # This is the user-configured option
        discretization=discretization,
        options=options,
        key=key)

    guider_config = get_guider(
        options=options,
        key=key,
        opt=opt)  # User configured generation option

    sampler = get_sampler(opt, sampler, steps, discretization_config, guider_config, key=key)
    if img2img_strength is not None:
        logger.warning(
            f"Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper"
        )
        sampler.discretization = Img2ImgDiscretizationWrapper(
            sampler.discretization, strength=img2img_strength
        )
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(
            sampler.discretization, strength=stage2strength, original_steps=steps
        )
    return sampler


def get_discretization(opt=None,
                       discretization="LegacyDDPMDiscretization",
                       options=None,
                       key=1):
    """
    Cremage added docstring:

    Returns:
        Configuration for instantiating the discretization (noise scheduler) class.
        Note that there are two noise schedulers:
          * Conventional (Legacy DDPM)
          * Karras (aka EDM)
    """
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        sigma_min = opt.discretization_sigma_min  # 0.0292 options.get("sigma_min", 0.03)
        sigma_max = opt.discretization_sigma_max  # 14.6146 options.get("sigma_max", 14.61)
        rho = opt.discretization_rho  # 3.0  # options.get("rho", 3.0)
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }

    return discretization_config

def get_sampler(opt, sampler_name, steps, discretization_config, guider_config, key=1):
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        s_churn = opt.sampler_s_churn  # 0.0 # st.sidebar.number_input(f"s_churn #{key}", value=0.0, min_value=0.0)
        s_tmin = opt.sampler_s_tmin  # 0.0 # st.sidebar.number_input(f"s_tmin #{key}", value=0.0, min_value=0.0)
        s_tmax = opt.sampler_s_tmax  # 999.0 # st.sidebar.number_input(f"s_tmax #{key}", value=999.0, min_value=0.0)
        s_noise =opt.sampler_s_noise  # 1.0 # st.sidebar.number_input(f"s_noise #{key}", value=1.0, min_value=0.0)

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
    elif (
        sampler_name == "EulerAncestralSampler"
        or sampler_name == "DPMPP2SAncestralSampler"
    ):
        s_noise = opt.sampler_s_noise  # 1.0 # st.sidebar.number_input("s_noise", value=1.0, min_value=0.0)
        eta = opt.sampler_eta  # 1.0 # st.sidebar.number_input("eta", value=1.0, min_value=0.0)

        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":
        order = opt.sampler_order  # 4  # FIXME st.sidebar.number_input("order", value=4, min_value=1)
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler


def load_img(
    image,
    display: bool = True,
    size: Union[None, int, Tuple[int, int]] = None,
    center_crop: bool = False,
):

    if image is None:
        return None
    w, h = image.size
    logger.debug(f"loaded input image of size ({w}, {h})")

    transform = []
    if size is not None:
        transform.append(transforms.Resize(size))
    if center_crop:
        transform.append(transforms.CenterCrop(size))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Lambda(lambda x: 2.0 * x - 1.0))

    transform = transforms.Compose(transform)
    img = transform(image)[None, ...]
    logger.debug(f"input min/max/mean: {img.min():.3f}/{img.max():.3f}/{img.mean():.3f}")
    return img


def get_init_img(batch_size=1, key=None):
    init_image = load_img(key=key).to(os.environ.get("GPU_DEVICE", "cpu"))
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    return init_image


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    opt=None,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    batch2model_input: List = None,
    return_latents=False,
    safety_filter=None,
    T=None,
    additional_batch_uc_fields=None,
    decoding_t=None,
):
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])  # Cremage: ['txt']
    batch2model_input = default(batch2model_input, [])  # []
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])  # []

    logger.debug("Sampling")

    if torch.cuda.is_available() and opt.precision=="autocast":
        precision_context = torch.autocast(device_type=os.environ.get("GPU_DEVICE", "cpu"))
    else:
        precision_context = contextlib.nullcontext()

    # precision_scope = autocast
    with torch.no_grad():
        with precision_context:
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]  # Code path

                logger.debug("Loading CLIP. GPU: " + gpu_memory_info())
                load_model(model.conditioner)  # Move clip to CUDA
                logger.debug("Loaded CLIP. GPU: " + gpu_memory_info())

                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                    device=os.environ.get("GPU_DEVICE", "cpu")
                )
                # batch["txt"] contains ["cute puppy", "cute puppy", "cute puppy"]
                c, uc = model.conditioner.get_unconditional_conditioning(  # marker
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                    embedding_dir=opt.embedding_path if opt is not None else None
                )
                # c["crossattn"].shape = (3, 77, 2048) Note 2048 instead of 768 for SD 1.5
                # c["vector"].shape = (3, 2816)

                # Create filler
                filler_value_dict = value_dict.copy()
                filler_value_dict["prompt"] = ""
                filler_value_dict["negative_prompt"] = ""
                filler_batch, filler_batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    filler_value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                    device=os.environ.get("GPU_DEVICE", "cpu")
                )

                # filler = model.get_learned_conditioning(batch_size * [""], embedding_dir=opt.embedding_path, clip_skip=opt.clip_skip)
                filler, filler_uc = model.conditioner.get_unconditional_conditioning(
                    filler_batch,
                    batch_uc=filler_batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                # Calculate the difference in size along axis 1
                diff = uc["crossattn"].shape[1] - c["crossattn"].shape[1]
                assert diff % 77 == 0
                diff = diff // 77
                if diff < 0:  # uc is smaller
                    diff = -diff
                    # Repeat the filler tensor the required number of times and concatenate it with uc
                    repeated_filler = filler["crossattn"].repeat(1, diff, 1)  # The 'diff' times repetition is along the second axis
                    uc["crossattn"] = torch.cat((uc["crossattn"], repeated_filler), axis=1)
                else:  # c is smaller or they are equal
                    repeated_filler = filler["crossattn"].repeat(1, diff, 1)  # Repeat along the second axis
                    c["crossattn"] = torch.cat((c["crossattn"], repeated_filler), axis=1)
                assert uc["crossattn"].shape[1] == c["crossattn"].shape[1]
                # end filler

                logger.debug("Unloading CLIP. GPU: " + gpu_memory_info())
                unload_model(model.conditioner)  # unload clip
                logger.debug("Unloaded CLIP. GPU: " + gpu_memory_info())

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to(os.environ.get("GPU_DEVICE", "cpu")), (c, uc)
                        )
                    if k in ["crossattn", "concat"] and T is not None:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=T)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=T)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=T)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=T)

                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert T is not None

                        if isinstance(
                            sampler.guider,
                            (
                                VanillaCFG,
                                LinearPredictionGuider,
                                TrianglePredictionGuider,
                            ),
                        ):
                            additional_model_inputs[k] = torch.zeros(
                                num_samples[0] * 2, num_samples[1]
                            ).to(os.environ.get("GPU_DEVICE", "cpu"))
                        else:
                            additional_model_inputs[k] = torch.zeros(num_samples).to(
                                os.environ.get("GPU_DEVICE", "cpu")
                            )
                    else:
                        additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to(os.environ.get("GPU_DEVICE", "cpu"))

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                logger.debug("Loading denoiser. GPU: " + gpu_memory_info())
                load_model(model.denoiser)
                load_model(model.model)
                logger.debug("Loaded denoiser. GPU: " + gpu_memory_info())
                
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)

                logger.debug("Unloading denoiser")
                unload_model(model.model)
                unload_model(model.denoiser)
                logger.debug("Unloaded denoiser")

                logger.debug("Loading VAE. GPU: " + gpu_memory_info())
                load_model(model.first_stage_model)
                logger.debug("Loaded VAE. GPU: " + gpu_memory_info())
                
                model.en_and_decode_n_samples_a_time = (
                    decoding_t  # Decode n frames at a time
                )

                # FIXME. Cremage: Make low_mem configurable
                # This is needed to avoid OOM for a large batch size
                # e.g. on 4090, it can handle bs=4, but bs=8
                # throws OOM in VAE.
                # The idea is spoon-feed VAE.
                low_mem = True
                if low_mem:
                    samples_list = list()
                    for z in samples_z:
                        z = torch.unsqueeze(z, dim=0)
                        single_x = model.decode_first_stage(z)
                        samples_list.append(single_x)
                    samples_x = torch.concat(samples_list)
                else:
                    samples_x = model.decode_first_stage(samples_z)

                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                logger.debug("Unloading VAE. GPU: " + gpu_memory_info())
                unload_model(model.first_stage_model)
                logger.debug("Unloaded VAE. GPU: " + gpu_memory_info())

                if safety_filter is not None:
                    # Input is torch tensor in b, c, h, w format with values in [0, 1]
                    # samples = safety_filter(samples)
                    samples, _ = safety_filter(torch_images=samples)

                # if T is None:
                #     grid = torch.stack([samples])
                #     grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                #     # outputs.image(grid.cpu().numpy())
                #     # FIXME
                #     enable_this = False
                #     if enable_this:  # CREMAGE FIXME
                #         images = grid.cpu().numpy()
                #         image_count = images.shape[0]
                #         for i in range(image_count):
                #             image = Image.fromarray((images[i] * 255).astype(np.uint8))
                #             image.save(f"outputs/{i}.png")
                # else:
                #     # as_vids = rearrange(samples, "(b t) c h w -> b t c h w", t=T)
                #     # for i, vid in enumerate(as_vids):
                #     #     grid = rearrange(make_grid(vid, nrow=4), "c h w -> h w c")
                #     #     st.image(
                #     #         grid.cpu().numpy(),
                #     #         f"Sample #{i} as image",
                #     #     )
                #     pass  # FIXME
                if return_latents:
                    return samples, samples_z
                return samples


def get_batch(
    keys,
    value_dict: dict,
    N: Union[List, ListConfig],
    device: str = os.environ.get("GPU_DEVICE", "cpu"),
    T: int = None,
    additional_batch_uc_fields: List[str] = [],
):
    """
    
    Note: Docstring was added by Cremage.

    Args:
        keys: Unique embedding keys from CLIP model
        value_dict: Input parameters
        N: [num_samples]

            Returns:
        Tuple of dicts (batch, batch_uc): Each dictionary that contains
        
    """
    batch = {}
    batch_uc = {}
    # List of keys:
    # Included:
    # "txt",
    # "original_size_as_tuple",
    # "crop_coords_top_left"
    # "target_size_as_tuple"
    # Not included:
    # "aesthetic_score"
    # "fps"
    # "fps_id"
    # "motion_bucket_id"
    # "pool_image"
    # "cond_aug"
    # "cond_frames"
    # "polars_rad"
    for key in keys:  
        if key == "txt":
            multiplier = math.prod(N)  # Cremage TODO: Check why prod? 4*3*2*1 = 24?
            # batch["txt"] = [value_dict["prompt"]] * math.prod(N)
            # batch_uc["txt"] = [value_dict["negative_prompt"]] * math.prod(N)

            batch["txt"] = [value_dict["prompt"]] * multiplier
            batch_uc["txt"] = [value_dict["negative_prompt"]] * multiplier

        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(os.environ.get("GPU_DEVICE", "cpu")),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        elif key == "polars_rad":
            batch[key] = torch.tensor(value_dict["polars_rad"]).to(device).repeat(N[0])
        elif key == "azimuths_rad":
            batch[key] = (
                torch.tensor(value_dict["azimuths_rad"]).to(device).repeat(N[0])
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    # Copy missing key value pair from batch to batch_uc
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key in additional_batch_uc_fields and key not in batch_uc:
            batch_uc[key] = copy.copy(batch[key])
    return batch, batch_uc


@torch.no_grad()
def do_img2img(
    img,
    model,
    sampler,
    value_dict,
    num_samples,
    opt=None,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    additional_kwargs={},
    offset_noise_level: int = 0.0,
    return_latents=False,
    skip_encode=False,
    safety_filter=None,
    add_noise=True,
):
    
    if torch.cuda.is_available() and opt.precision=="autocast":
        precision_context = torch.autocast(device_type=os.environ.get("GPU_DEVICE", "cpu"))
    else:
        precision_context = contextlib.nullcontext()

    # precision_scope = autocast
    with torch.no_grad():
        with precision_context:
            with model.ema_scope():
                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [num_samples],
                    device=os.environ.get("GPU_DEVICE", "cpu")
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                    embedding_dir=opt.embedding_path if opt is not None else None
                )

                # Create filler
                filler_value_dict = value_dict.copy()
                filler_value_dict["prompt"] = ""
                filler_value_dict["negative_prompt"] = ""
                filler_batch, filler_batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    filler_value_dict,
                    [num_samples],
                    device=os.environ.get("GPU_DEVICE", "cpu")
                )

                # filler = model.get_learned_conditioning(batch_size * [""], embedding_dir=opt.embedding_path, clip_skip=opt.clip_skip)
                filler, filler_uc = model.conditioner.get_unconditional_conditioning(
                    filler_batch,
                    batch_uc=filler_batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                # Calculate the difference in size along axis 1
                diff = uc["crossattn"].shape[1] - c["crossattn"].shape[1]
                assert diff % 77 == 0
                diff = diff // 77
                if diff < 0:  # uc is smaller
                    diff = -diff
                    # Repeat the filler tensor the required number of times and concatenate it with uc
                    repeated_filler = filler["crossattn"].repeat(1, diff, 1)  # The 'diff' times repetition is along the second axis
                    uc["crossattn"] = torch.cat((uc["crossattn"], repeated_filler), axis=1)
                else:  # c is smaller or they are equal
                    repeated_filler = filler["crossattn"].repeat(1, diff, 1)  # Repeat along the second axis
                    c["crossattn"] = torch.cat((c["crossattn"], repeated_filler), axis=1)
                assert uc["crossattn"].shape[1] == c["crossattn"].shape[1]
                # end filler

                unload_model(model.conditioner)
                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to(os.environ.get("GPU_DEVICE", "cpu")), (c, uc))

                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                if skip_encode:
                    z = img
                else:
                    load_model(model.first_stage_model)
                    z = model.encode_first_stage(img)
                    unload_model(model.first_stage_model)

                noise = torch.randn_like(z)

                sigmas = sampler.discretization(sampler.num_steps).to(os.environ.get("GPU_DEVICE", "cpu"))
                sigma = sigmas[0]

                logger.debug(f"all sigmas: {sigmas}")
                logger.debug(f"noising sigma: {sigma}")
                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(
                        torch.randn(z.shape[0], device=z.device), z.ndim
                    )
                if add_noise:
                    noised_z = z + noise * append_dims(sigma, z.ndim).to(os.environ.get("GPU_DEVICE", "cpu"))  # cuda()
                    noised_z = noised_z / torch.sqrt(
                        1.0 + sigmas[0] ** 2.0
                    )  # Note: hardcoded to DDPM-like scaling. need to generalize later.
                else:
                    noised_z = z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                load_model(model.first_stage_model)
                samples_x = model.decode_first_stage(samples_z)
                unload_model(model.first_stage_model)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if safety_filter is not None:
                    # samples = safety_filter(samples)
                    samples, _ = safety_filter(torch_images=samples)

                # CREMAGE Change: CHECK
                # grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                # outputs.image(grid.cpu().numpy())
  
                # grid = torch.stack([samples])
                # grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                # # outputs.image(grid.cpu().numpy())
                # # FIXME
                # images = grid.cpu().numpy()
                # image_count = images.shape[0]

                # enable_this = False
                # if enable_this:  # CREMAGE FIXME
                #     for i in range(image_count):
                #         image = Image.fromarray((images[i] * 255).astype(np.uint8))
                #         image.save(f"outputs/ref_{i}.png")  # 

                if return_latents:
                    return samples, samples_z
                return samples


def get_resizing_factor(
    desired_shape: Tuple[int, int], current_shape: Tuple[int, int]
) -> float:
    r_bound = desired_shape[1] / desired_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if aspect_r >= r_bound:
            factor = min(desired_shape) / min(current_shape)
        else:
            if aspect_r < 1.0:
                factor = max(desired_shape) / min(current_shape)
            else:
                factor = max(desired_shape) / max(current_shape)
    else:
        if aspect_r <= r_bound:
            factor = min(desired_shape) / min(current_shape)
        else:
            if aspect_r > 1:
                factor = max(desired_shape) / min(current_shape)
            else:
                factor = max(desired_shape) / max(current_shape)

    return factor


def load_img_for_prediction(
    image,
    W: int, H: int, display=True, key=None, device=os.environ.get("GPU_DEVICE", "cpu")
) -> torch.Tensor:

    if image is None:
        return None
    w, h = image.size

    image = np.array(image).astype(np.float32) / 255
    if image.shape[-1] == 4:
        rgb, alpha = image[:, :, :3], image[:, :, 3:]
        image = rgb * alpha + (1 - alpha)

    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32)
    image = image.unsqueeze(0)

    rfs = get_resizing_factor((H, W), (h, w))
    resize_size = [int(np.ceil(rfs * s)) for s in (h, w)]
    top = (resize_size[0] - H) // 2
    left = (resize_size[1] - W) // 2

    image = torch.nn.functional.interpolate(
        image, resize_size, mode="area", antialias=False
    )
    image = TT.functional.crop(image, top=top, left=left, height=H, width=W)

    return image.to(device) * 2.0 - 1.0


# Cremage: FIXME. Enable this.
# def save_video_as_grid_and_mp4(
#     video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5
# ):
#     os.makedirs(save_path, exist_ok=True)
#     base_count = len(glob(os.path.join(save_path, "*.mp4")))

#     video_batch = rearrange(video_batch, "(b t) c h w -> b t c h w", t=T)
#     video_batch = embed_watermark(video_batch)
#     for vid in video_batch:
#         save_image(vid, fp=os.path.join(save_path, f"{base_count:06d}.png"), nrow=4)

#         video_path = os.path.join(save_path, f"{base_count:06d}.mp4")
#         vid = (
#             (rearrange(vid, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
#         )
#         imageio.mimwrite(video_path, vid, fps=fps)

#         video_path_h264 = video_path[:-4] + "_h264.mp4"
#         os.system(f"ffmpeg -i '{video_path}' -c:v libx264 '{video_path_h264}'")
#         with open(video_path_h264, "rb") as f:
#             video_bytes = f.read()
#         os.remove(video_path_h264)
#         st.video(video_bytes)

#         base_count += 1
