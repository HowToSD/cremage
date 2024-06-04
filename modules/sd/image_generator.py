import os
import sys
import gc
import copy
import logging
import random
import json
import time
import contextlib

import cv2
import torch
import torch.nn.functional as F
import einops
from einops import rearrange
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL import ImageDraw, ImageFont
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from safetensors.torch import save_file, load_file
from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from .set_up_path import PROJECT_ROOT
LDM_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs/ldm/configs")
LDM_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/ldm")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
sys.path = [TOOLS_DIR] + sys.path
from cremage.configs.preferences import load_user_config
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler as DDIMControlNetSampler
from ldm.models.diffusion.k_diffusion_samplers import EulerAncestralSampler
from ip_adapter.ip_adapter_faceid import generate_face_embedding_from_image
from .options import parse_options
from cremage.utils.ml_utils import model_memory_usage_in_bytes
from cremage.utils.ml_utils import load_model, load_lora
from cremage.utils.ml_utils import face_id_model_weight_to_sd_15_model_weight
from cremage.utils.image_utils import bbox_for_multiple_of_64
from cremage.utils.image_utils import resize_with_padding
from cremage.utils.lora_utils import sd_weight_to_lora_weight
from cremage.ui.update_image_handler import update_image
from cremage.utils.generation_status_updater import StatusUpdater
from cremage.utils.sampler_utils import instantiate_sampler
from cremage.utils.hires_fix_upscaler_utils import hires_fix_upscaler_name_list
from cremage.utils.ml_utils import scale_pytorch_images
from cremage.utils.misc_utils import extract_embedding_filenames
from cremage.utils.wildcards import resolve_wildcards

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True
logger.debug(f"AutoFeatureExtractor and StableDiffusionSafetyChecker connection to internet disabled : {local_files_only_value}")

def load_img(path,
             height=None,
             width=None,
             padding_used=False):
    """
    Loads image from the file system and resize.

    When padding_used is set to True, height and width were already
    adjusted to the multiples of 64s, so no adjustment is needed.
    However, if the user specified an odd length (e.g. 509 w, 499h),
    then the crop bounding box has been defined which is used to
    crop the output. This bounding box can be different size
    than the image that this method loads.  Therefore, we need to
    compute the new cropbox and the caller needs to use that for cropping
    after image generation.

    Args:
    height: Target height of the image after loading
    width: Target width of the image after loading
    padding_used: Scale with padding

    Returns:
        Tuple of PIL image and a tuple of bounding box (x1, y1, x2, y2)
        to crop the image after the image generation.
    """

    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")

    if padding_used:
        # opt.W and opt.H were already adjusted to the multiple of 64s
        # so you can just use without further adjustment.
        image, bbox = resize_with_padding(
                            image,
                            target_width=width,
                            target_height=height, return_bbox=True)
        image = image.convert('RGB')
    else:
        if height is not None and width is not None:  # check if resize is needed
            w = width
            h = height
        else:
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        bbox = None

    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1., bbox


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def img2img_sampling(
        opt=None,
        use_control_net:bool=None,
        sampler=None,
        init_latent=None,
        start_code=None,
        full_denoising_sampling_steps=None,  # e.g. 50 full DDIM steps
        t_enc=None,  # e.g. 10 for 0.2 denoising strength
        batch_size=None,
        cond_dict=None,
        un_cond_dict=None,
        c=None,
        uc=None,
        eta=None,
        shape=None,
        scale=None,
        device=None,
        status_update_func=None
):
    intermediates = None

    if use_control_net:

        if isinstance(sampler, DDIMSampler) or isinstance(sampler, DDIMControlNetSampler):
            # encode (scaled latent)
            # t has been adjusted by denoising strength
            # e.g. for denoising strengh 0.5, t should be 25 for original 50 DDIM steps
            # So add initial 25 step worth of noise here
            z_enc = sampler.stochastic_encode(
                init_latent,  # Input image encoded in latent space

                # Number of steps to add noise
                # e.g. 10 DDIM int index steps if denoise is 0.2 and total steps = 50
                # This is used to add noise using the forward diffusion formula
                torch.tensor([t_enc]*batch_size).to(device)
            )                               
            # decode it
            # and denoise for 25 steps
            samples = sampler.decode(z_enc,
                                    cond_dict,  # passing dict for control net
                                    t_enc,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=un_cond_dict,  # passing dict for control net
                                    callback=status_update_func)
            
        else:  # K-diffusion sampler with ControlNet
            logger.debug("Adding noise using k-diffusion sampler")
            z_enc = sampler.stochastic_encode(
                init_latent,
                torch.tensor([t_enc]*batch_size).to(device),
                sampling_steps=full_denoising_sampling_steps
                )
            logger.debug("Denoising for img2img using k-diffusion sampler")
            samples, intermediates = sampler.sample(
                                        full_denoising_sampling_steps,
                                        batch_size,
                                        shape, 
                                        cond_dict, 
                                        verbose=False, 
                                        eta=eta,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=un_cond_dict,
                                        callback=status_update_func,
                                        x0=z_enc,
                                        denoising_steps=t_enc)

    else:  # No ControlNet

        if isinstance(sampler, DDIMSampler):
            # encode (scaled latent)
            # t has been adjusted by denoising strength
            # e.g. for denoising strengh 0.5, t should be 25 for original 50 DDIM steps
            # So add initial 25 step worth of noise here
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
            
            # decode it
            # and denoise for 25 steps
            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc,
                                    callback=status_update_func)
        else:  # K-diffusion sampler, no ControlNet
                    logger.debug("Adding noise using k-diffusion sampler")
                    z_enc = sampler.stochastic_encode(
                        init_latent,
                        torch.tensor([t_enc]*batch_size).to(device),
                        sampling_steps=full_denoising_sampling_steps
                        )
                    logger.debug("Denoising for img2img using k-diffusion sampler")
                    samples, intermediates = sampler.sample(S=full_denoising_sampling_steps,  # denoising steps
                                        conditioning=c,  # positive prompt
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,  # cfg
                                        unconditional_conditioning=uc,  # negative prompt
                                        eta=eta,
                                        x_T=start_code,
                                        callback=status_update_func,
                                        x0=z_enc,
                                        denoising_steps=t_enc)

    return samples, intermediates

def load_model_from_config(config, opt, verbose=False, inpainting=False):
    """
    
    Args:
        opt: End user-specified options
    """
    if inpainting:
        ckpt = opt.inpaint_ckpt
    else:
        ckpt = opt.ckpt

    logger.debug(f"Loading model from {ckpt}")
    pl_sd = load_model(ckpt)

    if "global_step" in pl_sd:  # if pl_sd["global_step"]
        print(f"Global Step: {pl_sd['global_step']}")

    # automatic1111's merge model uses a flat structure
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd

    # Load the main LatentDiffusionModel
    # model = instantiate_from_config(config.model)
    # Add additional config
        
    current_config = config.model

    # FaceID support
    if os.path.exists(opt.face_input_img) and os.path.exists(opt.face_model):
        # TODO. Consolidate check
        load_face_id = True
    else:
        load_face_id = False
    
    # Patch the parameters for LoRA
    if opt.lora_models is None or opt.lora_models == "":
        tmp_lora_model_list = []
        tmp_lora_weight_list = []
    else:
        tmp_lora_model_list = opt.lora_models.split(",")
        tmp_lora_weight_list = [float(v) for v in opt.lora_weights.split(",")]
    # Default initial value if no lora is used
    lora_ranks = []
    lora_weights = []
    loras = []  # model
    
    for i, lora_path in enumerate(tmp_lora_model_list):
        if len(lora_path) <= 0:
            continue
        print(f"Loading LoRA {i+1}: {lora_path}")

        lora, rank = load_lora(lora_path)
        lora_weight = tmp_lora_weight_list[i]
        lora_ranks.append(rank)
        lora_weights.append(lora_weight)
        loras.append(lora)

    if load_face_id:
        # Prepend as Face ID contains LoRA weights
        lora_ranks = [128] + lora_ranks
        lora_weights = [1.0] + lora_weights
        loras = [None] + loras
        current_config["params"]["unet_config"]["params"]["ipa_scale"] = opt.face_strength
        current_config["params"]["unet_config"]["params"]["ipa_num_tokens"] = 4
        
    current_config["params"]["unet_config"]["params"]["lora_ranks"] = lora_ranks
    current_config["params"]["unet_config"]["params"]["lora_weights"] = lora_weights
    current_config["params"]["cond_stage_config"]["params"]["lora_ranks"] = lora_ranks
    current_config["params"]["cond_stage_config"]["params"]["lora_weights"] = lora_weights

    # Instantiate the main model (LDM) using the config
    # Note this contains VAE, CLIP and UNet
    # Any weight that may be loaded here will be overridden later.
    logger.debug("Instantiating the main model (LDM).")
    print(current_config["params"]["unet_config"]["params"])
    model = instantiate_from_config(
                current_config)

    # Load ControlNet model
    if opt.control_models and os.path.exists(opt.control_models):
        logger.debug(f"Loading ControlNet model from {opt.control_models}")
        missing_keys, unexpected_keys = model.load_state_dict(
            load_state_dict(opt.control_models, location=os.environ.get("GPU_DEVICE", "cpu")),
            strict=False)
        if len(missing_keys) > 0 and verbose:
            print("missing keys:")
            print(missing_keys)
        if len(unexpected_keys) > 0 and verbose:
            print("unexpected keys:")
            print(unexpected_keys)

    # Load custom model parameter weights for the main model
    logger.debug("Loading the custom model weight values into the main model")
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if len(missing_keys) > 0 and verbose:
        print("missing keys:")
        print(missing_keys)
    if len(unexpected_keys) > 0 and verbose:
        print("unexpected keys:")
        print(unexpected_keys)

    # Override CLIP Text model with CLIP model parameter values from the custom model
    # Below CLIP loading should not be needed, so kept here for reference only
    # clip_sd_dict = dict()
    # for k, v in sd.items():
    #     if k.startswith("cond_stage_model.transformer"):
    #         k2 = k.replace("cond_stage_model.", "")
    #         if k2 != "transformer.text_model.embeddings.position_ids": # HACK. This key is missing in CLIP
    #             clip_sd_dict[k2] = v
    
    # # Add LoRA from original
    # for k, v in model.cond_stage_model.state_dict().items():
    #     if "lora" in k:
    #         clip_sd_dict[k] = v
    #         logger.debug(f"setting {k}")

    # model.cond_stage_model.load_state_dict(clip_sd_dict)
    # print("Overrode CLIP model parameters from the custom model")

    # Override VAE model with VAE model parameter values from the custom model
    #   weight path
    #   https://howtosd.com/parameter-list-of-v1-5-pruned-ckpt/
    #   first_stage_model.decoder.conv_in.bias
    #   https://howtosd.com/parameter-list-of-stable-diffusion-model-vae-ft-mse-840000-ema-pruned-ckpt/
    #   decoder.conv_in.bias
    if hasattr(opt, "vae_ckpt") and not opt.vae_ckpt.endswith("None"):
        vae_model = load_model(opt.vae_ckpt)
        if not isinstance(vae_model, dict):
            vae_model = vae_model.state_dict()
        if "state_dict" in vae_model:
            vae_model = vae_model["state_dict"]

        # HACK
        for unwanted_key in ["model_ema.decay", "model_ema.num_updates"]:
            if unwanted_key in vae_model:
                del vae_model[unwanted_key]
                logger.debug(f"Removed {unwanted_key} from the custom vae model before updating the main VAE")

        model.first_stage_model.load_state_dict(vae_model)
        logger.debug(f"Overrode VAE model parameters from the custom model {os.path.basename(opt.vae_ckpt)}")

    # Load LORA weight
    # Name mapping
    # Name in SD1.5
    #   model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.q_lora_downs.0.weight
    # Name in LoRA model file
    lora_count = 0

    for i, lora in enumerate(loras):
        if lora is None:
            if load_face_id:
                logger.debug("Skipping LoRA weight mapping for Face ID as this is loaded separately.")
                continue
            else:
                raise ValueError("LoRA weight is none despite the valid LoRA path.")

        for k, v in model.named_parameters():
            if "_lora_" in k:

                lora_weight_name = sd_weight_to_lora_weight(k)
                if lora_weight_name not in lora:
                    # Check to see if LoRA contains the specific LoRA weight
                    # Note that not all LoRA weights are in a LoRA model.                   
                    # e.g. LoRA may not contain CLIP weight
                    if lora_weight_name.startswith("lora_te_text_model_encoder_layers_"):
                        continue
                    else:
                        logger.debug(f"Missing unet lora weight: {lora_weight_name}")
                    continue
                # k = lora[lora_weight_name]

                # Split the sd weight name to get the lora index
                # We are only setting the lora that matches the lora index
                #  ...fc2_lora_ups.1.weight
                #  ...fc2_lora_alphas.0

                tmp_list = k.split(".")
                if tmp_list[-1] == "weight":
                    tmp_list.pop()

                lora_index = tmp_list[-1]
                try:
                    lora_index = int(lora_index)
                except:
                    ValueError("LoRA index of the parameter not found: {k}")

                if lora_index != i:  # skip if the lora weight for other loras
                    continue

                # LoRA is found, check the shape
                if v.shape != lora[lora_weight_name].shape:
                    raise ValueError(f"LoRA weight shape mismatch: {lora_weight_name}. Expecting: {lora[lora_weight_name].shape}  Got:{v.shape}")
                
                # Split the parameter name to separate the submodule path and the parameter name
                path_parts = k.split('.')
                submodule = model
                for part in path_parts[:-1]:  # Navigate to the submodule
                    submodule = getattr(submodule, part)
                
                # Now, set the parameter on the submodule
                setattr(submodule, path_parts[-1], torch.nn.Parameter(lora[lora_weight_name]))
                lora_count += 1

    if lora_count > 0:
        logger.debug(f"Number of overridden Lora parameters: {lora_count}")

    # Load Face ID
    # Note that the weight file contains image_proj key which is for the
    # transformers used in generating face embedding.
    # This weight is loaded during face embedding generation, so you do not
    # need to load them in here.
    # Refer to docs/ip_adapter_face_id_plus_weight_list.md for the names
    # of weights
    if load_face_id:
        ldm_model_sd = model.state_dict() 
        face_id_weights = load_model(opt.face_model)
        logger.debug("Loaded face model weight file.")
        for k, v in face_id_weights["ip_adapter"].items():
            sdw = face_id_model_weight_to_sd_15_model_weight(k)
            
            if sdw not in ldm_model_sd:
                raise ValueError(f"Missing: {sdw}")
            if ldm_model_sd[sdw].shape != v.shape:
                raise ValueError(f"weight mismatch for {k}: {v.shape} vs {ldm_model_sd[sdw].shape}")

            # Now actually load weight
            # Split the parameter name to separate the submodule path and the parameter name
            path_parts = sdw.split('.')
            submodule = model
            for part in path_parts[:-1]:  # Navigate to the submodule
                submodule = getattr(submodule, part)
            
            # Now, set the parameter on the submodule
            setattr(submodule, path_parts[-1], torch.nn.Parameter(v))

    # Print out memory usage
    before_half_usage = model_memory_usage_in_bytes(model, include_gradients=True)
    model.half()  # Convert to float16
    after_half_usage = model_memory_usage_in_bytes(model, include_gradients=True)
    logger.debug(f"GPU Memory usage. Before float16 conversion: {before_half_usage}. After: {after_half_usage}")

    model.to(os.environ.get("GPU_DEVICE", "cpu"))
    model.eval()
    after_eval = model_memory_usage_in_bytes(model, include_gradients=True)
    logger.debug(f"GPU Memory usage. After switching to eval mode: {after_eval}")
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        # image_file_path = os.path.join(PROJECT_ROOT, "resources", "images", "safety_replacement_background.png")
        # y = Image.open(image_file_path).convert("RGB").resize((hwc[1], hwc[0]))
        y = Image.new('RGB', (hwc[1], hwc[0]), "black")

        # Create a draw object
        draw = ImageDraw.Draw(y)

        # Set the text to be added
        text = "Safety check filtered potentially sensitive image.\nTo change settings, go to File | Preferenses."
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            logger.debug("arial.ttf not found. Using default font.")
            font = ImageFont.load_default()

        bbox = draw.multiline_textbbox((0, 0), text, font=font, align="left")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (hwc[1] - text_width) / 2
        text_y = (hwc[0] - text_height) / 2
        draw.multiline_text((text_x, text_y), text, font=font, fill="white", align="center")
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id, local_files_only=local_files_only_value)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id, local_files_only=local_files_only_value)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def scale_control_image(input_image, hires_fix_upscale_factor, num_samples):
    scaled_h = input_image.shape[0]*hires_fix_upscale_factor
    scaled_w = input_image.shape[1]*hires_fix_upscale_factor
    scaled_input_image = cv2.resize(input_image,
                                    (scaled_w, scaled_h),
                                        interpolation=cv2.INTER_LANCZOS4)

    scaled_control = torch.from_numpy(scaled_input_image.copy()).float().to(os.environ.get("GPU_DEVICE", "cpu")) / 255.0
    scaled_control = torch.stack([scaled_control for _ in range(num_samples)], dim=0)
    scaled_control = einops.rearrange(scaled_control, 'b h w c -> b c h w').clone()
    return scaled_control


def generate(opt,
             ui_thread_instance=None,
             generation_type="txt2img",
             status_queue=None):
    """
    Generate one or more images using the specified options.
    ui_thread_instance is used for callbacks to update UI.
    """
    # Add padding to generate the image that has edge length being the multiples of 64
    # Note that bbox_to_crop may be overridden in img2img2 preprocessing
    # so that the original source image won't be clipped or padded.
    original_width, original_height = opt.W, opt.H
    padded_w, padded_h = bbox_for_multiple_of_64(opt.W, opt.H)
    if padded_w != opt.W or padded_h != opt.H:
        padding_used = True
        # Compute left top to crop after the image is generated
        crop_x = (padded_w - opt.W) // 2 + (padded_w - opt.W) % 2
        crop_y = (padded_h - opt.H) // 2 + (padded_h - opt.H) % 2
        opt.W, opt.H = padded_w, padded_h
        bbox_to_crop = (crop_x, crop_y, crop_x + original_width, crop_y + original_height)
    elif generation_type == "img2img":
        padding_used = True
        bbox_to_crop = None  # This is computed later
    else:
        padding_used = False

    # Hires fix
    hires_fix_upscale_factor = opt.hires_fix_scale_factor
    if opt.hires_fix_upscaler and \
       opt.hires_fix_upscaler.lower() != "none" and \
       opt.hires_fix_upscaler in hires_fix_upscaler_name_list:
        use_hires_fix = True
    else:
        use_hires_fix = False

    # Seed
    if opt.seed == -1:
        seed = random.getrandbits(32)
    else:
        seed = opt.seed
    seed_everything(seed)

    # Determine if ControlNet is to be used
    use_control_net = (
        opt.control_models and
        os.path.exists(opt.control_models) and
        os.path.exists(opt.control_image_path)
    )

    # Read config yaml file
    if use_control_net:
        config = OmegaConf.load(f"{opt.control_net_config}")
    else:
        config = OmegaConf.load(f"{opt.config}")

    # Process FaceID
    load_face_id = False
    if os.path.exists(opt.face_input_img) and os.path.exists(opt.face_model):
        logger.debug("Face input image is specified and the face model is found. Generating face embedding")
        face_embedding, uc_face_embedding = generate_face_embedding_from_image(
            opt.face_input_img,
            opt.face_model,
            batch_size=opt.n_samples)
        logger.debug(f"Generated face_embedding. Shape: {face_embedding.shape}")
        load_face_id = True

        # # FIXME
        # face_embedding = torch.tensor(np.load("pos.npy"), dtype=torch.float16).to("cuda")
        # uc_face_embedding = torch.tensor(np.load("unc.npy"), dtype=torch.float16).to("cuda")

    elif os.path.exists(opt.face_input_img) and os.path(opt.face_model) is False:
        logger.debug("Face input image is specified but the face model is not found. Ignoring face image")

    # Load the main LDM model
    logger.debug("Instantiating the main model ...")
    model = load_model_from_config(config, opt)
    logger.debug("Instantiated the main model")

    device = torch.device(os.environ.get("GPU_DEVICE", "cpu"))
    model = model.to(device)

    if opt.save_memory:
        model.low_vram_shift(is_diffusing=False)

    # Instantiate sampler
    if generation_type != "txt2img":
        opt.sampler = "DDIM"
        logger.debug("Swiching the sampler to DDIM as image quality may not be optimum on Cremage with other samplers.")
    sampler = instantiate_sampler(opt.sampler, model, use_control_net)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    if opt.watermark:
        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    prompt = opt.prompt
    assert prompt is not None

    negative_prompt = opt.negative_prompt
    assert negative_prompt is not None

    data = [batch_size * [prompt]]  # "banana" becomes [["banana", "banana"]]
    data_negative = [batch_size * [negative_prompt]]
    data_original = copy.deepcopy(data)
    data_negative_original = copy.deepcopy(data_negative)

    sample_path = outpath
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
   
    if generation_type == "img2img":
        assert os.path.isfile(opt.init_img)
        # bbox_crop is to be used to crop the image
        init_image, bbox_to_crop = load_img(
            opt.init_img,
            height=opt.H,
            width=opt.W,
            padding_used=padding_used)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        if isinstance(sampler, DDIMSampler) or isinstance(sampler, DDIMControlNetSampler):
            sampler.make_schedule(ddim_num_steps=opt.sampling_steps, ddim_eta=opt.ddim_eta, verbose=False)

        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.sampling_steps)
        print(f"target t_enc is {t_enc} steps")
    else:  # txt2img
        t_enc = opt.sampling_steps

    # Initialize status updater object. This object contains a method
    # which is invoked as a callback for each sampling step.
    status_update_func = None
    if status_queue:
        if use_hires_fix:
            steps_to_track = t_enc + int(opt.strength * opt.sampling_steps)
        else:
            steps_to_track = t_enc
        
        status_updater = StatusUpdater(steps_to_track, int(opt.n_iter), status_queue)
        status_update_func = status_updater.status_update

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    if torch.cuda.is_available() and opt.precision=="autocast":
        precision_context = torch.autocast(device_type=os.environ.get("GPU_DEVICE", "cpu"))
    else:
        precision_context = contextlib.nullcontext()
 
    sampling_steps = opt.sampling_steps # 50
    scale = opt.scale # 7.5
    eta = opt.ddim_eta  # 0.0
        
    with torch.no_grad():  # Disable gradient computation
        with precision_context:
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for batch_index, n in enumerate(trange(opt.n_iter, desc="Number of batches")):

                    # Resolve wildcards
                    # Copy the original prompt with wildcards
                    data = copy.deepcopy(data_original)
                    data_negative = copy.deepcopy(data_negative_original)
                    # Cremage note:
                    # Use the same prompt within the same batch.
                    # This is due to current limitation because of the way
                    # we process in our custom CLIP code
                    resolved_prompt = resolve_wildcards(data[0][0], wildcards_dir=opt.wildcards_path)
                    resolved_prompt_negative = resolve_wildcards(data_negative[0][0], wildcards_dir=opt.wildcards_path)

                    for sample_index in range(opt.n_samples):
                        data[0][sample_index] = resolved_prompt
                        data_negative[0][sample_index] = resolved_prompt_negative

                    for prompts in tqdm(data, desc="data"):  # Batch size
                        # 1. Generate CLIP Text Embedding
                        uc = None
                        if opt.scale != 1.0:
                            # see ddpm.py for the method definition
                            # uc = model.get_learned_conditioning(batch_size * [""])
                            uc = model.get_learned_conditioning(data_negative[0], embedding_dir=opt.embedding_path, clip_skip=opt.clip_skip)

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts, embedding_dir=opt.embedding_path, clip_skip=opt.clip_skip)
                        # Negative embedding (uc) and positive embedding(c) can have
                        # a size mismatch if embedding is concatenated along the first axis.
                        # e.g. 308 vs 77 for tensor shape [2, 308, 768] vs [2, 77, 768]
                        # To cope with this, we will be using the filler embedding 
                        filler = model.get_learned_conditioning(batch_size * [""], embedding_dir=opt.embedding_path, clip_skip=opt.clip_skip)
                        
                        # Calculate the difference in size along axis 1
                        diff = uc.shape[1] - c.shape[1]
                        assert diff % 77 == 0
                        diff = diff // 77
                        if diff < 0:  # uc is smaller
                            diff = -diff
                            # Repeat the filler tensor the required number of times and concatenate it with uc
                            repeated_filler = filler.repeat(1, diff, 1)  # The 'diff' times repetition is along the second axis
                            uc = torch.cat((uc, repeated_filler), axis=1)
                        else:  # c is smaller or they are equal
                            repeated_filler = filler.repeat(1, diff, 1)  # Repeat along the second axis
                            c = torch.cat((c, repeated_filler), axis=1)
                        assert uc.shape[1] == c.shape[1]

                        if load_face_id:  # Append face embedding. New shape should be (1, 81, 768)
                            c = torch.concat((c, face_embedding), axis=1)
                            uc = torch.concat((uc, uc_face_embedding), axis=1)
                            assert(c.shape[1] % 77 == 4 and c.shape[2] == 768)
                            assert(uc.shape[1] % 77 == 4 and uc.shape[2] == 768)

                        # 2. Generate image in latent space using UNet
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                        if opt.save_memory:
                            model.low_vram_shift(is_diffusing=True)

                        if use_control_net:
                            logger.debug("Calling sampler with control image")
                            guess_mode = False 
                            num_samples = opt.n_samples
                            H = opt.H
                            W = opt.W

                            # Load control image
                            input_image = cv2.imread(opt.control_image_path)
                            control = torch.from_numpy(input_image.copy()).float().to(os.environ.get("GPU_DEVICE", "cpu")) / 255.0
                            control = torch.stack([control for _ in range(num_samples)], dim=0)
                            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

                            # Each conditioning is a dictionary.
                            # Each key in the dictionary has a value which is a list.
                            cond_dict = {
                                        "c_concat": [control],
                                        "c_crossattn": [c]
                                    }
                            un_cond_dict = {
                                        "c_concat": None if guess_mode else [control],
                                        "c_crossattn":
                                            [uc]}
                            shape = (4, H // 8, W // 8)

                            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
                            strength = 1.0 # FIXME
                            model.control_scales = [
                                    strength * (0.825 ** float(12 - i)) for i in range(13)
                                ] if guess_mode else ([strength] * 13)
                            
                            if generation_type == "img2img":  # with ControlNet

                                samples, _ = img2img_sampling(
                                    opt=opt,
                                    use_control_net=True,
                                    sampler=sampler,
                                    init_latent=init_latent,
                                    start_code=start_code,
                                    full_denoising_sampling_steps=sampling_steps,
                                    t_enc=t_enc,
                                    batch_size=batch_size,
                                    cond_dict=cond_dict,
                                    un_cond_dict=un_cond_dict,
                                    c=None,
                                    uc=None,
                                    eta=eta,
                                    shape=shape,  # shape of a single latent image (C, H, W)
                                    scale=scale,  # cfg
                                    device=device,
                                    status_update_func=status_update_func
                                )

                            else:  # txt2img with ControlNet
                                samples, intermediates = sampler.sample(
                                                            sampling_steps,
                                                            num_samples,
                                                            shape, 
                                                            cond_dict, 
                                                            verbose=False, 
                                                            eta=eta,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond_dict,
                                                            callback=status_update_func)

                                if use_hires_fix and opt.hires_fix_upscaler.lower() == "latent":

                                    if isinstance(sampler, DDIMSampler) or isinstance(sampler, DDIMControlNetSampler):
                                        sampler.make_schedule(ddim_num_steps=opt.sampling_steps, ddim_eta=opt.ddim_eta, verbose=False)

                                    samples = F.interpolate(samples, scale_factor=hires_fix_upscale_factor, mode='bilinear', align_corners=False)
                                    t_enc = int(opt.strength * opt.sampling_steps)
                                    print(f"target t_enc is {t_enc} steps")

                                    # Scale control image
                                    scaled_control = \
                                        scale_control_image(input_image, hires_fix_upscale_factor, num_samples)
                                   
                                    cond_dict = {
                                                "c_concat": [scaled_control],
                                                "c_crossattn": [c]
                                            }
                                    un_cond_dict = {
                                                "c_concat": None if guess_mode else [scaled_control],
                                                "c_crossattn":
                                                    [uc]}
                                    shape = (4, 
                                             opt.H * hires_fix_upscale_factor // 8,
                                             opt.W * hires_fix_upscale_factor // 8)

                                    samples, _ = img2img_sampling(
                                        opt=opt,
                                        use_control_net=True,
                                        sampler=sampler,
                                        init_latent=samples,
                                        start_code=start_code,
                                        full_denoising_sampling_steps=sampling_steps,
                                        t_enc=t_enc,
                                        batch_size=batch_size,
                                        cond_dict=cond_dict,
                                        un_cond_dict=un_cond_dict,
                                        c=None,
                                        uc=None,
                                        eta=eta,
                                        shape=shape,  # shape of a single latent image (C, H, W)
                                        scale=scale,  # cfg
                                        device=device,
                                        status_update_func=status_update_func
                                    )

                        else:  # No ControlNet
                            if generation_type == "img2img":

                               samples, _ = img2img_sampling(
                                    opt=opt,
                                    use_control_net=False,
                                    sampler=sampler,
                                    init_latent=init_latent,
                                    start_code=start_code,
                                    full_denoising_sampling_steps=sampling_steps,
                                    t_enc=t_enc,
                                    batch_size=batch_size,
                                    cond_dict=None,
                                    un_cond_dict=None,
                                    c=c,
                                    uc=uc,
                                    eta=eta,
                                    shape=shape,  # shape of a single latent image (C, H, W)
                                    scale=scale,
                                    device=device,
                                    status_update_func=status_update_func
                                )

                            elif generation_type == "txt2img":  # No ControlNet
                                samples, _ = sampler.sample(S=opt.sampling_steps,  # denoising steps
                                                                conditioning=c,  # positive prompt
                                                                batch_size=opt.n_samples,  # batch size
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=opt.scale,  # cfg
                                                                unconditional_conditioning=uc,  # negative prompt
                                                                eta=opt.ddim_eta,
                                                                x_T=start_code,
                                                                callback=status_update_func)

                                if use_hires_fix and opt.hires_fix_upscaler.lower() == "latent":

                                    if isinstance(sampler, DDIMSampler) or isinstance(sampler, DDIMControlNetSampler):
                                        sampler.make_schedule(ddim_num_steps=opt.sampling_steps, ddim_eta=opt.ddim_eta, verbose=False)

                                    # Scale in latent space
                                    samples = F.interpolate(samples, scale_factor=hires_fix_upscale_factor, mode='bilinear', align_corners=False)
                                    
                                    # Compute number of steps to denoise
                                    t_enc = int(opt.strength * opt.sampling_steps)
                                    print(f"target t_enc is {t_enc} steps")
        
                                    samples, _ = img2img_sampling(
                                            opt=opt,
                                            use_control_net=False,
                                            sampler=sampler,
                                            init_latent=samples,
                                            start_code=start_code,
                                            full_denoising_sampling_steps=sampling_steps,                                                
                                            t_enc=t_enc,
                                            batch_size=batch_size,
                                            cond_dict=None,
                                            un_cond_dict=None,
                                            c=c,
                                            uc=uc,
                                            eta=eta,
                                            shape=(4, opt.H * hires_fix_upscale_factor, opt.W * hires_fix_upscale_factor),  # shape of a single latent image (C, H, W)
                                            scale=scale,
                                            device=device,
                                            status_update_func=status_update_func
                                        )

                        if opt.save_memory:
                            model.low_vram_shift(is_diffusing=False)

                        # 3. Convert image from latent space to pixel space using VAE
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if use_hires_fix and opt.hires_fix_upscaler.lower() == "lanczos":

                            # Scale image in pixel space
                            scaled_samples = scale_pytorch_images(x_samples,
                                                                    width=int(opt.W * hires_fix_upscale_factor),
                                                                    height=int(opt.H * hires_fix_upscale_factor),
                                                                    interpolation=cv2.INTER_LANCZOS4)
                            scaled_samples = scaled_samples * 2.0 - 1.0  # [0, 1] to [0, 2] to [-1, -1]

                            # Convert to latents
                            init_latent = model.get_first_stage_encoding(
                                model.encode_first_stage(scaled_samples))

                            if isinstance(sampler, DDIMSampler) or isinstance(sampler, DDIMControlNetSampler):
                                sampler.make_schedule(ddim_num_steps=opt.sampling_steps, ddim_eta=opt.ddim_eta, verbose=False)

                            assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
                            t_enc = int(opt.strength * opt.sampling_steps)
                            print(f"target t_enc is {t_enc} steps")

                            if opt.save_memory:
                                model.low_vram_shift(is_diffusing=True)

                            # Right parameters should be already populated
                            # for txt2img, so do not repeat setting them here except below:

                            if use_control_net:
                                # Scale control image
                                scaled_control = \
                                    scale_control_image(input_image, hires_fix_upscale_factor, num_samples)
                                
                                cond_dict = {
                                            "c_concat": [scaled_control],
                                            "c_crossattn": [c]
                                        }
                                un_cond_dict = {
                                            "c_concat": None if guess_mode else [scaled_control],
                                            "c_crossattn":
                                                [uc]}
                                c_input = None
                                uc_input = None
                                cond_dict_input = cond_dict
                                un_cond_dict_input = un_cond_dict                                    
                            else:
                                c_input = c
                                uc_input = uc
                                cond_dict_input = None
                                un_cond_dict_input = None

                            samples, _ = img2img_sampling(
                                opt=opt,
                                use_control_net=use_control_net,
                                sampler=sampler,
                                init_latent=init_latent,
                                start_code=start_code,
                                full_denoising_sampling_steps=sampling_steps,
                                t_enc=t_enc,
                                batch_size=batch_size,
                                cond_dict=cond_dict_input,
                                un_cond_dict=un_cond_dict_input,
                                c=c_input,
                                uc=uc_input,
                                eta=eta,
                                shape=(4, opt.H * hires_fix_upscale_factor, opt.W * hires_fix_upscale_factor),  # shape of a single latent image (C, H, W)
                                scale=scale,  # cfg
                                device=device,
                                status_update_func=status_update_func
                            )

                            if opt.save_memory:
                                model.low_vram_shift(is_diffusing=False)

                            # 3. Convert image from latent space to pixel space using VAE
                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if opt.safety_check:
                            x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
                            # Pass B, H, W, C numpy image data to check_safety
                            x_checked_image, has_nsfw_concept = check_safety(x_samples)
                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        else:
                            x_checked_image_torch = x_samples

                        # Clean up LoRA params
                        if opt.lora_models:
                            paths = opt.lora_models.split(",")
                            lora_models = [os.path.basename(p) for p in paths]
                            lora_models = ",".join(lora_models)
                        else:
                            lora_models = []
                        if not opt.skip_save:
                            height = bbox_to_crop[3] - bbox_to_crop[1] if padding_used else opt.H
                            width = bbox_to_crop[2] - bbox_to_crop[0] if padding_used else opt.W

                            for i, x_sample in enumerate(x_checked_image_torch):

                                generation_parameters = {
                                    "time": time.time(),
                                    "positive_prompt": data[0][0],  # Note, currently, same prompt is used within the same batch
                                    "negative_prompt": data_negative[0][0],
                                    "ldm_model": os.path.basename(opt.ckpt),
                                    "vae_model": os.path.basename(opt.vae_ckpt),
                                    "lora_models": lora_models,
                                    "lora_weights": opt.lora_weights,
                                    "sampler": opt.sampler,
                                    "sampling_iterations": opt.sampling_steps,
                                    "cfg": opt.scale,
                                    "image_height": height,
                                    "image_width": width,
                                    "clip_skip": opt.clip_skip,
                                    "seed": seed + i,
                                    "watermark": opt.watermark,
                                    "safety_check": opt.safety_check
                                }

                                if use_control_net:
                                    generation_parameters["control_net"] = os.path.basename(opt.control_models)
                                if load_face_id:
                                    generation_parameters["face_image"] = os.path.basename(opt.face_input_img)
                                    generation_parameters["face_strength"] = opt.face_strength
                                if use_hires_fix:
                                    generation_parameters["hires_fix_upscaler"] = opt.hires_fix_upscaler
                                    generation_parameters["hires_fix_scale_factor"] = opt.hires_fix_scale_factor
                                    generation_parameters["upscale_width"] = opt.W * hires_fix_upscale_factor
                                    generation_parameters["upscale_height"] = opt.H * hires_fix_upscale_factor

                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                if opt.watermark:
                                    img = put_watermark(img, wm_encoder)
                                time_str = time.time()

                                str_generation_params = json.dumps(generation_parameters)
                                metadata = PngInfo()
                                metadata.add_text("generation_data", str_generation_params)

                                # Remove padding
                                if padding_used and bbox_to_crop:
                                    img = img.crop(bbox_to_crop)

                                # Extra processing start
                                if opt.auto_face_fix:
                                    from face_fixer import FaceFixer
                                    logger.debug("Applying face fix")
                                    status_queue.put("Applying face fix")
                                    app = load_user_config()
                                    face_fixer = FaceFixer(
                                        preferences=app,
                                        positive_prompt=opt.prompt,
                                        negative_prompt=opt.negative_prompt,
                                        procedural=True,
                                        status_queue=status_queue)
                                    img = face_fixer.fix_with_insight_face(img)
                                # Extra processing end
                                file_name = f"{base_count:05}_{time_str}.png"
                                img.save(
                                    os.path.join(sample_path, file_name),
                                    pnginfo=metadata)

                                if status_queue:
                                    status_queue.put(f"Saved {file_name}")

                                # Save image as a sample image for the embeddings used
                                # Extract embedding file name if any
                                embedding_images_dir = opt.embedding_images_dir
                                if embedding_images_dir:
                                    if os.path.exists(embedding_images_dir) is False:
                                        os.makedirs(embedding_images_dir)
                                for prompt in [opt.prompt, opt.negative_prompt]:
                                    embedding_file_names = extract_embedding_filenames(prompt)
                                    for file_name in embedding_file_names:
                                        img.save(
                                            os.path.join(embedding_images_dir,
                                                 f"{file_name}.png"),
                                            pnginfo=metadata)

                                # Pass img (PIL Image) to the main thread here!
                                if ui_thread_instance:
                                    update_image(ui_thread_instance,
                                                 img,
                                                 generation_parameters=str_generation_params)
                                    
                                base_count += 1
                            
                            # single image processing end
                        # end of skip or not skip save if
                # batch loop end
                toc = time.time()

    # Release memory
    del samples, x_samples, c, uc, start_code

    # Clear PyTorch's cache
    device_type = os.environ.get("GPU_DEVICE", "cpu")
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()  # Only call this if CUDA is available and intended for use

    # Manually collect garbage
    gc.collect()

    print(f"Image generation completed. Image were generated in: {outpath}")


def parse_options_and_generate(args=None, ui_thread_instance=None):
    opt = parse_options(args)    
    generate(opt, ui_thread_instance=ui_thread_instance)


if __name__ == "__main__":
    parse_options_and_generate()
