"""
Miscellaneous utility functions
"""
import os
import logging
from typing import List, Tuple

import torch
import einops
from safetensors.torch import load_file
import numpy as np
import cv2

from .lora_utils import LORA_WEIGHTS
from .safetensors_utils import sd_model_type_from_safetensors
from .safetensors_utils import MODEL_TYPE_UNKNOWN
from .safetensors_utils import MODEL_TYPE_SD_1_5
from .safetensors_utils import MODEL_TYPE_SDXL

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def scale_pytorch_images(samples:torch.tensor,
                         normalized:bool=True,
                         width:int=None,
                         height:int=None,
                         method:str=None,
                         **kwargs):
    """
    Scales images contained in torch tensor.

    Args:
        x_samples (torch.tensor): A torch tensor of shape (b, c, h, w).
        normalized (bool): If True, data contains pixel values [0.0-1.0].
            If False, data contains pixel values [0,255] in uint8 which
            does not require pre or postprocessing.
        width (int): The width of the scaled image.
        height (int): The height of the scaled image.
        method (str): A scaling method. Only "cv2" is supported.
        **kwargs: Extra argument. Currently interpolation is the only argument supported.
    
    Returns:
        Scaled torch.tensor.
    """
    b, c, _, _ = samples.shape
    samples = einops.rearrange(samples, "b c h w -> b h w c")
    device = samples.device
    interpolation_method=kwargs["interpolation"]
    scaled_images = list()

    if normalized:
        samples = (samples * 255.0).to(torch.uint8)

    scaled_images = np.empty((b, height, width, c), dtype=np.float32)  # Pre-allocate memory
    for i in range(b):
        img = samples[i].detach().cpu().numpy()
        # Note that cv2 size is specified by (width, height)
        scaled = cv2.resize(img, (width, height), interpolation=interpolation_method)
        scaled_images[i] = scaled

    scaled_images = torch.tensor(scaled_images).to(device)
    scaled_images = einops.rearrange(scaled_images, "b h w c->b c h w")

    if normalized:
        scaled_images = (scaled_images / 255.0).float()
    return scaled_images

def face_id_model_weight_to_sd_15_model_weight(w: str):
    """
    Converts Face ID mmodel weight name to SD15 model weight name.

    There are 32 blocks that correspond to non-sequentially named attention block names in SD:

    Example FaceID names:
    0.to_q_lora.down.weight
    1.to_q_lora.down.weight

    SD 1.5 weight names:                        (Mapping to Face ID index)
    ---------------------------------------------------------------------------------------
    Down blocks:
    model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.q_lora_downs.0.weight
    model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.q_lora_downs.0.weight
                                                (0, 1)
    ...                 
    ...                                2.1      (2, 3)
    ...                                4.1      (4, 5)
    ...                                5.1      (6, 7)
    ...                                7.1      (8, 9)
    ...
    model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.q_lora_downs.0.weight
                                                (10, 11)
    
    Up blocks
    model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.q_lora_downs.0.weight
    model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.q_lora_downs.0.weight
                                                (12, 13)
    ...
                                            4.1 (14, 15)
                                            5.1 (16, 17)
                                            6.1 (18, 19)
                                            7.1 (20, 21)
                                            8.1 (22, 23)
                                            9.1 (24, 25)
                                            10.1 (26, 27)
                                            11.1 (28, 29)

    Mid blocks
    model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.q_lora_downs.0.weight
    model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.q_lora_downs.0.weight
                                            (30, 31)
    Split SD to two components:
    attention_block_name: model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.
    weight_name: q_lora_downs.0.weight

    Variables in parentheses:
    attention_block_name: model.diffusion_model.(output)_blocks.(3.1).transformer_blocks.0.attn(1).

    See: 
    docs/ip_adapter_face_id_plus_weight_list.md
    modules/cremage/utils/sd15_weight_list_with_lora.py
    """
    def _get_sd_attention_block_name_from_face_id_block_index(face_id_block_index: int):
        """

        Args:
            face_id_block_index (int): 0-31
        """
        block_data = [
            ("input_blocks", "1.1", "1"),
            ("input_blocks", "1.1", "2"),
            ("input_blocks", "2.1", "1"),
            ("input_blocks", "2.1", "2"),
            ("input_blocks", "4.1", "1"),
            ("input_blocks", "4.1", "2"),
            ("input_blocks", "5.1", "1"),
            ("input_blocks", "5.1", "2"),
            ("input_blocks", "7.1", "1"),
            ("input_blocks", "7.1", "2"),
            ("input_blocks", "8.1", "1"),
            ("input_blocks", "8.1", "2"),

            ("output_blocks", "3.1", "1"),
            ("output_blocks", "3.1", "2"),
            ("output_blocks", "4.1", "1"),
            ("output_blocks", "4.1", "2"),
            ("output_blocks", "5.1", "1"),
            ("output_blocks", "5.1", "2"),
            ("output_blocks", "6.1", "1"),
            ("output_blocks", "6.1", "2"),
            ("output_blocks", "7.1", "1"),
            ("output_blocks", "7.1", "2"),
            ("output_blocks", "8.1", "1"),
            ("output_blocks", "8.1", "2"),
            ("output_blocks", "9.1", "1"),
            ("output_blocks", "9.1", "2"),
            ("output_blocks", "10.1", "1"),
            ("output_blocks", "10.1", "2"),
            ("output_blocks", "11.1", "1"),
            ("output_blocks", "11.1", "2"),

            ("middle_block", "1", "1"),
            ("middle_block", "1", "2")
        ]
        return block_data[face_id_block_index]

    components = w.split(".")
    if len(components) == 4:  # Lora weight
        # 0.to_q_lora.down.weight
        index = int(components[0])
        qkvout = components[1][3:]  # to_q_lora -> q_lora
        updown = components[2]  # up or down
        # w[3] = "weight"
        in_mid_out, block_num, attn_index = _get_sd_attention_block_name_from_face_id_block_index(index)
        retval = f"model.diffusion_model.{in_mid_out}.{block_num}.transformer_blocks.0.attn{attn_index}."
        retval += f"{qkvout}_{updown}s.0.weight"

    elif len(components) == 3:  # IP k, v weight
        index = int(components[0])
        kv = components[1]  # to_k_ip or to_v_ip
        # w[2] = "weight"
        in_mid_out, block_num, attn_index = _get_sd_attention_block_name_from_face_id_block_index(index)
        retval = f"model.diffusion_model.{in_mid_out}.{block_num}.transformer_blocks.0.attn{attn_index}."
        retval += f"{kv}a.weight"  # to_k_ipa.weight

    else:
        raise ValueError("Unexpected Face ID weight name found.")

    return retval

def model_memory_usage_in_bytes(model, include_gradients=False):
    total_bytes = 0
    for p in model.parameters():
        param_bytes = p.numel() * p.element_size()  # Number of elements * size of each element in bytes
        total_bytes += param_bytes
        if include_gradients and p.grad is not None:
            total_bytes += param_bytes  # Assuming the gradient is the same size as the parameter
    return total_bytes


def _files_in_model_dir(model_dir:str,
                        supported_extensions=[".pth", ".ckpt", ".safetensors"]):
    if os.path.exists(model_dir) is False:
        return list()
    
    files = os.listdir(model_dir)
    files = [f for f in files if os.path.splitext(f)[1].lower() in supported_extensions]
    files = sorted(files)
    return files


def load_ldm_model_paths(model_dir: str):
    files = _files_in_model_dir(
                model_dir,
                supported_extensions=[".ckpt", ".safetensors"])

    # Drop SDXL models
    files = [
        f for f in files 
            if (
                f.endswith("safetensors") and \
                    sd_model_type_from_safetensors(
                        os.path.join(model_dir, f)) != MODEL_TYPE_SDXL
                )
    ]
    # Drop refiner
    files = [f for f in files if f.upper().find("REFINER") < 0]

    # Drop inpaint models  TODO: Make this more robust
    files = [f for f in files if f.upper().find("INPAINT") < 0]
    return files

def load_ldm_inpaint_model_paths(model_dir: str):
    # FIX for SDXL
    files = _files_in_model_dir(
                model_dir,
                supported_extensions=[".ckpt", ".safetensors"])    
    files = [f for f in files if "inpaint" in f]

    # Drop SDXL models  TODO: Make this more robust
    files = [f for f in files if f.upper().find("XL") < 0]
    return files


def load_vae_model_paths(model_dir: str):
    """
    Get the list of VAE model file base names.
    The list element does not contain the directory part of the path.
    """
    return (_files_in_model_dir(model_dir))


def load_lora_model_paths(model_dir: str):
    """
    Get the list of lora model file base names.
    The list element does not contain the directory part of the path.
    """
    return _files_in_model_dir(model_dir)

def load_control_net_model_paths(model_dir: str):
    """
    Get the list of control net model file base names.
    The list element does not contain the directory part of the path.
    """
    return _files_in_model_dir(model_dir,
                               supported_extensions=[".bin", ".pth", ".ckpt", ".safetensors"])


def load_sdxl_ldm_model_paths(model_dir: str):
    files = _files_in_model_dir(
                model_dir,
                supported_extensions=[".safetensors"])

    # Filter out non-SDXL models
    files = [
        f for f in files 
            if (
                f.endswith("safetensors") and \
                    sd_model_type_from_safetensors(
                        os.path.join(model_dir, f)) == MODEL_TYPE_SDXL
                ) or \
            "refiner" in f
    ]

    # Drop inpaint models  TODO: Make this more robust
    files = [f for f in files if f.upper().find("INPAINT") < 0]

    return files


def load_sdxl_ldm_inpaint_model_paths(model_dir: str):
    files = _files_in_model_dir(
                model_dir,
                supported_extensions=[".safetensors"])
    files = [f for f in files if "inpaint" in f]
    files = [f for f in files if f.upper().find("XL") >= 0]

    # Drop SDXL models  TODO: Make this more robust
    files = [f for f in files if f.upper().find("XL") < 0]
    return files


def load_sdxl_vae_model_paths(model_dir: str):
    """
    Get the list of VAE model file base names.
    The list element does not contain the directory part of the path.
    """
    return (_files_in_model_dir(model_dir))


def load_sdxl_lora_model_paths(model_dir: str):
    """
    Get the list of lora model file base names.
    The list element does not contain the directory part of the path.
    """
    return _files_in_model_dir(model_dir)


def load_torch_model_paths(model_dir: str):
    """
    Get the list of pytorch model file base names.
    The list element does not contain the directory part of the path.
    """
    return _files_in_model_dir(
                model_dir,
                supported_extensions=[".pth", ".ckpt", ".safetensors", ".bin", ".pt"])


def load_model(model_path: str):
    if model_path.endswith("ckpt") or model_path.endswith("pth") or \
        model_path.endswith("bin") or model_path.endswith("pt"):
        logger.debug(f"Loading model from {model_path}")
        model_or_sd = torch.load(model_path, map_location="cpu")
    elif model_path.endswith("safetensors"):
        logger.debug(f"Loading safetensor-format model from {model_path}")
        model_or_sd = load_file(model_path, device="cpu")
    else:
        raise ValueError(f"Invalid model extension. Only ckpt, safetensors, pth, pt, bin are supported: {model_path}")
    return model_or_sd


def load_embedding(model_path: str) -> torch.Tensor:
    """
    Type 1:
    .pt format example (e.g. Foo.pt)
    ```
    {
        'string_to_token': {'*': 265},
        'string_to_param': {
            '*': tensor(
                    [
                        [-3.4567e-02, ...],
                        [-9.8765e-03, ...],
                        [ 1.2345e-02,  ... -4.5678e-03]])},
            'name': 'Foo',
            'step': None,
            'sd_checkpoint': None,
            'sd_checkpoint_name': None
        }
    }
    ```

    Type 2:
    {'emb_params': tensor(
        [
            [-0.0001, ...,  0.0002],
            ...,
            [-0.0003, ..., -0.0004]])}


    SDXL

    """   
    base_name = os.path.basename(model_path)
    file_extension = os.path.splitext(base_name)[1]
    if file_extension == ".safetensors":
        logger.info(f"Loading safetensor-format model from {model_path}")
        model = load_file(model_path, device="cpu")
    else:
        logger.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location="cpu")

    if isinstance(model, dict):
        if "string_to_param" in model and "*" in model["string_to_param"]:
            embedding = model["string_to_param"]["*"]
            logger.info("SD 1.5 Type 1 embedding detected:")
        elif "emb_params" in model:
            embedding = model["emb_params"]
            logger.info("SD 1.5 Type 2 embedding detected:")
        elif "clip_g" in model and "clip_l" in model:  # SDXL
            embedding = model
            logger.info("SDXL embedding detected:")
        else:
            logger.info("Unsupported dict-format embedding")
    else:
        logger.info("Unsupported non-dict-format embedding")       
        embedding = None

    return embedding


def load_lora(lora_file_full_path, model_type="SD 1.5", name_check=True):
    lora = load_model(lora_file_full_path)
    if isinstance(lora, dict) is False:
        logger.warning("LoRA is not a dict. Ignoring")
        return None

    if model_type == "SD 1.5" and name_check:
        for k in lora:
            if k not in LORA_WEIGHTS:
                logger.warning(f"Unsupported LoRA weight name is found {k}. Abort loading")
                return None

    # Check rank
    if model_type == "SD 1.5":
        w = lora['lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight']
    else: # sdxl
        w = lora['lora_unet_output_blocks_5_1_transformer_blocks_1_attn1_to_k.lora_down.weight']

    rank = w.shape[0]  # [4, 320]
    return lora, rank