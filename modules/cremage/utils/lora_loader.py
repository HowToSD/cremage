"""
Util function to load multiple LoRAs.
"""
import os
import logging
import re

import torch

from .ml_utils import load_lora

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def load_loras(lora_paths:str, lora_weights:str, model_type="SD 1.5", name_check=True):
    """
    
    Args:
        lora_paths (str): A comma-separated list of full paths of LoRA files
        lora_weights (str): A comma-separated list of weight for each LoRA file
    Returns:
        A tuple of three lists. Each list contains:
          * lora model's state dict,
          * rank
          * weight
        If no lora is specified, the list will be an empty list.
    """
    if lora_paths is None or lora_paths == "":
        tmp_lora_model_list = []
        tmp_lora_weight_list = []
    else:
        tmp_lora_model_list = lora_paths.split(",")
        tmp_lora_weight_list = [float(v) for v in lora_weights.split(",")]

    # if len(tmp_lora_model_list) != len(tmp_lora_weight_list):
    #    raise ValueError("Number of lora models and weights do not match.")

    # Default initial value if no lora is used
    lora_ranks = []
    lora_weights = []
    loras = []  # model
    
    for i, lora_path in enumerate(tmp_lora_model_list):
        if len(lora_path) <= 0:
            continue
        print(f"Loading LoRA {i+1}: {lora_path}")

        lora, rank = load_lora(lora_path, model_type=model_type, name_check=name_check)
        lora_weight = tmp_lora_weight_list[i]
        lora_ranks.append(rank)
        lora_weights.append(lora_weight)
        loras.append(lora)

    return loras, lora_ranks, lora_weights


def map_non_standard_lora_key_to_standard_lora_key(k):
    """
    Maps a single non-standard LoRA key to a standard LoRA key.

    Non-standard LoRA key : lora_unet_down_blocks_1_attentions_0_proj_in.alpha
    Standard LoRA key: lora_unet_input_blocks_4_1_proj_in.alpha

   Examples

    a: Non-standard, b:standard
    ---------------------------
    a: lora_unet_down_blocks_1_attentions_0_proj_in.alpha
    b: lora_unet_input_blocks_4_1_proj_in.alpha

    a: lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.alpha
    b: lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.alpha

    Mid
    a: lora_unet_mid_block_attentions_0_proj_in.alpha
    b: lora_unet_middle_block_1_proj_in.alpha

    Up
    a: lora_unet_up_blocks_0_attentions_0_proj_in.alpha
    b: lora_unet_output_blocks_0_1_proj_in.alpha

    Parameters to be discarded:
        lora_te2_text_projection.alpha
        lora_te2_text_projection.lora_down.weight
        lora_te2_text_projection.lora_up.weight

    Mapping of block numbers
    Total 10 major blocks
    (Non-standard to Standard)

    Down
    1, 0    4, 1
    1, 1    5, 1
    2, 0    7, 1
    2, 1    8, 1

    Mid
    0       1

    Up
    0, 0    0, 1
    0, 1    1, 1
    0, 2    2, 1
    1, 0    3, 1
    1, 1    4, 1
    1, 2    5, 1
    """
    if k.startswith("lora_unet_down_blocks"):
        k2 = "lora_unet_input_blocks"
        match = re.search(r"_([0-9])_attentions_([0-9])_(.+)$", k)
        if match is None:
            raise ValueError(f"Unexpected down block name detected: {k}")
        first_digit = match.group(1)
        second_digit = match.group(2)
        last = match.group(3)
        if first_digit == "1" and second_digit == "0":
            block_num = "4_1"
        elif first_digit == "1" and second_digit == "1":
            block_num = "5_1"
        elif first_digit == "2" and second_digit == "0":
            block_num = "7_1"
        elif first_digit == "2" and second_digit == "1":
            block_num = "8_1"
        else:
            raise ValueError(f"Unexpected down block name detected: {k}")

        k2 = f"{k2}_{block_num}_{last}"

    elif k.startswith("lora_unet_mid_block"):
        if k.startswith("lora_unet_mid_block_attentions_0"):
            k2 = k.replace("lora_unet_mid_block_attentions_0_", "lora_unet_middle_block_1_")
        else:
            raise ValueError(f"Unexpected middle block name detected: {k}")
    elif k.startswith("lora_unet_up_blocks"):
        k2 = "lora_unet_output_blocks"
        match = re.search(r"_([0-9])_attentions_([0-9])_(.+)$", k)
        if match is None:
            raise ValueError(f"Unexpected down block name detected: {k}")
        first_digit = match.group(1)
        second_digit = match.group(2)
        last = match.group(3)
        if first_digit == "0" and second_digit == "0":
            block_num = "0_1"
        elif first_digit == "0" and second_digit == "1":
            block_num = "1_1"
        elif first_digit == "0" and second_digit == "2":
            block_num = "2_1"
        elif first_digit == "1" and second_digit == "0":
            block_num = "3_1"
        elif first_digit == "1" and second_digit == "1":
            block_num = "4_1"
        elif first_digit == "1" and second_digit == "2":
            block_num = "5_1"                    
        else:
            raise ValueError(f"Unexpected down block name detected: {k}")

        k2 = f"{k2}_{block_num}_{last}"
    else:
        k2 = k
    return k2

def map_non_standard_lora_keys_to_standard_lora_keys(sd):
    """
    Maps non-standard LoRA's keys standard LoRA keys.
    """
    SKIP_KEYS = [
        "lora_te2_text_projection.alpha",
        "lora_te2_text_projection.lora_down.weight",
        "lora_te2_text_projection.lora_up.weight"
    ]
    sample_key = "lora_unet_down_blocks_1_attentions_0_proj_in.alpha"
    if sample_key in sd:
        logger.info("Non-standard LoRA key was detected. Mapping to standard LoRA keys")
    else:  # Standard LoRA keys, continue without mapping
        return sd

    out = dict()
    for k, v in sd.items():
        if k in SKIP_KEYS:
            continue
        else:
            k2 = map_non_standard_lora_key_to_standard_lora_key(k)

        out[k2] = torch.nn.Parameter(v)
    return out


def load_loras_state_dict_into_custom_model_state_dict(lora_sds, model_sd):
    """
    Loads SDXL LoRA's state dict to SDXL model's state dict.

    There are LoRA(s) that use non-standard keys.
    Those LoRA's keys are mapped to standard LoRA keys first.
 
    Args:
        lora_sds (List[Dict[str, tensor]])
        model_sd (Dict[str, tensor])
    """
    if lora_sds is None or len(lora_sds) == 0:
        return model_sd

    sd = model_sd
    for i, lora_sd in enumerate(lora_sds):
        lora_sd = map_non_standard_lora_keys_to_standard_lora_keys(lora_sd)
        for k, v in lora_sd.items():
            model_k = map_sdxl_lora_weight_name_to_mode_weight_name(k, i)
            model_sd[model_k] = v

    return sd

def map_sdxl_lora_weight_name_to_mode_weight_name(sd_key:str, lora_index:int):
    """
    
    Args:
        sd_key (str): A single key for the weight in LoRA model
        lora_index (str): 0-based index of the LoRA models to be loaded as specified
            in the option object.
    """
    if "lora_unet" in sd_key:
        k = sd_key.replace("lora_unet_", "model.diffusion_model.")
        k = re.sub(r'_(\d+)_', r'.\1.', k)
        k = re.sub(r'_(\d+)', r'.\1', k)
        k = re.sub(r'(\d+)_', r'\1.', k)
        k = k.replace(".lora", "_lora")
        k = k.replace("to_", "")
        k = k.replace(".alpha", "_lora_alpha")
        k = k.replace("out.0_lora", "out_lora")
        k = k.replace("ff_net", "ff.net")
        k = k.replace("net.2", "net_2")
        if k.endswith("alpha"):
            k = k + f"s.{lora_index}"
        elif k.endswith("weight"):
            pos = k.rfind("weight")
            k = f"{k[:pos-1]}s.{lora_index}.weight"
        else:
            raise ValueError(f"Unexpected weight name: {k}")
    elif "lora_te1" in sd_key:
        k = sd_key.replace("lora_te1_text_model_encoder_layers_",
                      "conditioner.embedders.0.transformer.text_model.encoder.layers.")

        # 0_self_attn_q_proj.alpha to 0.self_attn.q_lora_alphas.0
        # 0_self_attn_q_proj.lora_down.weight to 0.self_attn.q_lora_downs.0.weight
        k = k.replace("_mlp_", ".mlp.")
        k = k.replace("_self_attn_", ".self_attn.")
        k = k.replace("proj.alpha", f"lora_alphas.{lora_index}")
        k = k.replace("_proj.lora_down.weight", f"_lora_downs.{lora_index}.weight")
        k = k.replace("_proj.lora_up.weight", f"_lora_ups.{lora_index}.weight")
        k = k.replace("fc1.alpha", f"fc1_lora_alphas.{lora_index}")
        k = k.replace("fc1.lora_down.weight", f"fc1_lora_downs.{lora_index}.weight")
        k = k.replace("fc1.lora_up.weight", f"fc1_lora_ups.{lora_index}.weight")
        k = k.replace("fc2.alpha", f"fc2_lora_alphas.{lora_index}")
        k = k.replace("fc2.lora_down.weight", f"fc2_lora_downs.{lora_index}.weight")
        k = k.replace("fc2.lora_up.weight", f"fc2_lora_ups.{lora_index}.weight")

    elif "lora_te2" in sd_key:
        k = sd_key.replace("lora_te2_text_model_encoder_layers_",
                      "conditioner.embedders.1.model.transformer.resblocks.")
        # MLP
        # _mlp_fc1.alpha to .mlp_0_lora_alphas.0
        k = k.replace("_mlp_fc1.alpha", f".mlp_0_lora_alphas.{lora_index}")
        # _mlp_fc2.alpha to .mlp_2_lora_alphas.0
        k = k.replace("_mlp_fc2.alpha", f".mlp_2_lora_alphas.{lora_index}")
        # _mlp_fc1.lora_down.weight to .mlp_0_lora_downs.0.weight
        k = k.replace("_mlp_fc1.lora_down.weight", f".mlp_0_lora_downs.{lora_index}.weight")
        # _mlp_fc1.lora_up.weight to .mlp_0_lora_ups.0.weight
        k = k.replace("_mlp_fc1.lora_up.weight", f".mlp_0_lora_ups.{lora_index}.weight")
        # _mlp_fc2.lora_down.weight to .mlp_2_lora_downs.0.weight
        k = k.replace("_mlp_fc2.lora_down.weight", f".mlp_2_lora_downs.{lora_index}.weight")
        # _mlp_fc2.lora_up.weight to .mlp_2_lora_ups.0.weight
        k = k.replace("_mlp_fc2.lora_up.weight", f".mlp_2_lora_ups.{lora_index}.weight")

        # ATTN
        # _self_attn_q_proj.alpha to .attn.q_lora_alphas.0
        k = k.replace("_self_attn_", f".attn.")
        # q_proj.alpha to q_lora_alphas.0
        k = k.replace("_proj.alpha", f"_lora_alphas.{lora_index}")
        # q_proj.lora_down.weight to q_lora_downs.0.weight
        k = k.replace("_proj.lora_down.weight", f"_lora_downs.{lora_index}.weight")
        k = k.replace("_proj.lora_up.weight", f"_lora_ups.{lora_index}.weight")

    else:
        raise ValueError(f"Unexpected weight name: {sd_key}")
    return k
