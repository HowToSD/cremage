"""
Util function to load multiple LoRAs.
"""
import os
import logging
import re

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


def load_loras_state_dict_into_custom_model_state_dict(lora_sds, model_sd):
    """
    
    Args:
        lora_sds (List[Dict[str, tensor]])
        model_sd (Dict[str, tensor])
    """
    if lora_sds is None or len(lora_sds) == 0:
        return model_sd
    
    sd = model_sd
    for i, lora_sd in enumerate(lora_sds):
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
        logger.warning(f"Unexpected key found: {sd_key}")
        k = sd_key # FIXME
    return k

