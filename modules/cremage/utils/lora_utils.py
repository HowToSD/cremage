"""
TODO: Clean up this code as processing logic is now too complicated.

Util functions related to LoRA support

Som examples of SD 1.5 weights related to LoRA:
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q.weight
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k.weight
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_v.weight
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.weight
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.bias

LoRA weights that are added for Cremage model (for 3 LoRA case)
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.q_lora_downs.0.weight
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.q_lora_downs.1.weight
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.q_lora_downs.2.weight
model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.q_lora_ups.0.weight

model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj_lora_downs.0.weight
model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj_lora_alphas.0
model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net_2_lora_downs.0.weight
model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net_2_lora_alphas.0
"""
import os
import sys
from typing import Dict, Any

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.sd15_lora_weight_list import SD15_LORA_WEIGHT_LIST

LORA_WEIGHTS = set(filter(lambda e: len(e) > 0, SD15_LORA_WEIGHT_LIST.split("\n")))


def is_valid_lora_weight_name(name: str) -> bool:
    """
    Checks if the specified name is a valid LoRA (Low-Rank Adaptation) weight name.

    Args:
        name (str): The LoRA weight name to be tested for validity.

    Returns:
        bool: True if the name is valid for a LoRA weight, False otherwise.
    """
    return name in LORA_WEIGHTS

def _parse_sd_weight_for_lora_weight(name: str) -> str:
    if name.startswith("model.diffusion_model"):
        return _parse_sd_unet_weight_for_lora_weight(name)
    elif name.startswith("cond_stage_model"):
        return _parse_sd_clip_weight_for_lora_weight(name)
    else:
        raise ValueError(f"Invalid SD weight name: {name}")


def _parse_sd_clip_weight_for_lora_weight(name: str) -> str:
    """
    Converts SD1.5 model weight name to a corresponding LoRA weight name.
    Raises ValueError is SD1.5 is not a valid SD name that has the corresponding LoRA
    weight.

    Source examples:
    cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1_lora_downs.0.bias
    cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_lora_downs.0.bias
    cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_lora_alphas.0

    Args:
        name (str): The SD 1.5 weight name.
    Returns:
        str: The corresponding LoRA weight name.

    References:
    https://howtosd.com/parameter-list-of-stable-diffusion-model-v1-5-pruned-emaonly-safetensors/
    """
    if name.startswith("cond_stage_model.transformer.text_model.encoder.layers") is False:
        raise ValueError("Invalid weight name")
    
    name_list = name.split(".")[5:] # "0.mlp.fc1_lora_downs.0.bias"
    layer_index = name_list[0]
    layer_type = name_list[1]  # mlp or self_attn
    qkvoutfc = name_list[2].split("_")[0]
    downupalpha = name_list[2].split("_")[2][:-1]
    lora_index = name_list[3]

    if layer_index not in [str(i) for i in range(12)]:
        raise ValueError(f"Invalid layer index: {name}")
    if layer_type not in ["mlp", "self_attn"]:
        raise ValueError(f"Invalid layer type: {name}")
    if qkvoutfc not in ["q", "k", "v", "out", "fc1","fc2"]:
        raise ValueError(f"Invalid projection target type: {name}")
    if downupalpha not in ["down", "up", "alpha"]:
        raise ValueError(f"Invalid projection direction or not alpha: {downupalpha} in {name}")
    try:
        lora_index_num = int(lora_index)
    except:
        raise ValueError(f"Invalid lora index: {name}")

    return {
        "block_type": layer_type,  # "mlp", "self_attn"
        "block_num": layer_index,  # "0" to "11"
        "qkvout": qkvoutfc,  # "q", "k", "v", "out", "fc1" or "fc2"
        "downup": downupalpha,  # "down", "up" or "alpha"
        "lora_index": lora_index,
        "dict_type": "clip"
    }


def _parse_sd_unet_weight_for_lora_weight(name: str) -> str:
    """
    Converts SD1.5 model weight name to a corresponding LoRA weight name.
    Raises ValueError is SD1.5 is not a valid SD name that has the corresponding LoRA
    weight.

    Args:
        name (str): The SD 1.5 weight name.
    Returns:
        str: The corresponding LoRA weight name.

    References:
    https://howtosd.com/parameter-list-of-stable-diffusion-model-v1-5-pruned-emaonly-safetensors/
    """
    if name.find("_lora_") < 0:
        raise ValueError("Invalid weight name")
    
    name_list = name.split(".")

    block_type = None
    if name_list[2].startswith("input_blocks"):
        block_type = "input"
        block_num = name_list[3]
        sub_block_num = name_list[4]
        name_list = name_list[5:]
    elif name_list[2].startswith("middle_block"):
        block_type = "middle"
        block_num = name_list[3]
        sub_block_num = None
        name_list = name_list[4:]
    elif name_list[2].startswith("output_blocks"):
        block_type = "output"
        block_num = name_list[3]
        sub_block_num = name_list[4]
        name_list = name_list[5:] 
    else:
        raise ValueError(f"Invalid weight name:{name}")
    
    if name_list[0].startswith("proj_in_lora") or \
       name_list[0].startswith("proj_out_lora"):
        # model.diffusion_model.middle_block.1.proj_in_lora_downs.0.weight
        attention_block = name_list[0]  # Not used
    elif name_list[0] == "transformer_blocks":
        # model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.q_lora_downs.0.weight
        attention_block = None
        if name_list[2] not in ("attn1", "attn2", "ff"):
            raise ValueError(f"Invalid weight name:{name_list[2]}")
        attention_block = name_list[2]
        name_list = name_list[3:]
    else:
        raise ValueError(f"Invalid weight name:{name}")
 
    if name_list[-1] == "weight":
        lora_index = int(name_list[-2])  
    elif name_list[-2].endswith("alphas"):
        lora_index = int(name_list[-1])  
    else:
        raise ValueError(f"Invalid weight name: {name}")

    # split q_lora_downs
    if attention_block == "ff":
        # net.0.proj_lora_downs.0.weight
        # net.0.proj_lora_alphas.0
        # net_2_lora_downs.0.weight
        # net_2_lora_alphas.0
        if name_list[0].startswith("net") is False:
            raise ValueError(f"Missing net after ff. Found: {name[0]}")
        name = ".".join(name_list)
        name = name.replace(".", "_").replace("proj_lora", "lora")
        name_list = name.split("_")
        qkvout = f"{name_list[0]}_{name_list[1]}"  # net_0_proj, net_2
        if qkvout == "net_0":
            qkvout = "net_0_proj"

        downupalpha = name_list[3][:-1]
    else:
        qkvout_downup = name_list[0].split("_")
        qkvout = qkvout_downup[0]  # q, k, v, out, proj
        if qkvout == "proj":
            qkvout = "proj_" + qkvout_downup[1]
            downupalpha = qkvout_downup[3][:-1]  # up, down, alpha
        else:
            if len(qkvout_downup) < 3:
                raise ValueError(f"Unexpected value for qkvout_downup: {name}")
            downupalpha = qkvout_downup[2][:-1]  # up, down, alpha

    return {
        "block_type": block_type,  # "input", "middle", "output"
        "block_num": block_num,  # "6" for input & output, "1." for middle
        "sub_block_num": sub_block_num,  # 0, 1 for input & output, None for middle
        "attention_block": attention_block, # "attn1", "attn2", "ff"
        "qkvout": qkvout,  # "q", "k", "v" or "out"
        "downup": downupalpha,  # "down" or "up"
        "lora_index": lora_index,
        "dict_type": "unet"
    }


def map_sd_down_block_to_lora(block:str)->str:
    mapping = {
        1:("0", "0"),
        2:("0", "1"),
        4:("1", "0"),
        5:("1", "1"),
        7:("2", "0"),
        8:("2", "1")
    }
    return mapping[int(block)]


def map_sd_up_block_to_lora(block:str)->str:
    mapping = {
        3:("1", "0"),
        4:("1", "1"),
        5:("1", "2"),
        6:("2", "0"),
        7:("2", "1"),
        8:("2", "2"),
        9:("3", "0"),
        10:("3", "1"),
        11:("3", "2")
    }
    return mapping[int(block)]

def _generate_lora_weight_name_from_parsed_sd_weight(d: Dict[str, Any])-> str:
    if d["dict_type"] == "clip":
        return _generate_lora_weight_name_from_parsed_sd_clip_weight(d)
    elif d["dict_type"] == "unet":
        return _generate_lora_weight_name_from_parsed_sd_unet_weight(d)
    raise ValueError(f"Invalid input: {d['dict_type']}")


def _generate_lora_weight_name_from_parsed_sd_clip_weight(d: Dict[str, Any])-> str:
    """
    Target examples:
    'lora_te_text_model_encoder_layers_8_self_attn_v_proj.alpha',
    'lora_te_text_model_encoder_layers_8_self_attn_v_proj.lora_down.weight',
    'lora_te_text_model_encoder_layers_9_mlp_fc1.alpha',
    'lora_te_text_model_encoder_layers_9_mlp_fc1.lora_up.weight',
    """
    if d["block_type"] == "mlp":
        if d["downup"] == "alpha":
            retval = f'lora_te_text_model_encoder_layers_{d["block_num"]}_mlp_{d["qkvout"]}.alpha'
        else:
            retval = f'lora_te_text_model_encoder_layers_{d["block_num"]}_mlp_{d["qkvout"]}.lora_{d["downup"]}.weight'
    
    elif d["block_type"] == "self_attn":
        if d["downup"] == "alpha":
            retval = f'lora_te_text_model_encoder_layers_{d["block_num"]}_self_attn_{d["qkvout"]}_proj.alpha'
        else:
            retval = f'lora_te_text_model_encoder_layers_{d["block_num"]}_self_attn_{d["qkvout"]}_proj.lora_{d["downup"]}.weight'

    if retval not in LORA_WEIGHTS:
        raise ValueError(f"Mapped LoRA weight not found in the supported LoRA weight set: {name}")

    return retval

def _generate_lora_weight_name_from_parsed_sd_unet_weight(d: Dict[str, Any])-> str:
    """
    Generates LoRA weight name from parsed SD weight in a dictionary.

    Example output:
    lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight
    
    Block number mapping
    
    Down

    SD                 LoRA
    ====================================
    block   subblock   blocks attentions
    ------------------------------------
    loop 1  1.1        0      0
            2.1        0      1
    loop 2  4.1        1      0
            5.1        1      1
    Loop 3  7.1        2      0
            8.1        2      1
    Loop 4  No                
            
    Middle
    SD                 LoRA
    ====================================
    block              attentions
    ------------------------------------   
    1                  0

    Output block
    SD                 LoRA
    ====================================
    block   subblock   blocks attentions
    ------------------------------------   
    3.1                1      0
    4.1                1      1
    5.1                1      2
    6.1                2      0
    7.1                2      1
    8.1                2      2
    9.1                3      0
    10.1               3      1
    11.1               3      2
    """
    if d["qkvout"] == "out":
        d["qkvout"] += "_0"

    if d["block_type"] == "input" or d["block_type"] == "output":

        if d["block_type"] == "input":
            ud = "down"
            b, a = map_sd_down_block_to_lora(d["block_num"])
        else:
            ud = "up"
            b, a = map_sd_up_block_to_lora(d["block_num"])

        if d["attention_block"] == "ff":
            """
            lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.alpha
            lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight
            lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight
            lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2.alpha
            lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight
            lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight
            """
            prefix = f"lora_unet_{ud}_blocks_{b}_attentions_{a}_transformer_blocks_0_ff_"
            name = prefix + d["qkvout"] # net_0_proj or net_2
            if d["downup"] == "alpha":
                name = name + ".alpha"
            else:
                name = f'{name}.lora_{d["downup"]}.weight'
        else:
            if d["qkvout"] in ("proj_in", "proj_out"):
                if d["downup"] == "alpha":
                    name = f'lora_unet_{ud}_blocks_{b}_attentions_{a}_{d["qkvout"]}.{d["downup"]}'
                else:
                    name = f'lora_unet_{ud}_blocks_{b}_attentions_{a}_{d["qkvout"]}.lora_{d["downup"]}.weight'
            else:
                if d["downup"] == "alpha":
                    name = f'lora_unet_{ud}_blocks_{b}_attentions_{a}_transformer_blocks_0_{d["attention_block"]}_to_{d["qkvout"]}.{d["downup"]}'
                else:
                    name = f'lora_unet_{ud}_blocks_{b}_attentions_{a}_transformer_blocks_0_{d["attention_block"]}_to_{d["qkvout"]}.lora_{d["downup"]}.weight'
    elif d["block_type"] == "middle":
        if d["attention_block"] == "ff":
            prefix = f"lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_"
            name = prefix + d["qkvout"] # net_0_proj or net_2
            if d["downup"] == "alpha":
                name = name + ".alpha"
            else:
                name = f'{name}.lora_{d["downup"]}.weight'
        else:
            if d["qkvout"] in ("proj_in", "proj_out"):
                if d["downup"] == "alpha":
                    name = f'lora_unet_mid_block_attentions_0_{d["qkvout"]}.{d["downup"]}'       
                else:
                    name = f'lora_unet_mid_block_attentions_0_{d["qkvout"]}.lora_{d["downup"]}.weight'
            else:
                if d["downup"] == "alpha":
                    name = f'lora_unet_mid_block_attentions_0_transformer_blocks_0_{d["attention_block"]}_to_{d["qkvout"]}.{d["downup"]}'       
                else:
                    name = f'lora_unet_mid_block_attentions_0_transformer_blocks_0_{d["attention_block"]}_to_{d["qkvout"]}.lora_{d["downup"]}.weight'
    else:
        ValueError("Unexpected name")

    if name not in LORA_WEIGHTS:
        raise ValueError(f"Mapped LoRA weight not found in the supported LoRA weight set: {name}")

    return name


def sd_weight_to_lora_weight(w:str) -> str:
    """
    Generates LoRA weight name from an SD weight name.

    Args:
        w (str): SD 1.5 weight name. The weight name is for the Cremage-specific
                 LoRA weight name with the 0-based LoRA index.
    Returns:
        str: LoRA weight name.
    """
    weight_dict = _parse_sd_weight_for_lora_weight(w)
    lora_weight = _generate_lora_weight_name_from_parsed_sd_weight(weight_dict)
    return lora_weight


if __name__ == "__main__":
    names = ["cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1_lora_downs.0.bias",
    "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_lora_downs.0.bias",
    "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_lora_alphas.0",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.q_lora_downs.0.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj_lora_downs.0.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj_lora_alphas.0",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net_2_lora_downs.0.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net_2_lora_alphas.0"]
    for name in names:
        print(sd_weight_to_lora_weight(name))
