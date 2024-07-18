"""
Utility functions for safetensors.

Copyright (c) 2024 Hideyuki Inada.
"""
import os
import logging

from safetensors import safe_open

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Model type definitions
MODEL_TYPE_UNKNOWN = 0
MODEL_TYPE_SD_1_5 = 1
MODEL_TYPE_SDXL = 2


def sd_model_type_from_safetensors(file_path: str) -> int:
    """
    Analyzes the input model and determines the model type.

    Args:
        file_path (str): The full path of the model.
    Returns
        Model type (int)
    """
    with safe_open(file_path, framework='pt') as f:
        sdxl_tensor_name = "conditioner.embedders.0.transformer.text_model.encoder.layers.2.self_attn.q_proj.bias"
        sd15_tensor_name = "model.diffusion_model.input_blocks.2.0.in_layers.0.bias"
        if sdxl_tensor_name in f.keys():
            return MODEL_TYPE_SDXL
        elif sd15_tensor_name in f.keys():
            return MODEL_TYPE_SD_1_5
        else:
            return MODEL_TYPE_UNKNOWN

def is_supported_pixart_sigma_custom_model(file_path:str) -> bool:
    with safe_open(file_path, framework='pt') as f:
        key = "transformer_blocks.0.attn1.to_k.bias"
        if key in f.keys():
            return True
        else:
            return False


if __name__ == "__main__":
    dir_name = "/media/pup/ssd2/recoverable_data/sd_models/Stable-diffusion"
    dirs = os.listdir(dir_name)
    for f in dirs:
        p = os.path.join(dir_name, f)
        if p.endswith("safetensors"):
            info = sd_model_type_from_safetensors(p)
            print(f"{os.path.basename(p)}: {info}")
