"""Override PixArt-Sigma model's weights with the custom model's weights.

Copyright (c) 2024 Hideyuki Inada
"""
from safetensors.torch import load_file
from typing import Dict, Any
import logging

import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_pixart_sigma_model_with_custom_model(model: torch.nn.Module, custom_model_path: str) -> torch.nn.Module:
    try:
        custom_model = load_file(custom_model_path)
    except Exception as e:
        logger.error(f"Failed to load custom model from {custom_model_path}: {e}")
        return model

    state_dict = model.state_dict()
    updated_params = 0

    for k, v in state_dict.items():
        if k in custom_model:
            state_dict[k] = torch.nn.Parameter(custom_model[k])
            updated_params += 1
            continue
        logger.info(f"Missing {k} in custom model")

    model.load_state_dict(state_dict)
    logger.info(f"Overrode {updated_params}/{len(state_dict)} parameters from the custom model {custom_model_path}")

    return model
