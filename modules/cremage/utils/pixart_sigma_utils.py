"""Override PixArt-Sigma model's weights with the custom model's weights.

Copyright (c) 2024 Hideyuki Inada
"""
from safetensors.torch import load_file
from typing import Dict, Any
import logging

import torch
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
MODEL_ID_LIST = [
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "dataautogpt3/PixArt-Sigma-900M"
]


def pixart_sigma_model_id_cb_changed(app, combo):
    pixart_sigma_model_id = combo.get_child().get_text()
    toggle_pixart_sigma_ui(app, pixart_sigma_model_id)


def toggle_pixart_sigma_ui(app:Gtk.Window, model_id):
    if model_id == DEFAULT_MODEL_ID:
        app.pixart_sigma_fields_labels["pixart_sigma_ldm_model"].show()
        app.fields["pixart_sigma_ldm_model"].show()
        
    else:
        app.pixart_sigma_fields_labels["pixart_sigma_ldm_model"].hide()
        app.fields["pixart_sigma_ldm_model"].hide()


def update_pixart_sigma_model_id_value(app:Gtk.Window) -> None:
    app.pixart_sigma_model_ids = ["None"] + MODEL_ID_LIST

    if app.preferences["pixart_sigma_model_id"] not in MODEL_ID_LIST:
        app.preferences["pixart_sigma_model_id"] = "None"

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
