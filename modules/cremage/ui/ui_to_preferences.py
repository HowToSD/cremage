"""
Update preferences from UI field values.
Note that the preferences values do not persist unless explicitly saved
by the user.
"""
import os
import logging
import sys

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.configs.preferences import save_user_config
from cremage.utils.gtk_utils import text_view_get_text

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

COMBO_BOX_FIELDS = [
    "auto_face_fix",
    "auto_face_fix_face_detection_method",
    "vae_model",
    "control_model",
    "ldm_model",
    "ldm_inpaint_model",
    "lora_model_1",
    "lora_model_2",
    "lora_model_3",
    "lora_model_4",
    "lora_model_5",
    "sdxl_vae_model",
    "sdxl_ldm_model",
    "sdxl_ldm_inpaint_model",
    "sdxl_lora_model_1",
    "sdxl_lora_model_2",
    "sdxl_lora_model_3",
    "sdxl_lora_model_4",
    "sdxl_lora_model_5",
    "refiner_sdxl_vae_model",
    "refiner_sdxl_ldm_model",
    "refiner_sdxl_lora_model_1",
    "refiner_sdxl_lora_model_2",
    "refiner_sdxl_lora_model_3",
    "refiner_sdxl_lora_model_4",
    "refiner_sdxl_lora_model_5",
    "pixart_sigma_ldm_model",
    "pixart_sigma_model_id",
    "sdxl_image_resolution",
    "sampler",
    "sdxl_sampler",
    "hires_fix_upscaler",
    "generator_model_type",
    "guider",
    "discretization"
]

INT_FIELDS = [
    "sampling_steps",
    "image_width",
    "image_height",
    "clip_skip",
    "batch_size",
    "number_of_batches",
    "seed",
    "sampler_order"
]

FLOAT_FIELDS = [
    "denoising_strength",
    "sdxl_refiner_strength",
    "cfg",
    "face_strength",
    "discretization_sigma_min",
    "discretization_sigma_max",
    "discretization_rho",
    "linear_prediction_guider_min_scale",
    "linear_prediction_guider_max_scale",
    "triangle_prediction_guider_min_scale",
    "triangle_prediction_guider_max_scale",
    "sampler_s_churn",
    "sampler_s_tmin",
    "sampler_s_tmax",
    "sampler_s_noise",
    "sampler_eta",
    "auto_face_fix_strength"
]

TEXT_VIEW_FIELDS = [
    "positive_prompt_pre_expansion",
    "negative_prompt_pre_expansion",
    "positive_prompt_expansion",
    "negative_prompt_expansion",
    "auto_face_fix_prompt"
]

CHECK_BUTTON_FIELDS = [
    "sdxl_use_refiner"
]

SKIP_FIELDS = [
    "positive_prompt_pre_expansion_history",
    "negative_prompt_pre_expansion_history",
    "positive_prompt_expansion_history",
    "negative_prompt_expansion_history"
]

def copy_ui_field_values_to_preferences(app:Gtk.Window) -> None:
    """
    Updates preferences based on fields

    Args:
        app (Gtk.Window): The application window instance
        widget: The button clicked
    """
    # Copy UI field values to preferences object
    for key, field in app.fields.items():
        if key in COMBO_BOX_FIELDS:
            if isinstance(field, Gtk.ComboBoxText):
                app.preferences[key] = field.get_active_text()
            elif isinstance(field, Gtk.ComboBox):  # This is a Cremage specific combobox that contains Entry
                if key == "sdxl_image_resolution":
                    if app.fields["generator_model_type"].get_child().get_text() == "SDXL":
                        resolution = field.get_child().get_text()
                        w, h = resolution.split("x")
                        app.preferences["image_width"] = w
                        app.preferences["image_height"] = h
                    else: # SD 1.5 mode, so ignore
                        continue
                else:
                    app.preferences[key] = field.get_child().get_text()
        elif isinstance(field, Gtk.ComboBoxText):  # bool
            app.preferences[key] = field.get_active_text() == "True"
        elif key in INT_FIELDS:
            if key in ["image_width", "image_height"] and app.fields["generator_model_type"].get_child().get_text() == "SDXL":
                continue  # Ignore in SDXL mode
            app.preferences[key] = int(field.get_text())
        elif key in FLOAT_FIELDS:
            app.preferences[key] = float(field.get_text())
        elif key in TEXT_VIEW_FIELDS:
            app.preferences[key] = text_view_get_text(field)
        elif key in CHECK_BUTTON_FIELDS:
            app.preferences[key] = field.get_active()
        elif key in SKIP_FIELDS:
            continue
        else:  # str
            app.preferences[key] = field.get_text()

    # app.preferences["control_weight"] = app.control_weight.get_text()  # FIXME

