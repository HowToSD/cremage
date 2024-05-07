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
    "vae_model",
    "control_model",
    "ldm_model",
    "ldm_inpaint_model",
    "lora_model_1",
    "lora_model_2",
    "lora_model_3",
    "lora_model_4",
    "lora_model_5",
    "sampler",
    "hires_fix_upscaler"
]

INT_FIELDS = [
    "sampling_steps",
    "image_width",
    "image_height",
    "clip_skip",
    "batch_size",
    "number_of_batches",
    "seed"
]

FLOAT_FIELDS = [
    "denoising_strength",
    "cfg",
    "face_strength"
]

TEXT_VIEW_FIELDS = [
    "positive_prompt_expansion",
    "negative_prompt_expansion"
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
                app.preferences[key] = field.get_child().get_text()
        elif isinstance(field, Gtk.ComboBoxText):  # bool
            app.preferences[key] = field.get_active_text() == "True"
        elif key in INT_FIELDS:
            app.preferences[key] = int(field.get_text())
        elif key in FLOAT_FIELDS:
            app.preferences[key] = float(field.get_text())
        elif key in TEXT_VIEW_FIELDS:
            app.preferences[key] = text_view_get_text(field)
        else:  # str
            app.preferences[key] = field.get_text()

    # app.preferences["control_weight"] = app.control_weight.get_text()  # FIXME

