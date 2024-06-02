"""
Defines handlers for model path fields.
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
from cremage.utils.gtk_utils import update_combo_box
from cremage.utils.ml_utils import load_ldm_model_paths
from cremage.utils.ml_utils import load_ldm_inpaint_model_paths
from cremage.utils.ml_utils import load_vae_model_paths
from cremage.utils.ml_utils import load_control_net_model_paths
from cremage.utils.ml_utils import load_lora_model_paths


from cremage.utils.ml_utils import load_sdxl_ldm_model_paths
from cremage.utils.ml_utils import load_sdxl_ldm_inpaint_model_paths
from cremage.utils.ml_utils import load_sdxl_vae_model_paths
from cremage.utils.ml_utils import load_sdxl_lora_model_paths

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

NUM_LORA_MODELS_SUPPORTED = 5


def update_ldm_model_name_value_from_ldm_model_dir(app:Gtk.Window) -> None:
    # Get the list of ldm model files
    app.ldm_model_names = ["None"] + load_ldm_model_paths(app.preferences["ldm_model_path"])

    # Check if the ldm model name exists
    if app.preferences["ldm_model"] not in app.ldm_model_names:
        app.preferences["ldm_model"] = "None"

def update_sdxl_ldm_model_name_value_from_sdxl_ldm_model_dir(app:Gtk.Window) -> None:
    # Get the list of ldm model files
    app.sdxl_ldm_model_names = ["None"] + load_sdxl_ldm_model_paths(app.preferences["sdxl_ldm_model_path"])

    # Check if the ldm model name exists
    if app.preferences["sdxl_ldm_model"] not in app.sdxl_ldm_model_names:
        app.preferences["sdxl_ldm_model"] = "None"


def update_ldm_inpaint_model_name_value_from_ldm_model_dir(app:Gtk.Window) -> None:
    # Get the list of ldm model files
    app.ldm_inpaint_model_names = ["None"] + load_ldm_inpaint_model_paths(app.preferences["ldm_model_path"])

    # Check if the ldm model name exists
    if app.preferences["ldm_inpaint_model"] not in app.ldm_inpaint_model_names:
        app.preferences["ldm_inpaint_model"] = "None"


def update_sdxl_ldm_inpaint_model_name_value_from_sdxl_ldm_model_dir(app:Gtk.Window) -> None:
    # Get the list of ldm model files
    app.sdxl_ldm_inpaint_model_names = ["None"] + load_sdxl_ldm_inpaint_model_paths(app.preferences["sdxl_ldm_model_path"])

    # Check if the ldm model name exists
    if app.preferences["sdxl_ldm_inpaint_model"] not in app.sdxl_ldm_inpaint_model_names:
        app.preferences["sdxl_ldm_inpaint_model"] = "None"


def on_ldm_model_path_changed(app:Gtk.Window) -> None:       
    update_ldm_model_name_value_from_ldm_model_dir(app)
    update_combo_box(
        app.fields["ldm_model"],
        app.ldm_model_names,
        app.ldm_model_names.index(app.preferences["ldm_model"]))        

    update_ldm_inpaint_model_name_value_from_ldm_model_dir(app)
    update_combo_box(
        app.fields["ldm_inpaint_model"],
        app.ldm_inpaint_model_names,
        app.ldm_inpaint_model_names.index(app.preferences["ldm_inpaint_model"]))     


def on_sdxl_ldm_model_path_changed(app:Gtk.Window) -> None:       
    update_sdxl_ldm_model_name_value_from_sdxl_ldm_model_dir(app)
    update_combo_box(
        app.fields["sdxl_ldm_model"],
        app.sdxl_ldm_model_names,
        app.sdxl_ldm_model_names.index(app.preferences["sdxl_ldm_model"]))        

    update_sdxl_ldm_inpaint_model_name_value_from_sdxl_ldm_model_dir(app)
    update_combo_box(
        app.fields["sdxl_ldm_inpaint_model"],
        app.sdxl_ldm_inpaint_model_names,
        app.sdxl_ldm_inpaint_model_names.index(app.preferences["sdxl_ldm_inpaint_model"]))     


def update_vae_model_name_value_from_vae_model_dir(app:Gtk.Window) -> None:
    # Get the list of vae model files
    app.vae_model_names = ["None"] + load_vae_model_paths(app.preferences["vae_model_path"])

    # Check if the vae model name exists
    if app.preferences["vae_model"] not in app.vae_model_names:
        app.preferences["vae_model"] = "None"


def update_sdxl_vae_model_name_value_from_sdxl_vae_model_dir(app:Gtk.Window) -> None:
    # Get the list of vae model files
    app.sdxl_vae_model_names = ["None"] + load_sdxl_vae_model_paths(app.preferences["sdxl_vae_model_path"])

    # Check if the vae model name exists
    if app.preferences["sdxl_vae_model"] not in app.sdxl_vae_model_names:
        app.preferences["sdxl_vae_model"] = "None"


def on_vae_model_path_changed(app:Gtk.Window) -> None:
    update_vae_model_name_value_from_vae_model_dir(app)
    update_combo_box(
        app.fields["vae_model"],
        app.vae_model_names,
        app.vae_model_names.index(app.preferences["vae_model"]))


def on_sdxl_vae_model_path_changed(app:Gtk.Window) -> None:
    update_sdxl_vae_model_name_value_from_sdxl_vae_model_dir(app)
    update_combo_box(
        app.fields["sdxl_vae_model"],
        app.sdxl_vae_model_names,
        app.sdxl_vae_model_names.index(app.preferences["sdxl_vae_model"]))


def update_control_model_name_value_from_control_model_dir(app:Gtk.Window) -> None:
    # Get the list of control model files
    app.control_model_names = ["None"] + load_control_net_model_paths(app.preferences["control_model_path"])

    # Check if the control model name exists
    if app.preferences["control_model"] not in app.control_model_names:
        app.preferences["control_model"] = "None"    


def on_control_model_path_changed(app:Gtk.Window) -> None:
    update_control_model_name_value_from_control_model_dir(app)
    update_combo_box(
        app.fields["control_model"],
        app.control_model_names,
        app.control_model_names.index(app.preferences["control_model"]))


def update_lora_model_name_value_from_lora_model_dir(app:Gtk.Window) -> None:
    # Get the list of lora model files
    app.lora_model_names = ["None"] + load_lora_model_paths(app.preferences["lora_model_path"])

    # Check if the lora model name exists
    for i in range(1, NUM_LORA_MODELS_SUPPORTED + 1):
        if app.preferences[f"lora_model_{i}"] not in app.lora_model_names:
            app.preferences[f"lora_model_{i}"] = "None"


def update_sdxl_lora_model_name_value_from_sdxl_lora_model_dir(app:Gtk.Window) -> None:
    # Get the list of lora model files
    app.sdxl_lora_model_names = ["None"] + load_sdxl_lora_model_paths(app.preferences["sdxl_lora_model_path"])

    # Check if the lora model name exists
    for i in range(1, NUM_LORA_MODELS_SUPPORTED + 1):
        if app.preferences[f"sdxl_lora_model_{i}"] not in app.sdxl_lora_model_names:
            app.preferences[f"sdxl_lora_model_{i}"] = "None"


def on_lora_model_path_changed(app:Gtk.Window) -> None:
    update_lora_model_name_value_from_lora_model_dir(app)
    for i in range(1, NUM_LORA_MODELS_SUPPORTED + 1):
        update_combo_box(
            app.fields[f"lora_model_{i}"],
            app.lora_model_names,
            app.lora_model_names.index(app.preferences[f"lora_model_{i}"])),


def on_sdxl_lora_model_path_changed(app:Gtk.Window) -> None:
    update_sdxl_lora_model_name_value_from_sdxl_lora_model_dir(app)
    for i in range(1, NUM_LORA_MODELS_SUPPORTED + 1):
        update_combo_box(
            app.fields[f"sdxl_lora_model_{i}"],
            app.sdxl_lora_model_names,
            app.sdxl_lora_model_names.index(app.preferences[f"sdxl_lora_model_{i}"])),

def update_all_model_paths(app:Gtk.Window) -> None:
    # FIXME.
    # app.preferences = load_user_config()
    # if app.preferences["ldm_model_path"] != app.preferences_prev["ldm_model_path"]:
    #     app.on_ldm_model_path_changed()
    # if app.preferences["vae_model_path"] != app.preferences_prev["vae_model_path"]:
    #     app.on_vae_model_path_changed()
    # if app.preferences["lora_model_path"] != app.preferences_prev["lora_model_path"]:
    #     app.on_lora_model_path_changed()
    # app.preferences_prev = app.preferences
    on_ldm_model_path_changed(app)
    on_vae_model_path_changed(app)
    on_control_model_path_changed(app)
    on_lora_model_path_changed(app)

    on_sdxl_ldm_model_path_changed(app)
    on_sdxl_vae_model_path_changed(app)
    on_sdxl_lora_model_path_changed(app)
