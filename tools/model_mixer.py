"""
A tool to mix model weights.

Current limitations:
1. lazy loading is not supported, so it will require a large amount of memory
   if you want to mix multiple models.
2. Precision is set to fp16 only.
3. Keys of the first model is used as the keys of the combined model.
   If there are any missing keys that exist in subsequent models, those
   values are not set in the resultant model.

Copyright (c) 2024 Hideyuki Inada.
"""
import os
import sys
import logging

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__),  ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.const.const import GENERATOR_MODEL_TYPE_LIST
from cremage.utils.gtk_utils import create_combo_box_typeahead
from cremage.utils.gtk_utils import create_center_aligned_button
from cremage.utils.gtk_utils import show_error_dialog, show_info_dialog
from cremage.utils.ml_utils import load_ldm_model_paths, load_sdxl_ldm_model_paths
from cremage.utils.ml_utils import load_vae_model_paths, load_sdxl_vae_model_paths

from cremage.utils.ml_utils import load_model, save_model

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

NUM_MODELS = 5

class ModelMixer(Gtk.Window):

    def __init__(self,
                 ldm_model_dir=None,
                 sdxl_ldm_model_dir=None,
                 vae_model_dir=None,
                 sdxl_vae_model_dir=None,
                 callback=None):
        Gtk.Window.__init__(self, title="Model Mixer")
        self.callback = callback
        generator_model_type_list = GENERATOR_MODEL_TYPE_LIST
        self.ldm_model_dir = ldm_model_dir
        self.sdxl_ldm_model_dir = sdxl_ldm_model_dir
        self.sd_vae_model_dir = vae_model_dir
        self.sdxl_vae_model_dir = sdxl_vae_model_dir

        # self.set_default_size(400, 600)

        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.vbox.set_margin_start(10)  # Margin on the left side
        self.vbox.set_margin_end(10)    # Margin on the right side
        self.vbox.set_margin_top(10)    # Margin on the top
        self.vbox.set_margin_bottom(10) # Margin on the bottom
        self.add(self.vbox)
        self.set_resizable(False)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hbox.set_margin_start(10)  # Margin on the left side
        hbox.set_margin_end(10)    # Margin on the right side
        hbox.set_margin_top(10)    # Margin on the top
        hbox.set_margin_bottom(10) # Margin on the bottom
        label = Gtk.Label(label="Model type:")
        hbox.pack_start(label, False, False, 0)
        generator_model_type = \
            create_combo_box_typeahead(generator_model_type_list, 0)
        generator_model_type.connect("changed", self.generator_type_changed)

        hbox.pack_start(generator_model_type, False, False, 0)
        self.vbox.pack_start(hbox, False, False, 0)

        # Set to SD 1.5 first
        # LDM
        model_name_list = load_ldm_model_paths(self.ldm_model_dir)
        self.model_dir = self.ldm_model_dir
        # VAE
        vae_model_name_list = load_vae_model_paths(self.sd_vae_model_dir)
        self.vae_model_dir = self.sd_vae_model_dir
        # Set to UI
        self.vbox_model_list = self.setup_ui(
            model_name_list,
            vae_model_name_list)
        self.vbox.pack_start(self.vbox_model_list, False, False, 0)

    def setup_ui(self, model_name_list, vae_model_name_list):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        # Grid
        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)

        # Set margins for the grid
        grid.set_margin_start(10)  # Margin on the left side
        grid.set_margin_end(10)    # Margin on the right side
        grid.set_margin_top(10)    # Margin on the top
        grid.set_margin_bottom(10) # Margin on the bottom

        model_names = [""] + model_name_list
        vae_model_names = [""] + vae_model_name_list
        
        self.fields = {}
        for i in range(NUM_MODELS):
            self.fields[f"model_{i+1}"] = create_combo_box_typeahead(model_names)
            self.fields[f"weight_{i+1}"] = Gtk.Entry(text="0")

        # "vae_model": create_combo_box_typeahead(app.vae_model_names, app.vae_model_names.index(app.preferences["vae_model"])),

        # Add fields to the grid
        width = 40
        grid.attach(Gtk.Label(label="Model path"), 1, 0, width, 1)  # field, left, top, width, height
        grid.attach(Gtk.Label(label="Weight"), width + 1, 0, 1, 1)

        row = 1
        for _, (label_text, field) in enumerate(self.fields.items()):
            i = row
            if label_text.startswith("weight") is False:
                label = Gtk.Label(label=label_text.replace("_", " ").capitalize(), halign=Gtk.Align.START)
                grid.attach(label, 0, i, 1, 1)  # field, left, top, width, height
                grid.attach(field, 1, i, width, 1)

            else:  # weight
                field.set_alignment(1.0)  # right-align
                grid.attach(field, width + 1, i, 1, 1)
                field.set_size_request(10, -1)
                row += 1

        vbox.pack_start(grid, False, True, 0)

        create_combo_box_typeahead(model_names)
        # VAE to bake in
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hbox.set_margin_start(10)  # Margin on the left side
        hbox.set_margin_end(10)    # Margin on the right side
        hbox.set_margin_top(10)    # Margin on the top
        hbox.set_margin_bottom(10) # Margin on the bottom
        label = Gtk.Label(label="VAE to bake")
        self.vae_cb = create_combo_box_typeahead(vae_model_names)
        hbox.pack_start(label, False, False, 0)  # Do not expand but take up allocated space. No margin
        hbox.pack_start(self.vae_cb, True, True, 0)
        vbox.pack_start(hbox, False, True, 0)

        # Target file name
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hbox.set_margin_start(10)  # Margin on the left side
        hbox.set_margin_end(10)    # Margin on the right side
        hbox.set_margin_top(10)    # Margin on the top
        hbox.set_margin_bottom(10) # Margin on the bottom
        label = Gtk.Label(label="Target model name")
        self.target_model_name = Gtk.Entry()
        hbox.pack_start(label, False, False, 0)  # Do not expand but take up allocated space. No margin
        hbox.pack_start(self.target_model_name, True, True, 0)
        vbox.pack_start(hbox, False, True, 0)
        
        # Button
        ok_button, wrapper = create_center_aligned_button(label="Create")
        ok_button.connect("clicked", self.on_create_button_clicked)
        vbox.pack_start(wrapper, 
                        False,  # Does not expand
                        False,  # Does not take up the space of the cell allocated
                        10)
        return vbox

    def generator_type_changed(self, combo):
        generator_model_type = combo.get_child().get_text()
        self.vbox.remove(self.vbox_model_list)
        self.vbox_model_list = None

        if generator_model_type == "SD 1.5":
             model_name_list = load_ldm_model_paths(self.ldm_model_dir)
             self.model_dir = self.ldm_model_dir
             vae_model_name_list = load_vae_model_paths(self.sd_vae_model_dir)
             self.vae_model_dir = self.sd_vae_model_dir
        elif generator_model_type == "SDXL":
             model_name_list = load_sdxl_ldm_model_paths(self.sdxl_ldm_model_dir)
             self.model_dir = self.sdxl_ldm_model_dir
             vae_model_name_list = load_sdxl_vae_model_paths(self.sdxl_vae_model_dir)
             self.vae_model_dir = self.sdxl_vae_model_dir
        else:
            raise ValueError("Unexpected generator type")

        self.vbox_model_list = self.setup_ui(model_name_list, vae_model_name_list)
        self.vbox.pack_start(self.vbox_model_list, False, False, 0)
        self.show_all()

    def on_create_button_clicked(self, widget):
        # Get VAE path (optional)
        vae_model_name = self.vae_cb.get_child().get_text()
        if vae_model_name:
            vae_path = os.path.join(self.vae_model_dir, vae_model_name)
            if os.path.exists(vae_path) is False:
                show_error_dialog(win=self, message=f"{vae_path} not found.")
                return
        else:
            vae_path = None

        # Get target model name
        model_name = self.target_model_name.get_text()
        base_name = os.path.basename(model_name)
        if len(model_name) == 0:
            show_error_dialog(win=self, message="Specify the target model name")
            return
        
        if base_name != model_name:
            show_error_dialog(win=self, message=f"{model_name} contains a directory. Specify the base name only")
            return
        
        if model_name.endswith(".safetensors") is False:
            model_name += ".safetensors"

        target_path = os.path.join(self.model_dir, model_name)
        if os.path.exists(target_path):
            show_error_dialog(win=self, message=f"{target_path} already exists. Choose a different name.")
            return

        # Grab values from UI
        model_path_list = list()
        weight_list = list()
        for i in range(NUM_MODELS):
            weight = self.fields[f"weight_{i+1}"].get_text()
            if weight == 0:
                continue
            try:
                weight = float(weight)
            except:
                logger.warn(f"Ignoring invalid weight: {weight}")
                continue

            model_name = self.fields[f"model_{i+1}"].get_child().get_text()
            if len(model_name) <= 0:
                continue

            model_path = os.path.join(self.model_dir, model_name)
            if os.path.exists(model_path) is False:
                continue

            model_path_list.append(model_path)
            weight_list.append(weight)

        if not len(model_path_list):
            return

        if len(model_path_list) == 1:
            show_error_dialog(win=self, message="Only 1 model is specified.")
            return
        
        # Normalize weight
        weight_sum = sum(weight_list)
        if weight_sum == 0:
            show_error_dialog(win=self, message="All the weights are set to 0.")
            return
        
        weight_list = [w / weight_sum for w in weight_list]

        sd_list = list()
        for m in model_path_list:
            model = load_model(m)
            if isinstance(model, dict) is False:
                sd = model.state_dict()
            else:
                sd = model
            sd_list.append(sd)

        sd_target = dict()
        for k, v in sd_list[0].items():
            if v.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                continue

            v *= weight_list[0]
            for i, sd in enumerate(sd_list[1:]):
                if k not in sd:
                    v2 = torch.zeros_like(v)
                else:
                    v2 = sd[k]
                weight2 = weight_list[i+1]
                v += v2 * weight2
            sd_target[k] = v.half()

        # Bake vae
        if vae_path:
            vae_model = load_model(vae_path)
            if not isinstance(vae_model, dict):
                vae_model = vae_model.state_dict()
            if "state_dict" in vae_model:
                vae_model = vae_model["state_dict"]

            for k, v in vae_model.items():
                if isinstance(v, int) or isinstance(v, str):
                    continue
                if v.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                    half = False
                else:
                    half = True
                k_target = "first_stage_model." + k
                if k_target not in sd_target:
                    logger.info(f"{k_target} not found in sd_target")
                sd_target[k_target] = v.half() if half else v

            logging.info(f"VAE baked from {vae_path}")

        save_model(sd_target, target_path)
        show_info_dialog(win=self, message=f"Done. Model saved as {target_path}")
        if self.callback:
            self.callback()


if __name__ == "__main__":
    import os

    ldm_model_dir = "/media/pup/ssd2/recoverable_data/sd_models/Stable-diffusion"
    sdxl_ldm_model_dir = ldm_model_dir
    vae_model_dir = "/media/pup/ssd2/recoverable_data/sd_models/VAE"
    sdxl_vae_model_dir = "/media/pup/ssd2/recoverable_data/sd_models/VAE_sdxl"

    win = ModelMixer(
        ldm_model_dir=ldm_model_dir,
        sdxl_ldm_model_dir=sdxl_ldm_model_dir,
        vae_model_dir=vae_model_dir,
        sdxl_vae_model_dir=sdxl_vae_model_dir
        )
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
