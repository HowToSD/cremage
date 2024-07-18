import os
import sys

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

PROJECT_ROOT = os.path.realpath(os.path.dirname(__file__))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.configs.preferences import save_user_config
from cremage.utils.gtk_utils import create_combo_box

TRUE_FALSE_LIST = ["True", "False"]
INT_FIELDS = ["inpaint_max_edge_len"]
FLOAT_FIELDS = []

class PreferencesWindow(Gtk.Window):
    def __init__(self, parent, preferences, change_call_back=None):
        super().__init__(title="Preferences", transient_for=parent)
        self.preferences = preferences
        self.change_call_back = change_call_back

        self.set_border_width(10)
        self.set_default_size(900, 350)

        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        self.add(grid)

        # Create fields
        self.fields = {
            "ldm_model_path": Gtk.Entry(text=preferences["ldm_model_path"]),
            "vae_model_path": Gtk.Entry(text=preferences["vae_model_path"]),
            "lora_model_path": Gtk.Entry(text=preferences["lora_model_path"]),
            "control_model_path": Gtk.Entry(text=preferences["control_model_path"]),
            "embedding_path": Gtk.Entry(text=preferences["embedding_path"]),
            "sdxl_embedding_path": Gtk.Entry(text=preferences["sdxl_embedding_path"]),            
            "sdxl_ldm_model_path": Gtk.Entry(text=preferences["sdxl_ldm_model_path"]),
            "sdxl_vae_model_path": Gtk.Entry(text=preferences["sdxl_vae_model_path"]),
            "sdxl_lora_model_path": Gtk.Entry(text=preferences["sdxl_lora_model_path"]),
            "sd3_ldm_model_path": Gtk.Entry(text=preferences["sd3_ldm_model_path"]),
            "pixart_sigma_ldm_model_path": Gtk.Entry(text=preferences["pixart_sigma_ldm_model_path"]),
            "svd_model_path": Gtk.Entry(text=preferences["svd_model_path"]),
            "wildcards_path": Gtk.Entry(text=preferences["wildcards_path"]),
            "safety_check": create_combo_box(TRUE_FALSE_LIST, int(not preferences["safety_check"])),
            "watermark": create_combo_box(TRUE_FALSE_LIST, int(not preferences["watermark"])),
            "enable_hf_internet_connection": create_combo_box(TRUE_FALSE_LIST, int(not preferences["enable_hf_internet_connection"])),
            "inpaint_max_edge_len":  Gtk.Entry(text=preferences["inpaint_max_edge_len"])
        }

        # Add fields to the grid
        for i, (label_text, field) in enumerate(self.fields.items()):
            field.set_hexpand(True)  # Ensure fields expand horizontally.
                                     # Without this, fields do not expand leaving a lot of
                                     # blank space on the right.
            label = Gtk.Label(label=label_text.replace("_", " ").capitalize(), halign=Gtk.Align.START)
            grid.attach(label, 0, i, 1, 1)
            grid.attach(field, 1, i, 3, 1)

        # Save and Cancel buttons
        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", self.on_save_clicked)
        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", self.on_cancel_clicked)

        # Set a fixed width for the buttons
        button_width = 100  # Set this to your desired width
        save_button.set_size_request(button_width, -1)
        cancel_button.set_size_request(button_width, -1)
        grid.attach(save_button, 1, i + 1, 1, 1)
        grid.attach(cancel_button, 2, i + 1, 1, 1)

    def on_save_clicked(self, widget):
        # Update preferences based on fields
        for key, field in self.fields.items():
            if isinstance(field, Gtk.ComboBoxText):  # bool
                self.preferences[key] = field.get_active_text() == "True"
            else:
                if key in INT_FIELDS:
                    self.preferences[key] = int(field.get_text())
                elif key in FLOAT_FIELDS:
                    self.preferences[key] = float(field.get_text())
                else:  # str
                    self.preferences[key] = field.get_text()

        # Save to config.yaml
        save_user_config(self.preferences)
        if self.change_call_back:
            self.change_call_back()
        self.destroy()

    def on_cancel_clicked(self, widget):
        self.destroy()
