"""
Defines the event handler for generator type (SD 1.5 vs SDXL) change.
"""
import os
import logging
import sys
import subprocess
import platform

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path
from cremage.const.const import THUMBNAIL_IMAGE_EDGE_LENGTH
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from cremage.utils.app_misc_utils import get_next_face_file_path
from cremage.utils.misc_utils import open_os_directory

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def sdxl_use_refiner_check_box_changed(app, check_box):
    use_refiner = check_box.get_active()
    toggle_use_refiner_ui(app, use_refiner)


def toggle_use_refiner_ui(app:Gtk.Window, use_refiner):
    if use_refiner:
        for k, v in app.fields.items():
            if k.startswith("refiner_") or k == "sdxl_refiner_strength":
                v.show()

        for k, v in app.fields2_labels.items():
            if k.startswith("refiner_sdxl_") or k == "Refiner strength":
                v.show()
    else:
        for k, v in app.fields.items():
            if k.startswith("refiner_") or k == "sdxl_refiner_strength":
                v.hide()

        for k, v in app.fields2_labels.items():
            if k.startswith("refiner_sdxl_") or k == "Refiner strength":
                v.hide()


