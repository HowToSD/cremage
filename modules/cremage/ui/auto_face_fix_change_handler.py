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
from cremage.utils.gtk_utils import combo_box_get_text


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def auto_face_fix_changed(app, combo):
    auto_face_fix = combo_box_get_text(combo)
    if auto_face_fix is None:
        return  # No-op
    if auto_face_fix == "True":
        v = True
    else:
        v = False
    toggle_auto_face_fix_ui(app, v)


def toggle_auto_face_fix_ui(app:Gtk.Window, auto_face_fix):

    fields = [
        app.fields["auto_face_fix_strength"],
        app.fields["auto_face_fix_prompt"],
        app.fields["auto_face_fix_face_detection_method"],
        app.fields1_labels["Auto face fix strength"],
        app.fields1_labels["Auto face fix prompt"],
        app.fields1_labels["Auto face fix face detection method"]
    ]

    for f in fields:
        if auto_face_fix:
            f.show()
        else:
            f.hide()
