"""
Defines miscellaneous event handlers for face UI on the main UI app (not tools).
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

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path
from cremage.const.const import THUMBNAIL_IMAGE_EDGE_LENGTH
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from cremage.utils.app_misc_utils import get_next_face_file_path
from cremage.utils.misc_utils import open_os_directory

BLANK_INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "blank_input_image.png")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def open_face_dir_button_handler(app:Gtk.Window, widget:Gtk.Button, event:Gdk.EventButton) -> None:
    """
    Opens the face image directory.

    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
        event (Gdk.EventButton): The event
    """
    directory_path = app.face_dir
    open_os_directory(directory_path)


def copy_current_image_to_face_button_handler(app:Gtk.Window, widget:Gtk.Button, event:Gdk.EventButton) -> None:
    """
    Copies image from the main image window to the face input image area.

    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
        event (Gdk.EventButton): The event
    """
    logger.info("Copy output image clicked")

    # Check if the source_pixbuf is not None
    if app.current_image is None:
        logger.info("No image is selected in the image list box.")
        return

    app.face_input_image_file_path = get_next_face_file_path(app)  # Get the new path for the image
    app.current_image.save(app.face_input_image_file_path)

    # Update the input image representation. Do not resize this yet.
    app.face_input_image_original_size = Image.open(app.face_input_image_file_path)

    # Update the UI for input image
    resized_pil_image = resize_pil_image(app.face_input_image_original_size, THUMBNAIL_IMAGE_EDGE_LENGTH)
    set_pil_image_to_gtk_image(resized_pil_image, app.face_image_input)
