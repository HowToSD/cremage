"""
Defines input image view click handler.
"""
import os
import logging
import sys
from functools import partial

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path

from cremage.const.const import THUMBNAIL_IMAGE_EDGE_LENGTH
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.gtk_utils import open_file_chooser_dialog
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from cremage.utils.gtk_utils import set_image_to_gtk_image_from_file
from cremage.utils.app_misc_utils import copy_face_file_to_face_storage

BLANK_INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "blank_input_image.png")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def face_input_image_view_click_handler(app, widget, event):
    """
    Opens a file chooser that lets the user to choose the face input image.
    """
    image_path = open_file_chooser_dialog(app, title="Select an image file")
    if image_path:
        
        # Copy to face dir
        if os.path.dirname(image_path) == app.face_dir:
            app.face_input_image_file_path = image_path
        else:
            app.face_input_image_file_path = copy_face_file_to_face_storage(app, image_path)

        # Update the input image representation. Do not resize this yet.
        app.face_input_image_original_size = Image.open(app.face_input_image_file_path)

        # Update the UI for input image
        resized_pil_image = resize_pil_image(app.face_input_image_original_size, THUMBNAIL_IMAGE_EDGE_LENGTH)
        set_pil_image_to_gtk_image(resized_pil_image, app.face_image_input)


def face_input_image_view_close_handler(app, widget):
    """
    Deletes the face input image
    """
    app.face_input_image_original_size = None

    # Update the UI for input image
    set_image_to_gtk_image_from_file(
                                file_path=BLANK_INPUT_IMAGE_PATH,
                                gtk_image=app.face_image_input,
                                target_edge_length=THUMBNAIL_IMAGE_EDGE_LENGTH)


