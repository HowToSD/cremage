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

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path

from cremage.utils.gtk_utils import open_file_chooser_dialog
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from cremage.const.const import *
from cremage.utils.image_utils import resize_crop_pil_image
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image, resize_with_padding

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def video_generator_input_image_view_click_handler(app, widget, event):
    """
    Opens a file chooser that lets the user to choose the video generator input image.
    """
    file_path = open_file_chooser_dialog(app, title="Select an image file")
    if file_path:
        
        # Update the input image representation. Do not resize this yet.
        app.pil_image_original = Image.open(file_path)
        app.pil_image_resized =  resize_crop_pil_image(app.pil_image_original)

        # Update the UI for input image
        resized_pil_image = resize_with_padding(
                app.pil_image_original,
                target_width=VIDEO_GENERATOR_THUMBNAIL_IMAGE_WIDTH,
                target_height=VIDEO_GENERATOR_THUMBNAIL_IMAGE_HEIGHT)
        set_pil_image_to_gtk_image(resized_pil_image, app.video_generator_input_image_thumbnail)

        resized_pil_image = resize_with_padding(
                app.pil_image_resized,
                target_width=VIDEO_GENERATOR_THUMBNAIL_IMAGE_WIDTH,
                target_height=VIDEO_GENERATOR_THUMBNAIL_IMAGE_HEIGHT)
        set_pil_image_to_gtk_image(resized_pil_image, app.cropped_image_view)