"""
Defines graffiti editor button click handler.
"""
import os
import logging
import sys

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.const.const import *
from cremage.utils.image_utils import pil_image_from_pixbuf, resize_pil_image, pil_image_to_pixbuf


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def copy_image_widget_click_handler(app:Gtk.Window, widget:Gtk.Button, event:Gdk.EventButton) -> None:
    """
    Copies image from the main image window to the input image view for image to image.

    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
        event (Gdk.EventButton): The event
    """
    logger.info("Copy output image clicked")

    # Check if the source_pixbuf is not None
    if app.current_image is not None:
        # Set the retrieved pixbuf to the target image widget
        app.input_image_original_size = app.current_image.copy()
        source_pixbuf = pil_image_to_pixbuf(
            resize_pil_image(app.input_image_original_size, target_size=THUMBNAIL_IMAGE_EDGE_LENGTH))
        app.image_input.set_from_pixbuf(source_pixbuf)
    else:
        logger.info("No image is loaded in the main image view.")

