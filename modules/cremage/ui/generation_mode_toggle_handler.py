"""
Defines handlers for the image listbox.
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
from cremage.const.const import MODE_IMAGE_TO_IMAGE
from cremage.const.const import MODE_INPAINTING
from cremage.const.const import MODE_TEXT_TO_IMAGE

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def generation_mode_toggle_handler(app:Gtk.Window, button:Gtk.RadioButton, name:str) -> None:
    """
    The event handler for the checkbox's "toggled" signal

    Args:
        app (Gtk.Window): The main application window.
        button (Gtk.RadioButton): The ratio button to toggle between generation modes.
        name (str): The new generation mode.
    """
    if button.get_active():
        state = "on" if button.get_active() else "off"
        logger.debug(f"Button '{name}' was turned {state}")
        if name == "image to image":
            app.image_input.show()
            app.input_image_wrapper.show()
            app.copy_image_button.show()
            app.mask_image.hide()
            app.mask_image_wrapper.hide()
            app.generation_mode = MODE_IMAGE_TO_IMAGE
        elif name == "inpainting":
            app.image_input.show()
            app.input_image_wrapper.show()
            app.copy_image_button.show()
            app.mask_image_wrapper.show()
            app.mask_image.show()
            app.generation_mode = MODE_INPAINTING
        else:  # txt2image
            app.image_input.hide()
            app.input_image_wrapper.hide()
            app.copy_image_button.hide()
            app.mask_image_wrapper.hide()
            app.mask_image.hide()
            app.generation_mode = MODE_TEXT_TO_IMAGE
