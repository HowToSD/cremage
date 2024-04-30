"""
Defines mask image editor button click handler.
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
from mask_image_editor import MaskImageEditor
from cremage.const.const import *
from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.image_utils import resize_pil_image

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def update_mask_image_from_editor(app: Gtk.Window, image: Image):

        source_pixbuf = pil_image_to_pixbuf(
            resize_pil_image(image, target_size=THUMBNAIL_IMAGE_EDGE_LENGTH))
        app.mask_image.set_from_pixbuf(source_pixbuf)
        app.mask_image_original_size = image
        app.mask_image_container_box.show_all()


def mask_image_view_click_handler(app, widget, event):
        if app.input_image_original_size:
                width,height = app.input_image_original_size.size
        else:
                width = app.preferences["image_width"]
                height = app.preferences["image_height"]
        logger.info(f"Opening mask image editor with width: {width}, height: {height}")

        update_mask_image_from_editor_wrapper = partial(
                update_mask_image_from_editor, app
        )

        mask_editor = MaskImageEditor(
                        base_image=app.input_image_original_size,
                        mask_image=app.mask_image_original_size,
                        output_file_path=app.mask_image_path,
                        width=width,
                        height=height,
                        parent_window_update_func=update_mask_image_from_editor_wrapper)
        mask_editor.show_all()