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
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path

from cremage.const.const import THUMBNAIL_IMAGE_EDGE_LENGTH
from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.gtk_utils import open_file_chooser_dialog
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from cremage.ui.image_viewer import ImageViewer


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main_image_view_click_handler(app, widget, event):
    """
    Opens a file chooser that lets the user to choose the input image.
    """
    if event.type == Gdk.EventType.BUTTON_PRESS and app.current_image:  # single click
        # event.type == Gdk.EventType._2BUTTON_PRESS:
        image_viewer = ImageViewer(pil_image = app.current_image)
        # image_viewer.connect("destroy", Gtk.main_quit)
        image_viewer.show_all()
        
    # elif event.type == Gdk.EventType.BUTTON_PRESS:  # single click
    #     image_path = open_file_chooser_dialog(app, title="Select an image file")
    #     if image_path:
    #         # Update the input image representation. Do not resize this yet.
    #         app.current_image = Image.open(image_path)

    #         # Update the UI for input image
    #         resized_pil_image = resize_pil_image(app.current_image, THUMBNAIL_IMAGE_EDGE_LENGTH)
    #         set_pil_image_to_gtk_image(resized_pil_image, app.image)




