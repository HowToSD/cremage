"""
Defines graffiti editor button click handler.
"""
import os
import logging
import sys

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path

from cremage.configs.preferences import save_user_config
from tools.graffiti_editor import GraffitiEditor
from cremage.const.const import *
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.misc_utils import get_tmp_dir, get_tmp_image_file_name
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def update_control_image_view(app, pil_image: Image, control_net_image_path):
    resized_pil_image = resize_pil_image(pil_image, THUMBNAIL_IMAGE_EDGE_LENGTH)
    # Update thumbnail
    set_pil_image_to_gtk_image(resized_pil_image, app.control_net_image_view)
    app.control_net_image = pil_image
    # Update app's control net file path. This is passed to the generator
    app.control_net_image_file_path = control_net_image_path


class UpdateControlImageViewWrapper:
    def __init__(self, app, control_net_image_path):
        self.app = app
        self.control_net_image_path = control_net_image_path

    def update_control_image_view(self, pil_image):
        update_control_image_view(self.app, pil_image, self.control_net_image_path)


def graffiti_editor_widget_click_handler(app, widget, event) -> None:
    """
    Invokes graffiti editor.

    Args:
        app (Gtk.Window): The application window instance
        widget: The button clicked
    """
    tmp_control_image_path = os.path.join(get_tmp_dir(), get_tmp_image_file_name())
    if not hasattr(app, "graffiti_editor") or app.graffiti_editor is None:

        update_wrapper = UpdateControlImageViewWrapper(app, tmp_control_image_path)
        app.graffiti_editor = GraffitiEditor(
            pil_image=app.control_net_image,
            output_file_path=tmp_control_image_path,
            save_call_back=update_wrapper.update_control_image_view,
            preferences=app.preferences,  # Needed for image_width and image_height
            generation_information_call_back=None)
        app.graffiti_editor.connect("delete-event", lambda widget, event, app=app: graffiti_editor_delete_handler(app, widget, event))

    app.graffiti_editor.show_all()   

def graffiti_editor_delete_handler(app, widget, event):
    logger.info("Graffiti editor is destroyed")
    app.graffiti_editor = None 
