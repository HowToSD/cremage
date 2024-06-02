"""
Defines drag and drop handlers
"""
import os
import logging
import sys
import shutil

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path

from cremage.configs.preferences import save_user_config
from tools.graffiti_editor import GraffitiEditor
from cremage.const.const import *
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.misc_utils import get_tmp_dir, get_tmp_image_file_name
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from cremage.utils.app_misc_utils import copy_face_file_to_face_storage

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def _extract_file_path(data):
    file_path = data.get_text().strip()
    if file_path.startswith('file://'):
        file_path = file_path[7:]
    else:
        file_path = None
    return file_path

# drag & drop handlers
def control_image_drag_data_received(app, 
                                     widget, 
                                     drag_context, 
                                     x, 
                                     y, 
                                     data, 
                                     info, 
                                     time):
    """
    Drag and Drop handler for ControlNet image input thumbnail view.

    Args:
        data: Contains info for the dragged file name
    """
    file_path = _extract_file_path(data)
    if file_path is None:
        return
    pil_image = Image.open(file_path)
    resized_pil_image = resize_pil_image(pil_image, THUMBNAIL_IMAGE_EDGE_LENGTH)
    set_pil_image_to_gtk_image(resized_pil_image, app.control_net_image_view)
    app.control_net_image_file_path = file_path
    logger.info(f"Updated ControlNet image path to {app.control_net_image_file_path}")


def main_image_drag_data_received(app, 
                                     widget, 
                                     drag_context, 
                                     x, 
                                     y, 
                                     data, 
                                     info, 
                                     time):
    """
    Drag and Drop handler for image to main image view.

    Args:
        data: Contains info for the dragged file name
    """
    file_path = _extract_file_path(data)
    if file_path is None:
        return
    app.current_image = Image.open(file_path)

    # Update the UI for input image
    resized_pil_image = resize_pil_image(app.current_image, THUMBNAIL_IMAGE_EDGE_LENGTH)
    set_pil_image_to_gtk_image(resized_pil_image, app.image)


def input_image_drag_data_received(app, 
                                     widget, 
                                     drag_context, 
                                     x, 
                                     y, 
                                     data, 
                                     info, 
                                     time):
    """
    Drag and Drop handler for image to image input thumbnail view.

    Args:
        data: Contains info for the dragged file name
    """
    file_path = _extract_file_path(data)
    if file_path is None:
        return
    # Update the input image representation. Do not resize this yet.
    app.input_image_original_size = Image.open(file_path)

    # Update the UI for input image
    resized_pil_image = resize_pil_image(app.input_image_original_size, THUMBNAIL_IMAGE_EDGE_LENGTH)
    set_pil_image_to_gtk_image(resized_pil_image, app.image_input)


def face_input_image_drag_data_received(app, 
                                     widget, 
                                     drag_context, 
                                     x, 
                                     y, 
                                     data, 
                                     info, 
                                     time):
    """
    Drag and Drop handler for face input thumbnail view.

    Args:
        data: Contains info for the dragged file name
    """
    file_path = _extract_file_path(data)
    if file_path is None:
        return
    
    # Copy to face dir if the file is not from the face dir
    if os.path.dirname(file_path) == app.face_dir:
        app.face_input_image_file_path = file_path
    else:    
        app.face_input_image_file_path = copy_face_file_to_face_storage(app, file_path)

    # Update the input image representation. Do not resize this yet.
    app.face_input_image_original_size = Image.open(app.face_input_image_file_path)

    # Update the UI for input image
    resized_pil_image = resize_pil_image(app.face_input_image_original_size, THUMBNAIL_IMAGE_EDGE_LENGTH)
    set_pil_image_to_gtk_image(resized_pil_image, app.face_image_input)

