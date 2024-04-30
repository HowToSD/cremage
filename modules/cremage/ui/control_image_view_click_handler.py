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
from tools.control_net_annotator import ControlNetImageAnnotator
from cremage.const.const import *
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.misc_utils import get_tmp_dir, get_tmp_image_file_name
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from .graffiti_editor_widget_click_handler import UpdateControlImageViewWrapper

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def control_image_view_click_handler(app, widget, event):
    # width, height = app.input_image_original_size.size
    logger.info(f"Opening control_net_image_annotator")
    tmp_control_image_path = os.path.join(get_tmp_dir(), get_tmp_image_file_name())
    update_wrapper = UpdateControlImageViewWrapper(app, tmp_control_image_path)

    control_net_image_annotator = ControlNetImageAnnotator(
            pil_image=None,
            output_file_path=tmp_control_image_path,
            save_call_back=update_wrapper.update_control_image_view)
    control_net_image_annotator.show_all()
    control_net_image_annotator.on_annotator_changed(control_net_image_annotator.annotator_cb)
