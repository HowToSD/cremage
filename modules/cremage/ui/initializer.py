"""
Main UI definition
"""
import os
import logging
import sys
from functools import partial
import time

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.configs.preferences import load_user_config
from cremage.ui.ui_definition import main_ui_definition
from cremage.ui.window_realized_handler import window_realized_handler
from cremage.ui.generator_model_type_change_handler import toggle_genenator_model_type_ui
from cremage.ui.sdxl_use_refiner_check_box_handler import toggle_use_refiner_ui
from cremage.const.const import *

BLANK_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "blank_image_control_net.png")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def initializer(app:Gtk.Window) -> None:
    """
    Defines application's initialization code.

    Args:
        app (Gtk.Window): The application window instance
    """
    # clean_tmp_dir()  # FIXME
    first_init(app)  
    main_ui_definition(app)   
    final_init(app)


def first_init(app:Gtk.Window):
    """
    Init that has to be done before setting up UI.
    """

    # Call back definition
    def get_current_image_call_back():
        return app.current_image
    setattr(app, 'get_current_image_call_back', get_current_image_call_back)

    # Call back definition
    def get_current_face_image_call_back():
        return app.face_input_image_original_size
    setattr(app, 'get_current_face_image_call_back', get_current_face_image_call_back)

    # Directory set up
    app.tmp_dir = os.path.join(os.path.expanduser("~"), ".cremage", "tmp")
    if os.path.exists(app.tmp_dir) is False:
        os.makedirs(app.tmp_dir, exist_ok=True)

    # Image output directory
    app.output_dir = os.path.join(os.path.expanduser("~"), ".cremage", "outputs")  # FIXME. Make it configurable    
    if os.path.exists(app.output_dir) is False:
        os.makedirs(app.output_dir, exist_ok=True)
    app.trash_dir = os.path.join(os.path.expanduser("~"), ".cremage", "trash")
    if os.path.exists(app.trash_dir) is False:
        os.makedirs(app.trash_dir, exist_ok=True)       
    app.data_dir = os.path.join(os.path.expanduser("~"), ".cremage", "data")
    if os.path.exists(app.data_dir) is False:
        os.makedirs(app.data_dir, exist_ok=True)       
    app.face_dir = os.path.join(app.data_dir, "faces")
    if os.path.exists(app.face_dir) is False:
        os.makedirs(app.face_dir, exist_ok=True)
    app.embedding_images_dir = os.path.join(app.data_dir, "embedding_images")
    if os.path.exists( app.embedding_images_dir) is False:
        os.makedirs(app.embedding_images_dir, exist_ok=True)

    # Preferences
    app.preferences = load_user_config()
    app.preferences_prev = app.preferences

    # Image objects in PIL format in original resolution and relevant path
    app.current_image = None  # Image displayed in the main window.
    app.input_image_original_size = None  # Input image for img2img and inpainting
    app.mask_image_original_size = None  # Mask image for inpainting
    app.mask_image_path = os.path.join(app.tmp_dir, "mask_image.png")
    app.control_net_image = None
    app.control_net_image_file_path = None
    app.face_input_image_original_size = None  # FaceID source image
    app.face_input_image_file_path = None  # FaceID source image

    app.marked_image_path = None  # The path of an image marked by the user in the image listbox

    app.generation_mode = MODE_TEXT_TO_IMAGE
    app.current_image_generation_information_dict = None


def final_init(app:Gtk.Window):
    # Connect to the realize signal
    app.connect("realize", lambda widget, app=app: window_realized_handler(app, widget))

def setup_field_visibility(app:Gtk.Window):
    generator_model_type = "SDXL" if app.preferences["generator_model_type"] == "SDXL" else "SD 1.5"
    toggle_genenator_model_type_ui(app, generator_model_type)
    
    use_refiner = True if app.preferences["sdxl_use_refiner"] else False
    toggle_use_refiner_ui(app, use_refiner)

