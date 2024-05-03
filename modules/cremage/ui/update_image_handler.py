import os
import sys
import logging
import json
from typing import Dict, Any, Union

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import GLib
from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [PROJECT_ROOT, MODULE_ROOT] + sys.path
from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.gtk_utils import text_view_set_text
from cremage.ui.image_listbox_handlers import refresh_image_file_list
from cremage.ui.image_listbox_handlers import update_image_file_list_to_show_generated_images
from cremage.const.const import *

def update_image(app, img_data: Image, generation_parameters:Union[Dict[str, Any], str]=None) -> None:
    """
    Updates the Image widget with new image data

    Args:
        img_data: Generated image in PIL Image
        generation_parameters: Image generation parameters in JSON dict serialized to str.
            Key is "generation_data".
    """       
    app.current_image = img_data
    if generation_parameters:
        if isinstance(generation_parameters, str):
            app.current_image_generation_information_dict = json.loads(generation_parameters)
        elif isinstance(generation_parameters, dict):
            app.current_image_generation_information_dict = generation_parameters
            # Convert to string
            generation_parameters = str(generation_parameters)
        else:
            raise ValueError("generation parameters is in unsupported format")
    else:
        app.current_image_generation_information_dict = None
    pixbuf = pil_image_to_pixbuf(resize_pil_image(app.current_image, target_size=MAIN_IMAGE_CANVAS_SIZE))
    
    # Safely update the image on the main thread
    # Update the image list to show the newly generated images at the top
    GLib.idle_add(update_image_file_list_to_show_generated_images, app)

    # Set the last generated image in the main window
    GLib.idle_add(app.image.set_from_pixbuf, pixbuf)

    # Set the generation information of the image in the generation information field
    GLib.idle_add(text_view_set_text, app.generation_information, generation_parameters)