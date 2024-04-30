"""
Miscellaneous utility functions that requires app argument (Gtk.Window)
"""
import os
import sys
import time
import re
import logging
import shutil
from typing import List, Dict, Any, Tuple
import json

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def get_next_file_id_from_list_of_file_names_with_numbers(files:List[str])->int:

    # Extract the number from each file and put them in a list
    number_list = list()
    for f in files:
        match = re.search(r"face_([0-9]+).png", f)
        if match:
            number_list.append(int(match.group(1)))

    if not number_list:
        new_id = 0
    else:
        new_id = max(number_list) + 1

    return new_id


def get_next_face_file_path(app:Gtk.Window)->str:
    """
    Get the next file path to create in the Cremage data directory.

    Args:
        app (Gtk.Window): The application instance.

    Returns:
        New full path of the face file name.
    """
    # Error checks
    if hasattr(app, "face_dir") is False:
        raise ValueError("Face directory attribute is not set for the app instance")
    
    if os.path.exists(app.face_dir) is False:
        raise ValueError(f"{app.face_dir} is not found")
    
    # Get the target file name
    # Extract 6 digits for all basename without extention
    # e.g. face_123456.png
    # face_123456_crop_i.png  # insightface crop
    # face_123456_crop_c.png  # clip crop
    # face_123456_embedding   # final [4, 768] embedding
    # Get all files
    files = os.listdir(app.face_dir)
    new_id = get_next_file_id_from_list_of_file_names_with_numbers(files)
    new_file_name = os.path.join(app.face_dir, f"face_{new_id:06d}.png")
    if os.path.exists(new_file_name):
        raise ValueError(f"Face file name {new_file_name} already exists. Check directory {app.face_dir} for file system integrity.")
    return new_file_name


def copy_face_file_to_face_storage(app:Gtk.Window, source_file_name:str)->str:
    """
    Copies the source face file to a Cremage data directory.

    Args:
        app (Gtk.Window): The application instance.
        source_file_name (str): The source file name.
    Returns:
        New full path of the face file name.
    """
    new_file_name = get_next_face_file_path(app)
    shutil.copy(source_file_name, new_file_name)
    if os.path.exists(new_file_name) is False:
        raise ValueError(f"Copy face file name to {new_file_name} failed. Check directory {app.face_dir} for file system integrity.")
    return new_file_name