"""
Main UI definition
"""
import os
import logging
import sys
import re
import subprocess
import platform
from functools import partial

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.ui.model_path_update_handler import update_all_model_paths
from cremage.ui.preferences_ui import PreferencesWindow
from cremage.ui.token_viewer_window import TokenViewerWindow
from cremage.ui.embedding_file_viewer_window import EmbeddingFileViewerWindow
from cremage.utils.misc_utils import open_os_directory
from cremage.ui.ui_to_preferences import copy_ui_field_values_to_preferences

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main_menu_definition(app) -> None:
    """
    Defines main menu

    Args:
        app (Gtk.Window): The application window instance
        widget: The button clicked
    """
    accel_group = Gtk.AccelGroup()
    app.add_accel_group(accel_group)

    app.menu_bar = Gtk.MenuBar()  # Create menu bar

    # File menu
    file_menu = Gtk.Menu()  # Create file menu
    file_item = Gtk.MenuItem(label="File")  # Create file menu item
    file_item.set_submenu(file_menu)  # Set submenu

    # Preferences
    preferences_item = Gtk.MenuItem(label="Preferences")  # Create save menu item
    preferences_item.connect("activate", lambda widget, app=app: on_preferences_activate(app, widget))  # Connect to handler
    preferences_item.add_accelerator("activate", accel_group, ord('P'),
                            Gdk.ModifierType.CONTROL_MASK, Gtk.AccelFlags.VISIBLE)
    file_menu.append(preferences_item)

    # Favorites directory
    favorites_dir_viewer_item = Gtk.MenuItem(label="View favorites directory")
    favorites_dir_viewer_item.connect("activate", 
                                   lambda widget, app=app: open_favorites_dir_handler(app, widget))
    file_menu.append(favorites_dir_viewer_item)

    # Output directory
    output_dir_viewer_item = Gtk.MenuItem(label="View output directory")
    output_dir_viewer_item.connect("activate", 
                                   lambda widget, app=app: open_output_dir_handler(app, widget))
    file_menu.append(output_dir_viewer_item)

    app.menu_bar.append(file_item)  # Add file menu to menu bar

    # Tools menu
    tools_menu = Gtk.Menu()
    tools_item = Gtk.MenuItem(label="Tools")
    tools_item.set_submenu(tools_menu)

    # Token viewer menu item
    token_viewer_item = Gtk.MenuItem(label="Token viewer")
    token_viewer_item.connect("activate", lambda widget, app=app: open_token_viewer(app, widget))
    tools_menu.append(token_viewer_item)

    # Embedding file viewer menu item
    embedding_file_viewer_item = Gtk.MenuItem(label="TI Embedding")
    embedding_file_viewer_item.connect("activate", lambda widget, app=app:open_embedding_file_viewer(app, widget))   
    tools_menu.append(embedding_file_viewer_item)
    
    app.menu_bar.append(tools_item)


def open_output_dir_handler(app, widget):
    """
    Opens the directory using OS-specific file viewer application.

    This is used to drag and drop images into Cremage.
    """
    directory_path = app.output_dir
    open_os_directory(directory_path)

def open_favorites_dir_handler(app, widget):
    """
    Opens the directory using OS-specific file viewer application.

    This is used to drag and drop images into Cremage.
    """
    directory_path = app.favorites_dir
    open_os_directory(directory_path)


def on_preferences_activate(app, widget):
    logger.info("Preferences menu item selected")
    update_all_model_paths_wrapper = partial(update_all_model_paths, app)
    prefs_window = PreferencesWindow(app, app.preferences,
                                        change_call_back=update_all_model_paths_wrapper)
    prefs_window.show_all()

def open_token_viewer(app, widget):
    if not hasattr(app, "token_viewer_window") or not app.token_viewer_window:
        app.token_viewer_window = TokenViewerWindow()
        app.token_viewer_window.connect("delete-event", lambda widget, event, app=app: on_token_viewer_delete(app, widget, event))
    app.token_viewer_window.show_all()

def on_token_viewer_delete(app, widget, event):
    logger.info("Token viewer window is destroyed")
    app.token_viewer_window = None

def on_tool_palette_window_delete(app, widget, event):
    logger.info("Tool palette is destroyed")
    app.tool_palette_window = None    

# Embedding file viewer
def open_embedding_file_viewer(app, widget):
    if not hasattr(app, "embedding_file_viewer_window") or not app.embedding_file_viewer_window:

        copy_ui_field_values_to_preferences(app)

        if app.preferences["generator_model_type"] == "SD 1.5":
            embedding_path = app.preferences["embedding_path"]
        else:  # if sdxl
            embedding_path = app.preferences["sdxl_embedding_path"]

        app.embedding_file_viewer_window = EmbeddingFileViewerWindow(app=app,
                                                                     embedding_path=embedding_path,
                                                                     embedding_images_dir=app.embedding_images_dir)
        app.embedding_file_viewer_window.connect("delete-event", lambda widget, event, app=app: on_embedding_file_viewer_delete(app, widget, event))
    app.embedding_file_viewer_window.show_all()

def on_embedding_file_viewer_delete(app, widget, event):
    logger.info("embedding_file_viewer_window is destroyed")
    app.embedding_file_viewer_window = None
