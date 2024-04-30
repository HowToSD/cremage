"""
Save preference button handler for the main UI.
Fields containing preferences that can be saved should be in app.fields dictionary.
"""
import os
import logging
import sys

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.configs.preferences import save_user_config
from cremage.ui.ui_to_preferences import copy_ui_field_values_to_preferences

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def save_preference_handler(app, widget) -> None:
    """
    Updates preferences based on fields

    Args:
        app (Gtk.Window): The application window instance
        widget: The button clicked
    """
    copy_ui_field_values_to_preferences(app)
    save_user_config(app.preferences)
    logger.info("User preference saved.")
