"""
Defines prompt history viewer invocation handlers
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
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.const.const import *
from cremage.ui.prompt_history_viewer import PromptHistoryViewer
from cremage.utils.gtk_utils import text_view_set_text
from cremage.utils.prompt_history import update_prompt_history
from cremage.utils.prompt_history import POSITIVE_PROMPT_PATH
from cremage.utils.prompt_history import NEGATIVE_PROMPT_PATH
from cremage.utils.prompt_history import POSITIVE_PROMPT_PRE_EXPANSION_PATH
from cremage.utils.prompt_history import NEGATIVE_PROMPT_PRE_EXPANSION_PATH
from cremage.utils.prompt_history import POSITIVE_PROMPT_EXPANSION_PATH
from cremage.utils.prompt_history import NEGATIVE_PROMPT_EXPANSION_PATH
from cremage.utils.prompt_history import positive_prompts_data
from cremage.utils.prompt_history import negative_prompts_data
from cremage.utils.prompt_history import positive_prompts_expansion_data
from cremage.utils.prompt_history import negative_prompts_expansion_data

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def show_positive_prompt_history_handler(app:Gtk.Window, widget:Gtk.Button) -> None:
    """
    Displays the positive prompt history.
    
    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
    """
    logger.debug("Positive prompt history button clicked")
    data_path = POSITIVE_PROMPT_PATH

    update_handler = partial(update_positive_prompt, app)
    app.positive_prompt_history_window = PromptHistoryViewer(
        data_path=data_path,
        update_handler=update_handler)
    app.positive_prompt_history_window.connect("delete-event",
                                               lambda widget, event, app=app: on_positive_prompt_history_viewer_delete(app, widget, event))
    app.positive_prompt_history_window.show_all()

def on_positive_prompt_history_viewer_delete(app, widget, event):
    logger.info("positive_prompt_history_window is destroyed")
    app.positive_prompt_history_window = None

def update_positive_prompt(app, prompt):
    text_view_set_text(app.positive_prompt, prompt)

def show_negative_prompt_history_handler(app:Gtk.Window, widget:Gtk.Button) -> None:
    """
    Displays the negative prompt history.
    
    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
    """
    logger.debug("Negative prompt history button clicked")
    data_path = NEGATIVE_PROMPT_PATH
    update_handler = partial(update_negative_prompt, app)
    app.negative_prompt_history_window = PromptHistoryViewer(
        data_path=data_path,
        update_handler=update_handler)
    app.negative_prompt_history_window.connect("delete-event",
                                               lambda widget, event, app=app: on_negative_prompt_history_viewer_delete(app, widget, event))
    app.negative_prompt_history_window.show_all()

def on_negative_prompt_history_viewer_delete(app, widget, event):
    logger.info("negative_prompt_history_window is destroyed")
    app.negative_prompt_history_window = None

def update_negative_prompt(app, prompt):
    text_view_set_text(app.negative_prompt, prompt)

# Prompt expansion (post-expansion)

def show_positive_prompt_expansion_history_handler(app:Gtk.Window, widget:Gtk.Button) -> None:
    """
    Displays the positive prompt_expansion history.
    
    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
    """
    logger.debug("Positive prompt_expansion history button clicked")
    data_path = POSITIVE_PROMPT_EXPANSION_PATH

    update_handler = partial(update_positive_prompt_expansion, app)
    app.positive_prompt_expansion_history_window = PromptHistoryViewer(
        data_path=data_path,
        update_handler=update_handler)
    app.positive_prompt_expansion_history_window.connect("delete-event",
                                               lambda widget, event, app=app: on_positive_prompt_expansion_history_viewer_delete(app, widget, event))
    app.positive_prompt_expansion_history_window.show_all()

def on_positive_prompt_expansion_history_viewer_delete(app, widget, event):
    logger.info("positive_prompt_expansion_history_window is destroyed")
    app.positive_prompt_expansion_history_window = None

def update_positive_prompt_expansion(app, prompt_expansion):
    text_view_set_text(app.fields["positive_prompt_expansion"], prompt_expansion)

def show_negative_prompt_expansion_history_handler(app:Gtk.Window, widget:Gtk.Button) -> None:
    """
    Displays the negative prompt_expansion history.
    
    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
    """
    logger.debug("Negative prompt_expansion history button clicked")
    data_path = NEGATIVE_PROMPT_EXPANSION_PATH
    update_handler = partial(update_negative_prompt_expansion, app)
    app.negative_prompt_expansion_history_window = PromptHistoryViewer(
        data_path=data_path,
        update_handler=update_handler)
    app.negative_prompt_expansion_history_window.connect("delete-event",
                                               lambda widget, event, app=app: on_negative_prompt_expansion_history_viewer_delete(app, widget, event))
    app.negative_prompt_expansion_history_window.show_all()

def on_negative_prompt_expansion_history_viewer_delete(app, widget, event):
    logger.info("negative_prompt_expansion_history_window is destroyed")
    app.negative_prompt_expansion_history_window = None

def update_negative_prompt_expansion(app, prompt_expansion):
    text_view_set_text(app.fields["negative_prompt_expansion"], prompt_expansion)

# Prompt pre_expansion (post-pre_expansion)
def show_positive_prompt_pre_expansion_history_handler(app:Gtk.Window, widget:Gtk.Button) -> None:
    """
    Displays the positive prompt_pre_expansion history.
    
    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
    """
    logger.debug("Positive prompt_pre_expansion history button clicked")
    data_path = POSITIVE_PROMPT_PRE_EXPANSION_PATH

    update_handler = partial(update_positive_prompt_pre_expansion, app)
    app.positive_prompt_pre_expansion_history_window = PromptHistoryViewer(
        data_path=data_path,
        update_handler=update_handler)
    app.positive_prompt_pre_expansion_history_window.connect("delete-event",
                                               lambda widget, event, app=app: on_positive_prompt_pre_expansion_history_viewer_delete(app, widget, event))
    app.positive_prompt_pre_expansion_history_window.show_all()

def on_positive_prompt_pre_expansion_history_viewer_delete(app, widget, event):
    logger.info("positive_prompt_pre_expansion_history_window is destroyed")
    app.positive_prompt_pre_expansion_history_window = None

def update_positive_prompt_pre_expansion(app, prompt_pre_expansion):
    text_view_set_text(app.fields["positive_prompt_pre_expansion"], prompt_pre_expansion)

def show_negative_prompt_pre_expansion_history_handler(app:Gtk.Window, widget:Gtk.Button) -> None:
    """
    Displays the negative prompt_pre_expansion history.
    
    Args:
        app (Gtk.Window): The application window instance
        widget (Gtk.Button): The button clicked
    """
    logger.debug("Negative prompt_pre_expansion history button clicked")
    data_path = NEGATIVE_PROMPT_PRE_EXPANSION_PATH
    update_handler = partial(update_negative_prompt_pre_expansion, app)
    app.negative_prompt_pre_expansion_history_window = PromptHistoryViewer(
        data_path=data_path,
        update_handler=update_handler)
    app.negative_prompt_pre_expansion_history_window.connect("delete-event",
                                               lambda widget, event, app=app: on_negative_prompt_pre_expansion_history_viewer_delete(app, widget, event))
    app.negative_prompt_pre_expansion_history_window.show_all()

def on_negative_prompt_pre_expansion_history_viewer_delete(app, widget, event):
    logger.info("negative_prompt_pre_expansion_history_window is destroyed")
    app.negative_prompt_pre_expansion_history_window = None

def update_negative_prompt_pre_expansion(app, prompt_pre_expansion):
    text_view_set_text(app.fields["negative_prompt_pre_expansion"], prompt_pre_expansion)