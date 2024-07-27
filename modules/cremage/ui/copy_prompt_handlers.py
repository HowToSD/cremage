"""
Copy prompt handlers
"""
import os
import logging
import sys
import re
import json

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path
from cremage.utils.gtk_utils import text_view_get_text, text_view_set_text

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def _get_generation_dict(app):
    generation_string = text_view_get_text(app.generation_information)

    # Deal with a string where the user decided to add a new line
    # in text view.
    # In this case, generation string contains \n as opposed to \\n
    # which unedited generation string contains.
    # A single backslash \n causes JSON parse error, so you need
    # to add the second backslash.
    # To do so, we first define the regex pattern to match \n not preceded by \
    # The pattern uses a negative lookbehind assertion to ensure that the \n is not part of \\n
    # (?<!...) is the syntax for negative lookbehind
    # (?<!\\) ensures that the position is not preceded by a single backslash
    pattern = r'(?<!\\)\n'

    # Then replace \n with \\n using the regex pattern
    generation_string = re.sub(pattern, r'\\n', generation_string)

    try:
        print(generation_string)
        info_dict = json.loads(generation_string)
    except:
        logger.warn("json parse error.")
        info_dict = dict()

    return info_dict


def copy_positive_prompt_handler(app, widget):
    info = _get_generation_dict(app)
    if "positive_prompt" in info:
        text_view_set_text(app.positive_prompt, info["positive_prompt"])

def copy_negative_prompt_handler(app, widget):
    info = _get_generation_dict(app)
    if "negative_prompt" in info:
        text_view_set_text(app.negative_prompt, info["negative_prompt"])

