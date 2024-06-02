"""
Visual prompt builder.
Generate a prompt based on the set of selected images displayed on the UI.

Copyright (c) 2024 Hideyuki Inada.
"""
import os
import sys
import logging

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GdkPixbuf

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.gtk_utils import text_view_get_text, text_view_set_text

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class PromptBuilder:
    def __init__(self, app=None, input_directory=None):
        # Process arguments
        self.app = app
        if self.app:
            self.positive_prompt_field = self.app.positive_prompt
        else:
            self.positive_prompt_field = None
        self.input_directory = input_directory

        # Create the main window
        self.window = Gtk.Window(title="Visual Prompt Builder")
        self.window.set_default_size(800, 600)

        # Start UI layout
        # Create the main vertical box
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.window.add(vbox)

        # Create the notebook (tabbed interface)
        self.notebook = Gtk.Notebook()
        vbox.pack_start(self.notebook, True, True, 0)

        # Create the prompt text view field
        self.prompt_text_view = Gtk.TextView()
        self.prompt_text_view.set_editable(False)
        vbox.pack_start(self.prompt_text_view, False, False, 0)

        # Copy to the prompt fiel
        copy_button = Gtk.Button(label="Copy to prompt")
        copy_button.connect("clicked", self.on_copy_clicked)
        vbox.pack_start(copy_button, False, True, 0)
        # End UI layout

        # Load categories and create tabs
        self.load_categories()

        self.window.show_all()

    def on_copy_clicked(self, widget):
        if self.prompt_text_view is None or self.positive_prompt_field is None:
            return

        current_prompt = text_view_get_text(self.positive_prompt_field)
        tags = text_view_get_text(self.prompt_text_view)
        if tags not in current_prompt:
            new_prompt = current_prompt + " " + tags
            text_view_set_text(self.positive_prompt_field, new_prompt)
            
    def load_categories(self):
        if not os.path.exists(self.input_directory):
            logger.info(f"Directory {self.input_directory} does not exist")
            return

        categories = [d for d in os.listdir(self.input_directory) if os.path.isdir(os.path.join(self.input_directory, d))]

        categories = sorted(categories)
        for category in categories:
            self.create_category_tab(category)

    def create_category_tab(self, category):
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        flowbox = Gtk.FlowBox()
        flowbox.set_valign(Gtk.Align.START)
        flowbox.set_max_children_per_line(4)
        flowbox.set_selection_mode(Gtk.SelectionMode.NONE)

        category_path = os.path.join(self.input_directory, category)
        for filename in os.listdir(category_path):
            if filename.endswith(".jpg"):
                self.create_thumbnail(flowbox, category_path, filename)

        scrolled_window.add(flowbox)
        self.notebook.append_page(scrolled_window, Gtk.Label(label=category))

    def create_thumbnail(self, flowbox, category_path, filename):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        image_path = os.path.join(category_path, filename)
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(image_path, 256, 256, False)
        image = Gtk.Image.new_from_pixbuf(pixbuf)
        vbox.pack_start(image, True, True, 0)

        checkbox = Gtk.CheckButton()
        checkbox.connect("toggled", self.on_checkbox_toggled, filename[:-4])
        vbox.pack_start(checkbox, False, False, 0)

        label = Gtk.Label(label=filename[:-4])
        vbox.pack_start(label, False, False, 0)

        flowbox.add(vbox)

    def on_checkbox_toggled(self, checkbox, tag):
        buffer = self.prompt_text_view.get_buffer()
        start_iter = buffer.get_start_iter()
        end_iter = buffer.get_end_iter()
        current_text = buffer.get_text(start_iter, end_iter, False)

        tags = set(current_text.split(", "))
        if checkbox.get_active():
            tags.add(tag)
        else:
            tags.discard(tag)

        new_text = ", ".join(sorted(tags))
        buffer.set_text(new_text)


if __name__ == "__main__":
    input_directory = "data/prompt_builder"
    app = PromptBuilder(input_directory=input_directory)
    Gtk.main()
