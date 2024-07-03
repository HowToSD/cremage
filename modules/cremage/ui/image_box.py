"""
ImageBox
"""
import os
import sys
from typing import List, Any, Dict

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from PIL import Image

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.gtk_utils import resized_gtk_image_from_pil_image
from cremage.utils.image_utils import resize_pil_image
from cremage.utils.gtk_utils import open_file_chooser_dialog
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image
from cremage.ui.drag_and_drop_handlers import extract_file_path


class ImageBox(Gtk.Box):

    def __init__(self, pil_image=None, target_width=384, target_height=384, callback=None):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        self.pil_image = pil_image
        self.target_width = target_width
        self.target_height = target_height
        self.callback = callback
        self.set_halign(Gtk.Align.CENTER)  # Center the button horizontally

        # Wrap the input image in an event box to handle click events
        self.input_image_wrapper = Gtk.EventBox()
        self.add(self.input_image_wrapper)
        
        # Input image
        self.input_image = resized_gtk_image_from_pil_image(
                                self.pil_image,
                                target_width=target_width,
                                target_height=target_height)

        # Input image drag and drop support
        self.input_image.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.input_image.drag_dest_add_text_targets()
        self.input_image.connect('drag-data-received', self.input_image_drag_data_received)

        # Input image click support
        self.input_image_wrapper.add(self.input_image)
        self.input_image_wrapper.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.input_image_wrapper.connect("button-press-event",
                                        self.input_image_view_click_handler)

        # Clear button
        self.clear_button = Gtk.Button()
        icon = Gtk.Image.new_from_icon_name("window-close", Gtk.IconSize.BUTTON)
        self.clear_button.set_image(icon)
        self.clear_button.set_relief(Gtk.ReliefStyle.NONE)
        self.clear_button.set_size_request(20, 20)  # Set size of the button
        self.clear_button.set_size_request(20, 20)
        self.clear_button.set_halign(Gtk.Align.END)  # Align button to the end of the box
        self.clear_button.set_valign(Gtk.Align.START)  # Align button to the top of the box
        self.clear_button.connect("clicked", self.clear_image)
        self.pack_start(self.clear_button, False, False, 0)

    def _set_image_from_pil_image(self, pil_image: Image.Image) -> None:
        # Update the UI for input image
        resized_pil_image = resize_pil_image(pil_image, min(self.target_width, self.target_height))
        set_pil_image_to_gtk_image(resized_pil_image, self.input_image)

    def _set_image_from_file_path(self, image_path: str) -> None:
        """
        Sets image to self.pil_image and Gtk.Image.
        
        Args:
            image_path (str): PIL format image path.
        """
        if image_path:
            # Update the input image representation. Do not resize this yet.
            self.pil_image = Image.open(image_path)
            if self.callback:
                self.callback(self.pil_image)
            self._set_image_from_pil_image(self.pil_image)
            self.clear_button.show()
        else:
            pass  # No op (e.g. file was may not have been selected)

    def input_image_drag_data_received(self, 
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
        image_path = extract_file_path(data)
        self._set_image_from_file_path(image_path)

    def input_image_view_click_handler(self, widget, event):
        """
        Opens a file chooser that lets the user to choose the input image.
        """
        image_path = open_file_chooser_dialog(None, title="Select an image file")
        self._set_image_from_file_path(image_path)

    def clear_image(self, widget):
        """
        Clears the current image and sets self.pil_image to None.
        """
        self.pil_image = Image.new('RGBA', (self.target_width, self.target_height), "gray")
        # self.input_image.clear()
        self._set_image_from_pil_image(self.pil_image)
        if self.callback:
            self.callback(None)
        self.clear_button.hide()