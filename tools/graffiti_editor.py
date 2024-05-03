"""
Mask image editor.
Invoked from the main UI when edit mask button is pressed.
"""
import os
import sys
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf
import cairo
import PIL
from PIL import Image, ImageOps

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

sys.path = [PROJECT_ROOT, MODULE_ROOT, TOOLS_ROOT] + sys.path
from cremage.utils.image_utils import pil_image_to_gtk_image
from cremage.utils.image_utils import pil_image_to_pixbuf, pil_image_from_pixbuf
from cremage.utils.image_utils import pil_image_to_binary_pil_image
from cremage.utils.gtk_utils import save_pil_image_by_file_chooser_dialog
from cremage.utils.gtk_utils import open_file_chooser_dialog

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

@dataclass
class LinesData:
    """
    Represents a drawing command for a point in a path.

    Attributes:
        is_move_to (bool): Indicates whether the point is the start of a new sub-path.
            True for `move_to`, False for `line_to`.
        point (Tuple[float, float]): The (x, y) coordinates of the point.
    """
    points: Tuple[float, float]
    pen_width: float
    is_eraser: bool


DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 768

class GraffitiEditor(Gtk.Window):
    """
    Graffiti editor to edit a binary graffiti image.
    """
    def __init__(self, pil_image=None,
                 output_file_path=None,
                 save_call_back=None,
                 generation_information_call_back=None,
                 preferences=None):

        super().__init__(title="Graffiti Editor")
        self.pil_image = pil_image  # PIL format image
        self.save_call_back = save_call_back
        self.output_file_path = output_file_path
        self.preferences = dict() if preferences is None else preferences
        self.generation_information_call_back = generation_information_call_back    
        self.width = int(self.preferences["image_width"]) if "image_width" in self.preferences else DEFAULT_WIDTH
        self.height = int(self.preferences["image_height"]) if "image_height" in self.preferences else DEFAULT_HEIGHT
        self.set_default_size(self.width, self.height)
        self.pen_width = 1
        self.is_eraser = False
        self.lines_list = []  # Store drawing points

        # Create a offscreen drawing cr
        if self.pil_image is None:
            self.pil_image = Image.new('RGBA', (self.width, self.height), "white")
        else:
            self.pil_image = pil_image_to_binary_pil_image(self.pil_image)
        self.buffer_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        self.buffer_cr = cairo.Context(self.buffer_surface)

        # Paint initial image on offscreen cr.
        pixbuf = pil_image_to_pixbuf(self.pil_image)   
        Gdk.cairo_set_source_pixbuf(self.buffer_cr, pixbuf, 0, 0)
        self.buffer_cr.paint()  # render the content of pixbuf on cr

        # Layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Create menu
        menubar = self.create_menu()
        vbox.pack_start(menubar, False, False, 0)

        # Drawing area
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(self.width, self.height)
        vbox.pack_start(self.drawing_area, True, True, 0)

        # Event handling for drawing
        self.drawing_area.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK)
        
        self.drawing_area.connect("button-press-event", self.on_button_press)  # LMB click
        self.drawing_area.connect("motion-notify-event", self.on_motion_notify)  # LMB drag
        self.drawing_area.connect("button-release-event", self.on_button_release)  # LMB release
        self.drawing_area.connect("draw", self.on_draw)

        # Slider for pen width
        self.slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1, 150, 1)
        self.slider.set_value(self.pen_width)
        self.slider.connect("value-changed", self.on_slider_value_changed)
        vbox.pack_start(self.slider, False, True, 0)

        # Button box
        button_box = Gtk.Box(spacing=6)
        vbox.pack_start(button_box, False, True, 0)

        if save_call_back:
            # Send control image to the caller
            self.update_caller_button = Gtk.Button.new_with_label("Copy to ControlNet")
            self.update_caller_button.connect("clicked", self.on_update_caller_clicked)
            button_box.pack_start(self.update_caller_button, True, True, 0)            

        # Invert button
        self.invert_button = Gtk.Button.new_with_label("Invert")
        self.invert_button.connect("clicked", self.on_invert_clicked)
        button_box.pack_start(self.invert_button, True, True, 0)

        # Clear button
        self.clear_button = Gtk.Button.new_with_label("Clear")
        self.clear_button.connect("clicked", self.on_clear_clicked)
        button_box.pack_start(self.clear_button, True, True, 0)

        # Eraser checkbox - toggles between pen and eraser
        self.eraser_checkbox = Gtk.CheckButton(label="Eraser")
        self.eraser_checkbox.connect("toggled", self.on_eraser_checkbox_toggled)
        button_box.pack_start(self.eraser_checkbox, False, False, 0)

    def create_menu(self):
        """
        Create a MenuBar
        """
        ## Accelerator
        accel_group = Gtk.AccelGroup()
        self.add_accel_group(accel_group)
        
        menubar = Gtk.MenuBar()

        # File menu items
        filemenu = Gtk.Menu()
        file_item = Gtk.MenuItem(label="File")
        file_item.set_submenu(filemenu)

        # File | Open
        open_item = Gtk.MenuItem(label="Open")  # Create open menu item
        open_item.connect("activate", self.on_open_clicked)  # Connect to handler
        open_item.add_accelerator("activate", accel_group, ord('O'),
                                Gdk.ModifierType.CONTROL_MASK, Gtk.AccelFlags.VISIBLE)
        filemenu.append(open_item)  # Add open item to file menu

        # File | Save
        save_item = Gtk.MenuItem(label="Save")
        save_item.connect("activate", self.on_save_clicked)
        save_item.add_accelerator("activate", accel_group, ord('S'),
                                Gdk.ModifierType.CONTROL_MASK, Gtk.AccelFlags.VISIBLE)
        filemenu.append(save_item)
        
        # File | Exit
        exit_item = Gtk.MenuItem(label="Exit")
        exit_item.connect("activate", self.close_window)
        filemenu.append(exit_item)

        menubar.append(file_item)
        return menubar

    def close_window(self, widget):
        self.close()

    def paint_drawing_area(self, cr):
        """
        Paints the drawing area.

        Note:
          lines_list: A list of lines
          lines: Lines is defined from a point from LMB down to LMB up
                 It can have multiple segments.
          is_eraser, color, width are defined for each lines.
        """
        cr.set_line_join(cairo.LINE_JOIN_ROUND)  # Or cairo.LINE_JOIN_BEVEL

        if self.lines_list:
            cr.set_line_width(self.pen_width)  # Initial line width
            for i, lines in enumerate(self.lines_list):
                # Set color based on whether it's an eraser
                if lines.is_eraser:
                    # cr.set_operator(cairo.OPERATOR_CLEAR)
                    cr.set_operator(cairo.OPERATOR_OVER)  # Default drawing
                    cr.set_source_rgba(1, 1, 1, 1)  # White color for the line
                else:
                    cr.set_operator(cairo.OPERATOR_OVER)  # Default drawing
                    cr.set_source_rgba(0, 0, 0, 1)  # White color for the line
                cr.set_line_width(lines.pen_width)
                cr.move_to(*lines.points[0])
                for p in (lines.points[1:]):
                    cr.line_to(*p)
                cr.stroke()

    def on_draw(self, widget, cr):
        """
        Repaints the drawing area. This is invoked by Gtk.
        """
        # Draw on offscreen cr first
        self.paint_drawing_area(self.buffer_cr)

        # Copy from off-screen cr to cr
        cr.set_source_surface(self.buffer_surface, 0, 0)  # Specify offscreen as the source
        cr.paint()  # Paint from offscreen cr to cr

    def on_button_press(self, widget, event):
        """
        Handles the LMB press event.
        """
        # Create new lines data
        ld = LinesData(pen_width=self.pen_width,
                       is_eraser=self.is_eraser,
                       points=[(event.x, event.y)])
        self.lines_list.append(ld)

    def on_motion_notify(self, widget, event):
        """
        Handles the LMB drag event.
        """
        if event.state & Gdk.ModifierType.BUTTON1_MASK:
            current_lines_data_index = len(self.lines_list) - 1
            current_lines_data = self.lines_list[current_lines_data_index]
            current_lines_data.points.append((event.x, event.y))
            self.drawing_area.queue_draw()

    def on_button_release(self, widget, event):
        """
        Handles the LMB release event
        """
        current_lines_data_index = len(self.lines_list) - 1
        current_lines_data = self.lines_list[current_lines_data_index]
        current_lines_data.points.append((event.x, event.y))
        self.drawing_area.queue_draw()
 
    def on_open_clicked(self, widget):
        """
        Handles the load mask image button press event.
        """
        image_path = open_file_chooser_dialog(self, title="Select an image file")
        if image_path:
            self.pil_image = Image.open(image_path)
            self.pil_image = pil_image_to_binary_pil_image(self.pil_image)

            # Get the new mask image size
            new_width, new_height = self.pil_image.size
            
            # Update the drawing area and window size to match the new mask image
            self.drawing_area.set_size_request(new_width, new_height)
            
            # Adjust the main window size here if necessary
            self.set_default_size(new_width, new_height)
            self.resize(new_width, new_height)
            self.width = new_width
            self.height = new_height

            self.buffer_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
            self.buffer_cr = cairo.Context(self.buffer_surface)

            # Paint initial image on offscreen cr.
            pixbuf = pil_image_to_pixbuf(self.pil_image)   
            Gdk.cairo_set_source_pixbuf(self.buffer_cr, pixbuf, 0, 0)
            self.buffer_cr.paint()  # render the content of pixbuf on cr
            self.drawing_area.queue_draw()

    def on_save_clicked(self, widget):
        width, height = self.buffer_surface.get_width(), self.buffer_surface.get_height()
        data = self.buffer_surface.get_data().tobytes()  # Convert memoryview to bytes

        # Convert to a PIL Image
        img = Image.frombuffer("RGBA", (width, height), data, "raw", "BGRA", 0, 1)
        img = img.convert("L")  # Convert to grayscale
        file_path = save_pil_image_by_file_chooser_dialog(self, img)
        if file_path and self.output_file_path != None:
            self.output_file_path = file_path
            if self.save_call_back is not None:
                self.save_call_back(img)

    def on_update_caller_clicked(self, widget):
        width, height = self.buffer_surface.get_width(), self.buffer_surface.get_height()
        data = self.buffer_surface.get_data().tobytes()  # Convert memoryview to bytes

        # Convert to a PIL Image
        img = Image.frombuffer("RGBA", (width, height), data, "raw", "BGRA", 0, 1)
        img = img.convert("L")  # Convert to grayscale
        
        assert self.output_file_path is not None
        assert self.save_call_back is not None

        img.save(self.output_file_path)
        self.save_call_back(img)

    def on_invert_clicked(self, widget):
        self.lines_list.clear()

        # Create pil from buffer
        width, height = self.buffer_surface.get_width(), self.buffer_surface.get_height()
        data = self.buffer_surface.get_data().tobytes()  # Convert memoryview to bytes

        # Convert to a PIL Image
        img = Image.frombuffer("RGBA", (width, height), data, "raw", "BGRA", 0, 1)
        img = img.convert("L")
        img = ImageOps.invert(img)
        self.pil_image = img.convert("RGBA")
        
        # Write back to Cairo
        pixbuf = pil_image_to_pixbuf(self.pil_image)   
        Gdk.cairo_set_source_pixbuf(self.buffer_cr, pixbuf, 0, 0)
        self.buffer_cr.paint()  # render the content of pixbuf on cr
        self.drawing_area.queue_draw()

    def on_clear_clicked(self, widget):
        self.pil_image=None
        self.lines_list.clear()
        self.buffer_cr.set_source_rgb(1, 1, 1)  # White color
        self.buffer_cr.paint()
        self.drawing_area.queue_draw()

    def on_slider_value_changed(self, slider):
        self.pen_width = slider.get_value()
        print(f"Pen width: {self.pen_width}")

    def on_eraser_checkbox_toggled(self, checkbox):
        if checkbox.get_active():
            self.is_eraser = True
        else:
            self.is_eraser = False
           

def main():
    app = GraffitiEditor()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()