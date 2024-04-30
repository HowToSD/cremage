"""
Mask image editor.
Invoked from the main UI when edit mask button is pressed.
"""
import os
import sys
import logging
from dataclasses import dataclass
from typing import Tuple

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf
import cairo
import PIL
from PIL import Image

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
from cremage.utils.image_utils import pil_image_to_gtk_image
from cremage.utils.image_utils import pil_image_to_pixbuf, pil_image_from_pixbuf
from cremage.utils.image_utils import display_pil_image_from_mask_pil_image
from cremage.utils.image_utils import display_pil_image_to_mask_pil_image
from cremage.utils.image_utils import load_resized_pil_image
from cremage.utils.image_utils import get_png_paths
from cremage.utils.gtk_utils import text_view_set_text

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


class MaskImageEditor(Gtk.Window):
    """
    Mask image editor.

    Note:
        Mask file for both input and output shall contain black or white pixels.
        White pixels are for inpainting to generate.
        Black pixels should be untouched by inpainting.

        Upon input and loading, black pixels will be replaced with transparent pixels.
        Upon saving: transparent pixels will be replaced with black pixels.
    """

    def __init__(self,
                 base_image:Image=None,
                 mask_image:Image=None,
                 output_file_path:str="mask_image.png",
                 width:int=512,
                 height:int=768,
                 parent_window_update_func=None):
        super().__init__(title="Mask Image Editor")
        self.width = width
        self.height = height
        self.base_image = base_image  # PIL format image
        self.mask_image = mask_image
        if mask_image:
            logger.info("Converting mask image to transparency")
            self.mask_image = display_pil_image_from_mask_pil_image(self.mask_image)  # PIL format image
        self.output_file_path = output_file_path
        self.set_default_size(self.width, self.height)
        self.parent_window_update_func = parent_window_update_func
        self.pen_width = 10
        self.is_eraser = False

        # Layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Drawing area
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(width, height)
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

        self.lines_list = []  # Store drawing points

        # Slider for pen width
        self.slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1, 256, 1)
        self.slider.set_value(self.pen_width)
        self.slider.connect("value-changed", self.on_slider_value_changed)
        vbox.pack_start(self.slider, False, True, 0)

        # Button box
        button_box = Gtk.Box(spacing=6)
        vbox.pack_start(button_box, False, True, 0)

        # Load mask image button
        self.load_base_image_button = Gtk.Button.new_with_label("Load base")
        self.load_base_image_button.connect("clicked", self.on_load_base_image_clicked)
        button_box.pack_start(self.load_base_image_button, True, True, 0)

        # Load mask image button
        self.load_mask_image_button = Gtk.Button.new_with_label("Load mask")
        self.load_mask_image_button.connect("clicked", self.on_load_mask_image_clicked)
        button_box.pack_start(self.load_mask_image_button, True, True, 0)

        # Save mask image button
        self.save_button = Gtk.Button.new_with_label("Save")
        self.save_button.connect("clicked", self.on_save_clicked)
        button_box.pack_start(self.save_button, True, True, 0)

        # Clear current mask button
        self.clear_button = Gtk.Button.new_with_label("Clear")
        self.clear_button.connect("clicked", self.on_clear_clicked)
        button_box.pack_start(self.clear_button, True, True, 0)

        # Eraser checkbox - toggles between pen and eraser
        self.eraser_checkbox = Gtk.CheckButton(label="Eraser")
        self.eraser_checkbox.connect("toggled", self.on_eraser_checkbox_toggled)
        button_box.pack_start(self.eraser_checkbox, False, False, 0)

        # Set base image in pixbuf so that cr can use this to draw the base image first
        # during on paint handling.
        if self.base_image is None:
            self.base_image = Image.new('RGBA', (self.width, self.height), "black")
        self.pixbuf_with_base_image = pil_image_to_pixbuf(self.base_image)

        if self.mask_image:  # do the same for the mask image if any
            self.pixbuf_with_mask_image = pil_image_to_pixbuf(self.mask_image)

    def redraw_mask(self, cr):
        """
        lines_list: A list of lines
        lines: Lines is defined from a point from LMB down to LMB up
               It can have multiple segments.

        is_eraser, color, width are defined for each lines.

        Undo simply pops the lines_list
        """
        cr_save = cr
        # Create additional context
        surface2 = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        cr = cairo.Context(surface2)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)  # Or cairo.LINE_JOIN_BEVEL

        # Paint mask image that is loaded
        if self.mask_image:
            Gdk.cairo_set_source_pixbuf(cr, self.pixbuf_with_mask_image, 0, 0)
            cr.paint()  # render the content of mask image

        # Paint the mask image that the user is creating in this session
        if self.lines_list:
            cr.set_line_width(self.pen_width)  # Initial line width

            for i, lines in enumerate(self.lines_list):
                # Set color based on whether it's an eraser
                if lines.is_eraser:
                    cr.set_operator(cairo.OPERATOR_CLEAR)
                else:
                    cr.set_operator(cairo.OPERATOR_OVER)  # Default drawing
                    cr.set_source_rgba(1, 1, 1, 1)  # White color for the line

                cr.set_line_width(lines.pen_width)
                        
                cr.move_to(*lines.points[0])

                for p in (lines.points[1:]):
                    cr.line_to(*p)
                cr.stroke()

        # merge to the original
        cr_save.set_source_surface(surface2, 0, 0)
        cr_save.paint()

    def on_draw(self, widget, cr):
        # if self.base_image:
        #     base_image = self.base_image
        # else:
        #     base_image = Image.new('RGBA', (self.width, self.height), "black")

        # self.pixbuf_with_base_image = pil_image_to_pixbuf(base_image)   
        Gdk.cairo_set_source_pixbuf(cr, self.pixbuf_with_base_image, 0, 0)
        cr.paint()  # render the content of pixbuf in the source buffer on the canvas

        self.redraw_mask(cr)

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

    def create_dialog(self, title):
        dialog = Gtk.FileChooserDialog(title=title, parent=self,
                                       action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                           Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

        filter_png = Gtk.FileFilter()
        filter_png.set_name("PNG files")
        filter_png.add_mime_type("image/png")
        dialog.add_filter(filter_png)

        filter_jpg = Gtk.FileFilter()
        filter_jpg.set_name("JPG files")
        filter_jpg.add_mime_type("image/jpeg")
        dialog.add_filter(filter_jpg)
        return dialog

    def on_load_base_image_clicked(self, widget):
        """
        Handles the load mask image button press event.
        """
        dialog = self.create_dialog("Select the base image file")
        response = dialog.run()
        # Stores the result if a file is chosen
        if response == Gtk.ResponseType.OK:
            image_path = dialog.get_filename()
            self.base_image = Image.open(image_path)
            self.pixbuf_with_base_image = pil_image_to_pixbuf(self.base_image)

            # Get the new mask image size
            new_width, new_height = self.base_image.size
            
            # Update the drawing area and window size to match the new mask image
            self.drawing_area.set_size_request(new_width, new_height)
            
            # Adjust the main window size here if necessary
            self.set_default_size(new_width, new_height)
            self.resize(new_width, new_height)

            self.drawing_area.queue_draw()
        dialog.destroy()

    def on_load_mask_image_clicked(self, widget):
        """
        Handles the load mask image button press event.
        """
        dialog = self.create_dialog("Select the mask image file")

        response = dialog.run()
        # Stores the result if a file is chosen
        if response == Gtk.ResponseType.OK:
            mask_image_path = dialog.get_filename()
            self.mask_image = display_pil_image_from_mask_pil_image(Image.open(mask_image_path))
            assert self.mask_image.mode == 'RGBA'
            self.pixbuf_with_mask_image = pil_image_to_pixbuf(self.mask_image)
            self.lines_list.clear()  # Clear previous points

            # Get the new mask image size
            new_width, new_height = self.mask_image.size
            
            # Update the drawing area and window size to match the new mask image
            self.drawing_area.set_size_request(new_width, new_height)
            
            # Adjust the main window size here if necessary
            self.set_default_size(new_width, new_height)
            self.resize(new_width, new_height)

            self.drawing_area.queue_draw()
        dialog.destroy()

    def on_save_clicked(self, widget):
        if not self.lines_list:
            print("No drawing to save.")
            return

        # allocated_width = self.drawing_area.get_allocated_width()
        # allocated_height = self.drawing_area.get_allocated_height()

        # Create a new transparent surface
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                     self.width, # allocated_width,
                                     self.height) # allocated_height)
        cr = cairo.Context(surface)

        # Fill background with black
        black_background_image = Image.new('RGBA', (self.width, self.height), "black")
        pixbuf = pil_image_to_pixbuf(black_background_image)   
        Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
        cr.paint()       

        self.redraw_mask(cr)

        # Save the transparent surface (with drawings) to a PNG file
        surface.write_to_png(self.output_file_path)  # Saves only the overlay
        print(f"Mask image saved as {self.output_file_path}.")

        # if self.parent_window_update_func is not None and self.parent_instance is not None:
        #     self.parent_window_update_func(self.parent_instance, Image.open(self.output_file_path))
        if self.parent_window_update_func is not None:
            self.parent_window_update_func(Image.open(self.output_file_path))

    def on_clear_clicked(self, widget):
        self.mask_image=None
        self.lines_list.clear()
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
    app = MaskImageEditor()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()