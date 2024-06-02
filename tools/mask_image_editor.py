import os
import sys
import logging
from dataclasses import dataclass
from typing import Tuple

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
import cairo
import PIL
from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.image_utils import display_pil_image_from_mask_pil_image

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

@dataclass
class LinesData:
    points: Tuple[float, float]
    pen_width: float
    is_eraser: bool


class MaskImageEditor(Gtk.Window):
    def __init__(self,
                 base_image: Image = None,
                 mask_image: Image = None,
                 output_file_path: str = "mask_image.png",
                 width: int = 512,  # canvas
                 height: int = 768, # canvas
                 parent_window_update_func=None):
        super().__init__(title="Mask Image Editor")

        self.canvas_width = width
        self.canvas_height = height

        # Set base image in pixbuf so that cr can use this to draw the base image first
        # during on paint handling.
        if base_image is None:
            self.base_image = Image.new('RGBA', (self.canvas_width, self.canvas_height), "black")
            self.image_width = self.canvas_width
            self.image_height = self.canvas_height
        else:
            self.base_image = base_image  # PIL format image
            self.image_width, self.image_height = self.base_image.size

        self.pixbuf_with_base_image = pil_image_to_pixbuf(self.base_image)

        # Mask image
        self.mask_image = mask_image
        if mask_image:
            logger.info("Converting mask image to transparency")
            self.mask_image = display_pil_image_from_mask_pil_image(self.mask_image)  # PIL format image
            self.pixbuf_with_mask_image = pil_image_to_pixbuf(self.mask_image)

        self.output_file_path = output_file_path

        # Set app window size
        self.set_default_size(self.canvas_width, self.canvas_height)
        self.set_resizable(False)  # Make the window size not adjustable
        self.parent_window_update_func = parent_window_update_func
        self.pen_width = 10
        self.is_eraser = False

        self.translation = [0, 0]
        self.scale_factor = 1.0
        self.transform_matrix = cairo.Matrix()
        self.update_transform_matrix()

        # Layout
        # App window
        #  vbox (container)
        #    hbox (drawing area and vertical slider)
        #    horizontal slider
        #    container for push buttons
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Create a horizontal Box layout for the drawing area and vertical slider
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        vbox.pack_start(hbox, True, True, 0)

        # Add the drawing area to the horizontal box
        self.drawing_area = Gtk.DrawingArea()
        # This needs to be updated to match the size of the base image
        self.drawing_area.set_size_request(self.canvas_width, self.canvas_height)
        hbox.pack_start(self.drawing_area, False, False, 0)

        # Create the vertical slider for Y translation
        self.v_slider = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL)
        # self.v_slider.set_range(0, 1000)  # Set initial range
        self.v_slider.set_range(0, 0)  # Set initial range
        self.v_slider.set_value(0)
        # self.v_slider.set_inverted(True)
        self.v_slider.set_draw_value(False)  # Remove the number inside the slider
        self.v_slider.connect("value-changed", self.on_vscroll)
        hbox.pack_start(self.v_slider, False, False, 0)

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

        # Create the horizontal slider for X translation
        self.h_slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        # self.h_slider.set_range(0, 1000)  # Set initial range
        self.h_slider.set_range(0, 0)  # Set initial range
        self.h_slider.set_value(0)
        self.h_slider.set_draw_value(False)  # Remove the number inside the slider
        self.h_slider.connect("value-changed", self.on_hscroll)
        vbox.pack_start(self.h_slider, False, False, 0)

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

        # Add the zoom in button
        self.zoom_in_button = Gtk.Button(label="+")
        self.zoom_in_button.connect("clicked", self.on_zoom_in)
        button_box.pack_start(self.zoom_in_button, False, False, 0)

        # Add the zoom out button
        self.zoom_out_button = Gtk.Button(label="-")
        self.zoom_out_button.connect("clicked", self.on_zoom_out)
        button_box.pack_start(self.zoom_out_button, False, False, 0)

        # Add the reset button
        self.reset_button = Gtk.Button(label="Reset")
        self.reset_button.connect("clicked", self.on_reset)
        button_box.pack_start(self.reset_button, False, False, 0)

        # End UI layout
        self._adjust_matrix_redraw_canvas()

    def update_transform_matrix(self):
        """
        Update the transformation matrix with translation and scaling.
        """
        self.transform_matrix = cairo.Matrix()  # Create an identity matrix
        self.transform_matrix.translate(self.translation[0], self.translation[1])
        self.transform_matrix.scale(self.scale_factor, self.scale_factor)

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
        surface2 = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.image_width, self.image_height)
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
        cr.set_matrix(self.transform_matrix)
        Gdk.cairo_set_source_pixbuf(cr, self.pixbuf_with_base_image, 0, 0)
        cr.paint()  # render the content of pixbuf in the source buffer on the canvas

        self.redraw_mask(cr)

    def screen_to_cairo_coord(self, x, y) -> Tuple[int, int]:
        """
        Converts screen coordinates to Cairo coordinates.

        Screen coordinates are physical coordinates used in mouse events.
        Cairo coordinates are logical coordinate used for Cairo drawing.

        Args:
            x (int): x in screen coordinates
            y (int): y in screen coordinates
        Returns:
            Tuple of x, y in Cairo coordinates
        """
        inv_transform = cairo.Matrix(*self.transform_matrix)
        inv_transform.invert()  # screen coordinates to cairo logical coordinates
        x_logical, y_logical = inv_transform.transform_point(x, y)
        return x_logical, y_logical
    
    def on_button_press(self, widget, event):
        """
        Handles the LMB press event.
        """
        x_logical, y_logical = self.screen_to_cairo_coord(event.x, event.y)

        # Create new lines data
        ld = LinesData(pen_width=self.pen_width,
                       is_eraser=self.is_eraser,
                       points=[(x_logical, y_logical)])
        self.lines_list.append(ld)

    def on_motion_notify(self, widget, event):
        """
        Handles the LMB drag event.
        """
        x_logical, y_logical = self.screen_to_cairo_coord(event.x, event.y)

        if event.state & Gdk.ModifierType.BUTTON1_MASK:
            current_lines_data_index = len(self.lines_list) - 1
            current_lines_data = self.lines_list[current_lines_data_index]
            current_lines_data.points.append((x_logical, y_logical))
            self.drawing_area.queue_draw()

    def on_button_release(self, widget, event):
        """
        Handles the LMB release event.
        """
        x_logical, y_logical = self.screen_to_cairo_coord(event.x, event.y)

        current_lines_data_index = len(self.lines_list) - 1
        current_lines_data = self.lines_list[current_lines_data_index]
        current_lines_data.points.append((x_logical, y_logical))
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
            self.image_width, self.image_height = self.base_image.size
            canvas_edge_length = 768  # FIXME
            
            # Compute the new canvas dimension
            if self.image_width > self.image_height:  # landscape
                new_width = canvas_edge_length
                new_height = int(canvas_edge_length * self.image_height / self.image_width)
            else: # portrait
                new_height = canvas_edge_length
                new_width = int(canvas_edge_length * self.image_width / self.image_height)

            self.canvas_width = new_width
            self.canvas_height = new_height

            # Update the canvas size to match the new base image
            self.drawing_area.set_size_request(new_width, new_height)
            
            # Adjust the main window size here if necessary
            self.set_default_size(new_width, new_height)
            self.resize(new_width, new_height)

            self._adjust_matrix_redraw_canvas()
            # self.drawing_area.queue_draw()
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
            mask_image = Image.open(mask_image_path)
            self.mask_image = mask_image.resize(self.base_image.size, Image.LANCZOS)
            self.mask_image = display_pil_image_from_mask_pil_image(self.mask_image)
            assert self.mask_image.mode == 'RGBA'
            self.pixbuf_with_mask_image = pil_image_to_pixbuf(self.mask_image)
            self.lines_list.clear()  # Clear previous points
            self.drawing_area.queue_draw()
        dialog.destroy()

    def on_save_clicked(self, widget):
        if not self.lines_list:
            print("No drawing to save.")
            return

        # Create a new transparent surface
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                     self.image_width, # allocated_width,
                                     self.image_height) # allocated_height)
        cr = cairo.Context(surface)

        # Fill background with black
        black_background_image = Image.new('RGBA', (self.image_width, self.image_height), "black")
        pixbuf = pil_image_to_pixbuf(black_background_image)   
        Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
        cr.paint()       

        self.redraw_mask(cr)

        # Save the transparent surface (with drawings) to a PNG file
        surface.write_to_png(self.output_file_path)  # Saves only the overlay
        print(f"Mask image saved as {self.output_file_path}.")

        if self.parent_window_update_func is not None:
            self.parent_window_update_func(Image.open(self.output_file_path))

    def on_clear_clicked(self, widget):
        self.mask_image = None
        self.lines_list.clear()
        self.drawing_area.queue_draw()

    def on_slider_value_changed(self, slider):
        self.pen_width = slider.get_value()
        logger.debug(f"Pen width: {self.pen_width}")

    def on_eraser_checkbox_toggled(self, checkbox):
        self.is_eraser = checkbox.get_active()

    def _adjust_matrix_redraw_canvas(self):
        self.update_transform_matrix()
        self.drawing_area.queue_draw()
        self.update_slider_ranges()        

    def on_zoom_in(self, button):
        """
        Scale up
        """
        self.scale_factor *= 1.1
        self._adjust_matrix_redraw_canvas()

    def on_zoom_out(self, button):
        """
        Scale down
        """
        self.scale_factor /= 1.1
        self._adjust_matrix_redraw_canvas()

    def on_reset(self, button):
        self.scale_factor = 1.0
        self.translation = [0, 0]
        self._adjust_matrix_redraw_canvas()

    def update_slider_ranges(self):
        """
        Called when the image is scaled.
        """
        image_width_scaled = self.image_width * self.scale_factor
        image_height_scaled = self.image_height * self.scale_factor

        self.h_slider.set_range(0, max(image_width_scaled - self.canvas_width, 0))
        self.h_slider.set_value(0)
        self.v_slider.set_range(0, max(image_height_scaled - self.canvas_height, 0))
        self.v_slider.set_value(0)

    def on_hscroll(self, adjustment):
        self.translation[0] = -adjustment.get_value()
        self.update_transform_matrix()
        self.drawing_area.queue_draw()

    def on_vscroll(self, adjustment):
        self.translation[1] = -adjustment.get_value()
        self.update_transform_matrix()
        self.drawing_area.queue_draw()


def main():
    app = MaskImageEditor()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
