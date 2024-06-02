import os
import sys
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
import cairo

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.gtk_utils import create_surface_from_pil

DRAWING_AREA_WIDTH = 500
DRAWING_AREA_HEIGHT = 750

class DrawingArea(Gtk.DrawingArea):
    """
    Defines the canvas to display an image.
    """
    def __init__(self, file_name=None, pil_image=None):
        super().__init__()
        self.set_size_request(DRAWING_AREA_WIDTH, DRAWING_AREA_HEIGHT)  # Set the initial size of the drawing area
        self.connect("draw", self.on_draw)

        if pil_image:
            self.image_surface = create_surface_from_pil(pil_image)
        else: # file_name
            self.image_surface = cairo.ImageSurface.create_from_png(file_name)

        # Grab the image size here
        self.image_width = self.image_surface.get_width()
        self.image_height = self.image_surface.get_height()

        # Compute the scale factor
        self.scale_factor = min(DRAWING_AREA_WIDTH / self.image_width,
                                DRAWING_AREA_HEIGHT / self.image_height)

        self.translation = [0, 0]
        self.transform_matrix = cairo.Matrix()
        self.update_transform_matrix()

    def update_transform_matrix(self):
        self.transform_matrix = cairo.Matrix()  # This creates an identity matrix
        self.transform_matrix.translate(self.translation[0], self.translation[1])
        self.transform_matrix.scale(self.scale_factor, self.scale_factor)  # if 0.5, set s1 and s2 to 0.5

    def on_draw(self, widget, cr):
        cr.set_matrix(self.transform_matrix)
        cr.set_source_surface(self.image_surface, 0, 0)
        cr.paint()  # Copy from source surface to screen

    def update_size_request(self):
        """
        Called when zoom in, zoom out, size reset buttons are pressed.

        self.image_surface.get_width() is the actual image width (e.g. 2048px w)
        image_width is the size required to show the scaled image.
        For example, if the actual image is 2400px and scale_factor is 0.5,
        then 1200px is required to show the image.
        """
        image_width = self.image_surface.get_width() * self.scale_factor
        image_height = self.image_surface.get_height() * self.scale_factor
        self.set_size_request(int(image_width), int(image_height))

    def set_translation(self, x, y):
        self.translation = [x, y]
        self.update_transform_matrix()
        self.queue_draw()


class ImageViewer(Gtk.Window):

    def __init__(self, file_name=None, pil_image=None):
        """
        
        Args:
            pil_image: PIL image
            file_name: Image file name. If both pil_image and file_name are specified,
                pil_image takes precedence.
        
        """

        if pil_image:
            title = "Image viewer"
        else:
            title = os.path.basename(file_name) if file_name else ""
        super().__init__(title=title)
        self.set_default_size(600, 800)  # Ensure the window is large enough

        # Create a vertical Box layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Create a scrolled window
        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_policy(Gtk.PolicyType.ALWAYS, Gtk.PolicyType.ALWAYS)
        vbox.pack_start(self.scrolled_window, True, True, 0)

        # Add the drawing area to the scrolled window
        self.drawing_area = DrawingArea(file_name=file_name, pil_image=pil_image)
        self.scrolled_window.add(self.drawing_area)

        # Create a horizontal Box for the buttons
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        vbox.pack_start(hbox, False, False, 0)

        # Add the zoom in button
        zoom_in_button = Gtk.Button(label="+")
        zoom_in_button.connect("clicked", self.on_zoom_in)
        hbox.pack_start(zoom_in_button, False, False, 0)

        # Add the zoom out button
        zoom_out_button = Gtk.Button(label="-")
        zoom_out_button.connect("clicked", self.on_zoom_out)
        hbox.pack_start(zoom_out_button, False, False, 0)

        # Add the reset button
        reset_button = Gtk.Button(label="Reset")
        reset_button.connect("clicked", self.on_reset)
        hbox.pack_start(reset_button, False, False, 0)

        # Add the label to display the current image size
        self.size_label = Gtk.Label(label=f"w:{self.drawing_area.image_width}px, h:{self.drawing_area.image_height}px")
        hbox.pack_start(self.size_label, False, False, 0)

        # Add the label to display the scaling factor
        self.scale_label = Gtk.Label(label="100%")
        hbox.pack_start(self.scale_label, False, False, 0)

        # Connect scrollbars adjustments
        hadj = self.scrolled_window.get_hadjustment()
        vadj = self.scrolled_window.get_vadjustment()
        hadj.connect("value-changed", self.on_hscroll)
        vadj.connect("value-changed", self.on_vscroll)

    def on_zoom_in(self, button):
        self.drawing_area.scale_factor *= 1.1
        self.drawing_area.update_transform_matrix()
        self.drawing_area.update_size_request()
        self.drawing_area.queue_draw()
        self.update_labels()

    def on_zoom_out(self, button):
        self.drawing_area.scale_factor /= 1.1
        self.drawing_area.update_transform_matrix()
        self.drawing_area.update_size_request()
        self.drawing_area.queue_draw()
        self.update_labels()

    def on_reset(self, button):
        self.drawing_area.scale_factor = 1.0
        self.drawing_area.translation = [0, 0]
        self.drawing_area.update_transform_matrix()
        self.drawing_area.update_size_request()
        self.drawing_area.queue_draw()
        self.update_labels()

    def update_labels(self):
        self.scale_label.set_text(f"{int(self.drawing_area.scale_factor * 100)}%")

    def on_hscroll(self, adjustment):
        self.drawing_area.set_translation(-adjustment.get_value(), self.drawing_area.translation[1])  # x, y

    def on_vscroll(self, adjustment):
        self.drawing_area.set_translation(self.drawing_area.translation[0], -adjustment.get_value())  # x, y


if __name__ == "_main__":
    win = ImageViewer(file_name="input_image.png")
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
