"""
Copyright (C) 2024, Hideyuki Inada. All rights reserved.

Load image
        self.image_pixbuf = GdkPixbuf.Pixbuf.new_from_file(file_path)

Save image
        cropped.savev(filename, 'png', [], [])


Coordinate systems:
1. Widget coordinate:
  Physical coordinate as obtained by mouse down, mouse up, mouse drag
  All variables are prefixed with _w

Drawing coordinate:
  Logical coordinate which matches the actual image (e.g. 512 w x 768 h).
  This is used by Cairo.

matrix: tell Cairo to use API coordinate (Drawing coordinate) to convert to global window coordinate
e.g. if there is a menu bar, you have to tell focus to shift y by 25.

3. Global Window coordinate (0, 0) is the grid top-left (including menu)
Widget Window coordinate (0, 0) is the drawing area top left (excluding menu)

Mouse click:
1. Coordinates are in Widget window coordinate
2. Add menu height to convert to Global Window coordinate
3. Transform using inv matrix to drawing coordinate

x'    xx  xy  x0     x
y' =  yx  yy  y0  @  y
1      0   0   1     1

Argument order
xx, yx, xy, yy, x0, y0
"""
import os
import sys
import logging
from typing import Dict, Any, Optional, Callable
from io import BytesIO

import numpy as np
from PIL import Image
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf
import cairo

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path

from tool_base import ToolBase
from cremage.utils.image_utils import pil_image_to_pixbuf, pil_image_from_pixbuf

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def to3by3(m_in: np.ndarray):
    """
    Steps:
    Convert 2x3 to 3x3 matrix
    """
    return np.concatenate([m_in, np.array([[0, 0, 1]])], axis=0)

def invert_affine_matrix(m_in: np.ndarray):
    """
    Steps:
    Convert 2x3 to 3x3 matrix
    Invert
    Extract the 2x3 matrix   
    """
    m = np.concatenate([m_in, np.array([[0, 0, 1]])], axis=0)
    m = np.linalg.inv(m)
    return m[:-1,:]


def numpy_mat_to_cairo_mat(m: np.ndarray) -> cairo.Matrix:
    """
    It can handle both 3x3 and 2x3 matrix.
    TODO: Add shape check
    """
    return cairo.Matrix(
        m[0, 0],
        m[1, 0],
        m[0, 1],
        m[1, 1],
        m[0, 2],
        m[1, 2]
    )

class ImageCropper(ToolBase):  # Subclass Window object
    def __init__(
            self,
            title:str = "Image cropper",
            callback=None,
            aspect_ratio_w=None,
            aspect_ratio_h=None,
            **kwargs):
        super().__init__(title="Crop Tool", **kwargs)
        self.callback = callback
        self.aspect_ratio_w = aspect_ratio_w  # int
        self.aspect_ratio_h = aspect_ratio_h  # int
        
    def set_up_ui(self):
        """
        This method is called at the end of __init__ in ToolBase.
        Customization of UI should be done in this method.
        """
        self.drawing_area = Gtk.DrawingArea()  # canvas
        self.drawing_area.set_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK)
        
        self.drawing_area.connect('draw', self.on_draw)
        self.drawing_area.connect('button-press-event', self.on_button_press)
        self.drawing_area.connect('button-release-event', self.on_button_release)
        self.drawing_area.connect('motion-notify-event', self.on_motion_notify)

        self.image_pixbuf = None  # image buffer

        if self.pil_image is not None:
            self.load_image_from_pil_image(self.pil_image)
            logger.info("Set self.pil_image")
        else:
            logger.info("self.pil_image is None")

        self.crop_start_w = None  # xy pos
        self.crop_end_w = None # xy pos

        self.matrix = None
        self.inv_matrix = None

        # Setup drag and drop for the drawing area
        self.drawing_area.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.drawing_area.drag_dest_add_text_targets()
        self.drawing_area.connect('drag-data-received', self.on_drag_data_received)

        # self.create_menu()
        self.menubar = self.create_menu()
        self.menu_bar_height = None

        # Make sure the window can be focused to receive key press events
        self.set_can_focus(True)

        # # Connect the key press event signal to the handler
        # self.connect("key-press-event", self.on_key_press)

        # Use Gtk.Grid for layout
        self.drawing_area.set_hexpand(True)
        self.drawing_area.set_vexpand(True)

        grid = Gtk.Grid()
        grid.attach(self.menubar, 0, 0, 1, 1)  # Attach menu bar to grid, col pos, row pos, col count, row count
        grid.attach(self.drawing_area, 0, 1, 1, 1)  # Attach drawing area below menu bar

        self.crop_button = Gtk.Button(label="Crop")
        self.crop_button.connect("clicked", self.on_crop_clicked)
        grid.attach(self.crop_button, 0, 2, 1, 1)

        self.add(grid)  # Add grid to window

        # Load the image to canvas
        if self.pil_image:
            self.load_image_from_pil_image(self.pil_image)

    def on_menu_bar_realize(self, widget):
        allocation = widget.get_allocation()
        print(f"Menu bar height: {allocation.height}")
        self.menu_bar_height = allocation.height

    def on_draw(self, widget, cr):
        print(str(widget))
        # if self.menu_bar_height is None:
        #     return
        # You have to be very careful before you mess with matrix
        # as it changes whether it has a menubar or not!
        mat_cr_org = cr.get_matrix()
        print(mat_cr_org)  
        # without menubar: cairo.Matrix(1, 0, 0, 1, 0, 0)
        # with menubar: cairo.Matrix(1, 0, 0, 1, 0, 25)

        if self.image_pixbuf:  # if image buffer is not empty
            alloc = widget.get_allocation() # window coordinate
            width = self.image_pixbuf.get_width()  # image width in drawing (actual) coordinate
            height = self.image_pixbuf.get_height()  # image height in drawing (actual) coordinate
            
            scale = min(alloc.width / width, alloc.height / height)  # canvas width / image width

            # cr.scale(scale, scale)  # set up matrix?  so in cr, you use original 2048w 3076px?
            # convert screen to cairo
            # In cairo calls, always use actual image positions
            # e.g. if the image is 2048w, 3076, to draw a line use (0, 0) to (2048, 3076)

            # widget coordinate to global window coordinate
            translation_mat_np = np.array(
                [
                    [1, 0, mat_cr_org.x0],
                    [0, 1, mat_cr_org.y0],
                    [0, 0, 1]
                ]
            )

            self.translation_mat_np = translation_mat_np

            # drawing coordinate to widget coordinate
            scale_mat_np = np.array(
                [
                    [scale,     0, 0],
                    [    0, scale, 0],
                    [    0,     0, 1]
                ]
            )

            self.scale_mat_np = scale_mat_np

            # x2 = T@S@x1
            # Apply scaling first
            # Output: global window coordinate e.g. (0, 25)
            mat_np = translation_mat_np @ scale_mat_np
            self.inv_mat_np = np.linalg.inv(mat_np)
            mat_cr = numpy_mat_to_cairo_mat(mat_np)
            
            # Set the custom transformation matrix
            cr.set_matrix(mat_cr)
            self.matrix = mat_cr

            self.inv_matrix_np = np.linalg.inv(mat_np)
            self.inv_matrix = numpy_mat_to_cairo_mat(self.inv_matrix_np)      
          
            # Set image to cairo
            Gdk.cairo_set_source_pixbuf(cr, self.image_pixbuf, 0, 0)
            cr.paint()

            if self.crop_start_w and self.crop_end_w:
                # these are window coordinates, so you need to convert to drawing coordinates
                # You need to one more step as this is the coordinate based on
                # windows client including menu bar, so you need to add menu bar height

                p_start = np.array([
                    [self.crop_start_w[0]],
                    [self.crop_start_w[1]],
                    [1]  # dummy to make it 3x1
                ])

                p_end = np.array([
                    [self.crop_end_w[0]],
                    [self.crop_end_w[1]],
                    [1]  # dummy to make it 3x1
                ])

                # inv takes global window coordinate and output image coordinate
                # that you give to Cairo, so you need to convert widget coordinate
                # to window coordinate firt
                # widget coordinate to global window coordinate
                p_start = self.translation_mat_np @ p_start
                p_end = self.translation_mat_np @ p_end

                # Convert to global window coordinate to image coordinate
                p_start = self.inv_matrix_np @ p_start
                p_end = self.inv_matrix_np @ p_end

                cr.set_source_rgba(1, 0, 0, 0.5)  # Red color with transparency
                cr.rectangle(p_start[0, 0],
                             p_start[1, 0],
                             p_end[0, 0] - p_start[0, 0],
                             p_end[1, 0] - p_start[1, 0])
                cr.fill()

    # Mouse event handlers
    def on_button_press(self, widget, event):
        # You cannot convert to drawing coordinate as the mat may change
        # after the crop region is selected.
        self.crop_start_w = (event.x, event.y)
        self.crop_end_w = None
        print(self.crop_start_w)

    def adjust_end_y_pos(self):
        if self.aspect_ratio_h and self.aspect_ratio_w:
            width = self.crop_end_w[0] - self.crop_start_w[0]
            height = int(width * self.aspect_ratio_h / self.aspect_ratio_w)
            y_pos = self.crop_start_w[1] + height
            self.crop_end_w = (self.crop_end_w[0], y_pos)
    def on_button_release(self, widget, event):
        self.crop_end_w = (event.x, event.y)
        print(self.crop_end_w)
        self.adjust_end_y_pos()

        self.drawing_area.queue_draw()  # invalidate

    def on_motion_notify(self, widget, event):
        # If LMB drag
        if self.crop_start_w and (event.state & Gdk.ModifierType.BUTTON1_MASK):
            self.crop_end_w = (event.x, event.y)
            self.adjust_end_y_pos()
            self.drawing_area.queue_draw()  # invalidate

    # drag & drop handlers
    def on_drag_data_received(self, widget, drag_context, x, y, data, info, time):
        """Drag and Drop handler.

        data: Contains info for the dragged file name
        """
        logger.debug(" on_drag_data_received")
        file_path = data.get_text().strip()
        if file_path.startswith('file://'):
            file_path = file_path[7:]
        self.load_image(file_path)

    def load_image(self, file_path):
        """Load image from the specified file path"""
        logger.debug(f"file path: {file_path}")
        self.file_dir = os.path.dirname(file_path)
        self.file_stem = os.path.splitext(file_path)[0]
        self.file_ext = os.path.splitext(file_path)[1]
        # self.cropped_file_name = self.file_stem + "_cropped" + self.file_ext
        self.image_pixbuf = GdkPixbuf.Pixbuf.new_from_file(file_path)
        self.drawing_area.queue_draw()  # Invalidate

    def load_image_from_pil_image(self, image):
        """Load image from the PIL image"""
        # print(f"DEBUG: file path: {output_file_path}")
        # self.cropped_file_name = output_file_path
        self.image_pixbuf = pil_image_to_pixbuf(image)
        self.drawing_area.queue_draw()  # Invalidate

    def on_open_clicked(self, widget):
        super().on_open_clicked(widget)
        if self.pil_image:
            self.load_image_from_pil_image(self.pil_image)

    def on_crop_clicked(self, widget):
        if self.output_file_path:
            if self.image_pixbuf and self.crop_start_w and self.crop_end_w:
                self.output_pil_image = self.get_cropped_pil_image()
                if self.output_pil_image:
                    super().on_update_clicked(widget)
        else:
            if self.callback:
                self.callback(self.get_cropped_pil_image())
            else:
                self.on_save_clicked(widget)

    def on_update_clicked(self, widget):
        """
        Update caller menu item is selected
        """
        if self.image_pixbuf and self.crop_start_w and self.crop_end_w:
            self.output_pil_image = self.get_cropped_pil_image()
            if self.output_pil_image:
                super().on_update_clicked(widget)

    def on_save_clicked(self, widget):
        """
        Save menu item is selected
        """
        # if image buffer is not empty (image is loaded)
        # and crop start point
        # and crop end point are set
        if self.image_pixbuf and self.crop_start_w and self.crop_end_w:
            # Show file chooser dialog
            # Instantiate the choose
            self.output_pil_image = self.get_cropped_pil_image()
            if self.output_pil_image:
                super().on_save_clicked(widget)

    def get_cropped_pil_image(self):
        if not self.crop_start_w or not self.crop_end_w:
            return None

        p_start = np.array([
                [self.crop_start_w[0]],
                [self.crop_start_w[1]],
                [1]  # dummy to make it 3x1
            ])

        p_end = np.array([
            [self.crop_end_w[0]],
            [self.crop_end_w[1]],
            [1]  # dummy to make it 3x1
        ])

        # inv takes global window coordinate and output image coordinate
        # that you give to Cairo, so you need to convert widget coordinate
        # to window coordinate firt
        # widget coordinate to global window coordinate
        p_start = self.translation_mat_np @ p_start
        p_end = self.translation_mat_np @ p_end

        # Convert to global window coordinate to image coordinate
        p_start = self.inv_matrix_np @ p_start
        p_end = self.inv_matrix_np @ p_end

        crop_start_x = p_start[0, 0]
        crop_start_y = p_start[1, 0]
        crop_end_x = p_end[0, 0]
        crop_end_y = p_end[1, 0]       

        # Compute the top left positions        
        x, y = int(min(crop_start_x, crop_end_x)), int(min(crop_start_y, crop_end_y))

        # Compute the width and height
        width, height = int(abs(crop_end_x - crop_start_x)), int(abs(crop_end_y - crop_start_y))
        # Crop the image using ipb.new_subpixbuf. Note use pixbuf to mean image
        # new_subimage
        # cropped = self.image_pixbuf.new_subpixbuf(x, y, width, height)

        # Get the dimensions of the current image
        image_width = self.image_pixbuf.get_width()
        image_height = self.image_pixbuf.get_height()

        # Ensure x and y are within bounds
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))

        # Ensure width and height do not exceed the dimensions of the image
        width = min(width, image_width - x)
        height = min(height, image_height - y)

        # Create the subpixbuf with the adjusted dimensions
        cropped = self.image_pixbuf.new_subpixbuf(x, y, width, height)

        cropped_pil_image = pil_image_from_pixbuf(cropped)
        return cropped_pil_image
    

if __name__ == "__main__":
    app = ImageCropper()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()

