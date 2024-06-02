"""
Copyright (C) 2024, Hideyuki Inada. All rights reserved.

Algorithm
1 Identify masked regions
2 For each region,
2.1 resize so that the longest edge is the target edge length
2.2 keep the aspect ratio.
2.2 Pad the image so that image size will be target edge wide, target edge high.
2.3 Send to image to image
2.4 Remove the padding
2.5 Resize to the original dimension
2.6 Paste the updated image in the original image.

Mask editor produces connected regions in RGBA PIL image format:
(0, 0, 0, 255): Unmasked
(255, 255, 255, 255): Masked

The goal is to identify the bounding box for each mask which is a connected region.
"""
import os
import sys
import logging
import tempfile
import threading
import shutil
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import cv2 as cv
import PIL
from PIL import Image
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk
import cairo

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
from sd.img2img import img2img_parse_options_and_generate
from cremage.utils.image_utils import pil_image_to_pixbuf, get_bounding_boxes_from_grayscale_image
from cremage.utils.gtk_utils import show_alert_dialog
from cremage.utils.gtk_utils import text_view_get_text, create_combo_box_typeahead
from cremage.utils.misc_utils import generate_lora_params
from cremage.utils.misc_utils import get_tmp_dir
from cremage.ui.model_path_update_handler import update_ldm_model_name_value_from_ldm_model_dir

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

TARGET_EDGE_LEN = 512  # FIXME
SPOT_FIX_TMP_DIR = os.path.join(get_tmp_dir(), "spot_fix.tmp")
SELECTED_COLOR =  (0, 0x7B/255.0, 0xFF/255.0, 0.5) # "#007BFF"
UNSELECTED_COLOR = (0xD3/255.0, 0xD3/255.0, 0xD3/255.0, 0.5)

@dataclass
class BoxRect:
    left: int
    top: int
    right: int
    bottom: int
    
# TODO: Consolidate with mask editor
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


class SpotInpainter(Gtk.Window):  # Subclass Window object

    def __init__(self, pil_image=None,
                 output_file_path=None,
                 save_call_back=None,
                 positive_prompt=None,
                 negative_prompt=None,
                 generation_information_call_back=None,
                 preferences=None,
                 status_queue=None):
        """
        Args:
            pil_image: An image to inpaint.
            procedural (bool): True if called from img2img. False if invoked on the UI
        """
        super().__init__(title="Spot inpainting")

        self.show_mask = True  # To display mask (area to be inpainted) on canvas
        self.show_bounding_box = False  # To display a bounding box around the mask
        self.bounding_boxes = []
        self.selected_bounding_box_index = None
        self.prev_x = None  # LMB press position
        self_prev_y = None
        self.output_file_path = output_file_path
        self.save_call_back = save_call_back
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.preferences = dict() if preferences is None else preferences
        self.generation_information_call_back = generation_information_call_back
        self.pen_width = 10
        self.is_eraser = False
        self.pil_image = pil_image  # PIL format image
        self.status_queue = status_queue
        self.ldm_model_names = None # This is populated in update_ldm_model_name_value_from_ldm_model_dir
        self.enable_lora = False
        update_ldm_model_name_value_from_ldm_model_dir(self)

        self.generation_information = dict()
        if self.generation_information_call_back is not None:
            d = self.generation_information_call_back()
            if d:  # The original image may not have generation info
                self.generation_information = d

        # Create an Image widget
        if pil_image is None:
            pil_image = Image.new('RGBA', (512, 768), "gray")
        self.pil_image = pil_image
        self.pil_image_original = self.pil_image  # Make a copy to restore

        # Put placeholder values - These are re-computed at the end of this method
        self.image_width, self.image_height = self.pil_image.size
        canvas_edge_length = 768
        self.canvas_width = canvas_edge_length
        self.canvas_height = canvas_edge_length
        self.set_default_size(self.canvas_width + 200, self.canvas_height)  # This will be re-computed too.

        self.set_border_width(10)

        # Create a vertical Gtk.Box
        root_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(root_box)
 
        self.set_up_menu(root_box)
        self.set_up_ui(root_box)

    def set_up_menu(self, menu_container):
       # Create a MenuBar
        ## Accelerator
        accel_group = Gtk.AccelGroup()
        self.add_accel_group(accel_group)
        
        menubar = Gtk.MenuBar()

        # File menu items
        filemenu = Gtk.Menu()
        file_item = Gtk.MenuItem(label="File")
        file_item.set_submenu(filemenu)

        # File | Save
        save_item = Gtk.MenuItem(label="Save")  # Create save menu item
        save_item.connect("activate", self.on_save_activate)  # Connect to handler
        filemenu.append(save_item)  # Add save item to file menu

        # File | Exit
        exit_item = Gtk.MenuItem(label="Exit")
        exit_item.connect("activate", Gtk.main_quit)
        filemenu.append(exit_item)

        menubar.append(file_item)
        menu_container.pack_start(menubar, False, False, 0)

    def set_up_ui(self, root_box):
        # Horizontal Gtk.Box to contain the scrolled window and control elements
        container_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root_box.pack_start(container_box, True, True, 0)  # Add container_box to root_box under the menu

        # Create a horizontal Box layout for the drawing area and vertical slider
        drawing_area_wrapper_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        container_box.pack_start(drawing_area_wrapper_box, True, True, 0)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        drawing_area_wrapper_box.pack_start(hbox, True, True, 0)

        # Create a Gtk.Image and set the Pixbuf
        self.drawing_area = Gtk.DrawingArea()

        # Setup drag and drop for the image area
        # Click support
        self.drawing_area.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK)

        self.drawing_area.connect("button-press-event", self.on_button_press)  # LMB click
        self.drawing_area.connect("motion-notify-event", self.on_motion_notify)  # LMB drag
        self.drawing_area.connect("button-release-event", self.on_button_release)  # LMB release
        self.drawing_area.connect("draw", self.on_draw)

        # Drag & drop support
        self.drawing_area.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.drawing_area.drag_dest_add_text_targets()
        self.drawing_area.connect('drag-data-received', self.on_drag_data_received)

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

        # Create the horizontal slider for X translation
        self.h_slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        # self.h_slider.set_range(0, 1000)  # Set initial range
        self.h_slider.set_range(0, 0)  # Set initial range
        self.h_slider.set_value(0)
        self.h_slider.set_draw_value(False)  # Remove the number inside the slider
        self.h_slider.connect("value-changed", self.on_hscroll)
        drawing_area_wrapper_box.pack_start(self.h_slider, False, False, 0)

        self.lines_list = []  # Store drawing points

        # Vertical Box for controls next to the ScrolledWindow
        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        container_box.pack_start(controls_box, False, True, 0)
   
        #
        # Apply inpainting
        #
        apply_inpainting_button = Gtk.Button(label="Apply inpainting")
        controls_box.pack_start(apply_inpainting_button, False, True, 0)
        apply_inpainting_button.connect("clicked", self.on_apply_inpainting_clicked)

        #
        # Clear marks button
        #
        clear_marks_button = Gtk.Button(label="Clear masks")
        controls_box.pack_start(clear_marks_button, False, True, 0)
        clear_marks_button.connect("clicked", self.on_clear_marks_clicked)

        # Eraser checkbox - toggles between pen and eraser
        self.eraser_checkbox = Gtk.CheckButton(label="Eraser")
        self.eraser_checkbox.connect("toggled", self.on_eraser_checkbox_toggled)
        controls_box.pack_start(self.eraser_checkbox, False, False, 0)

        # Show bounding box checkbox
        self.show_bounding_box_checkbox = Gtk.CheckButton(label="Show bounding boxes")
        self.show_bounding_box_checkbox.connect("toggled", self.on_show_bounding_box_toggled)
        controls_box.pack_start(self.show_bounding_box_checkbox, False, False, 0)

        # Show mask checkbox
        self.show_mask_checkbox = Gtk.CheckButton(label="Show mask")
        self.show_mask_checkbox.connect("toggled", self.on_show_mask_toggled)
        controls_box.pack_start(self.show_mask_checkbox, False, False, 0)
        self.show_mask_checkbox.set_active(True)

        # LDM model
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        root_box.pack_start(box, False, False, 0)
        model_name = self.preferences["ldm_model"]
        # Check to see if we have generation info to override
        if "ldm_model" in self.generation_information:
            model_name = self.generation_information["ldm_model"]
        ldm_label = Gtk.Label()
        ldm_label.set_text("Model")
        ldm_label.set_halign(Gtk.Align.START)  # Align label to the left
        box.pack_start(ldm_label, False, False, 0)
        
        if model_name in self.ldm_model_names:
            ind = self.ldm_model_names.index(model_name)
            self.enable_lora = True
        else:
            model_name = self.preferences["ldm_model"]
            ind = self.ldm_model_names.index(model_name)
            # ind = 0

        self.ldm_model_cb = create_combo_box_typeahead(
            self.ldm_model_names,
            ind)
        box.pack_start(self.ldm_model_cb, False, False, 0)


        #
        # Denoise entry
        #
        # box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        denoise_label = Gtk.Label()
        denoise_label.set_text("Denoising strength")
        denoise_label.set_halign(Gtk.Align.START)  # Align label to the left
        box.pack_start(denoise_label, False, False, 0)
    
        self.denoise_text = Gtk.Entry(text=self.preferences["denoising_strength"])
        box.pack_start(self.denoise_text, False, True, 0)

        # Slider for pen width
        pen_width_label = Gtk.Label()
        pen_width_label.set_text("Pen width")
        pen_width_label.set_halign(Gtk.Align.START)  # Align label to the left
        root_box.pack_start(pen_width_label, False, False, 0)
        # root_box.pack_start(box, False, False, 0)

        self.slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1, 256, 1)
        self.slider.set_value(self.pen_width)
        self.slider.connect("value-changed", self.on_slider_value_changed)
        root_box.pack_start(self.slider, False, False, 0)

        #
        # Positive prompt fields
        #
        # Label for the Positive Prompt text field
        positive_prompt_label = Gtk.Label()
        positive_prompt_label.set_text("Positive prompt override")
        positive_prompt_label.set_halign(Gtk.Align.START)  # Align label to the left
        root_box.pack_start(positive_prompt_label, False, False, 0)

        # Frame for the text view with a 1-pixel black border
        positive_frame = Gtk.Frame()
        positive_frame.set_shadow_type(Gtk.ShadowType.IN)  # Gives the impression of an inset (bordered) widget
        positive_frame.set_border_width(1)  # Adjust the positive_frame border width if necessary

        # Positive prompt multi-line text field
        self.positive_prompt_field = Gtk.TextView()
        self.positive_prompt_field.set_wrap_mode(Gtk.WrapMode.WORD)

        # Add the text view to the positive_frame, and then add the positive_frame to the vbox
        positive_frame.add(self.positive_prompt_field)
        root_box.pack_start(positive_frame, False, False, 0)

        # Button box
        button_box = Gtk.Box(spacing=6)
        root_box.pack_start(button_box, False, True, 0)

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

        # Translation support
        self.translation = [0, 0]
        self.scale_factor = 1.0

        # Set up matrix with scale and translate elements. This is to convert screen to 
        # cairo (logical coordinates)
        # Below internally calls self.update_transform_matrix(), so no need to call it
        # before.
        # result is stored in self.transform_matrix
        self._process_new_image()

        # Variables required to show highlighting the mouse drag
        self.drag_start_pos = None
        self.drag_end_pos = None
        self.in_motion = False

    def update_transform_matrix(self, cr):
        """
        Update the transformation matrix with translation and scaling.
        """
        # This is to convert mouse click pos to Cairo conversion
        self.transform_matrix_1 = cairo.Matrix()  # Create an identity matrix
        self.transform_matrix_1.translate(self.translation[0], self.translation[1])
        self.transform_matrix_1.scale(self.scale_factor, self.scale_factor)
        
        # this is used by Cairo to draw
        # This incorporates menu and other offsets
        self.transform_matrix = cr.get_matrix()
        self.transform_matrix.translate(self.translation[0], self.translation[1])
        self.transform_matrix.scale(self.scale_factor, self.scale_factor)

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
        inv_transform = cairo.Matrix(*self.transform_matrix_1)
        inv_transform.invert()  # screen coordinates to cairo logical coordinates
        x_logical, y_logical = inv_transform.transform_point(x, y)
        return x_logical, y_logical

    def on_clear_clicked(self, widget):
        self.mask_image=None
        self.lines_list.clear()
        self.drawing_area.queue_draw()

    def get_current_mask(self)->np.ndarray:
        """Returns cv mask image in rank 2 ndarray (h, w)
        Each pixel contains one of the following uint8 values:
            255: masked (to be inpainted)
              0: not to be inpainted
        """
        # Generate an image from drawing area
        # Create a new transparent surface
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 
                                     self.image_width,
                                     self.image_height)

        # Create a new canvas
        cr = cairo.Context(surface)

        # Fill background with black
        black_background_image = Image.new('RGBA', (self.image_width, self.image_height), "black")
        pixbuf = pil_image_to_pixbuf(black_background_image)   
        Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
        cr.paint()       

        self.redraw_mask(cr)

        # Convert the surface to a PIL Image
        width, height = surface.get_width(), surface.get_height()
        data = surface.get_data().tobytes()  # Convert memoryview to bytes

        # FIXME: Check to see if we need to convert ARGB to RGBA
        # Conver to a PIL Image
        img = Image.frombuffer("RGBA", (width, height), data, "raw", "BGRA", 0, 1)

        cv_img = np.array(img)
        cv_img = cv.cvtColor(cv_img, cv.COLOR_RGBA2GRAY)
        
        # Clip by the image size
        cv_img = cv_img[0:self.pil_image.size[1], 0:self.pil_image.size[0]]
        return cv_img

    def on_show_mask_toggled(self, checkbox):
        if checkbox.get_active():
            self.show_mask = True
        else:
            self.show_mask = False
        self.drawing_area.queue_draw()

    def on_show_bounding_box_toggled(self, checkbox):
        if checkbox.get_active():
            self.show_bounding_box = True
            self.detect_bounding_boxes()
        else:
            self.show_bounding_box = False
        self.drawing_area.queue_draw()

    def detect_bounding_boxes(self):
        # Create a new off-screen canvas to mirror drawing area
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                     self.image_width,
                                     self.image_height)
        cr = cairo.Context(surface)

        # Fill background with black
        black_background_image = Image.new('RGBA',
                                           (self.image_width, self.image_height),
                                           "black")
        pixbuf = pil_image_to_pixbuf(black_background_image)   
        Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
        cr.paint()       

        # Paint masked regions on canvas
        self.redraw_mask(cr)

        # Convert the surface to a PIL Image
        width, height = surface.get_width(), surface.get_height()
        data = surface.get_data().tobytes()  # Convert memoryview to bytes

        # FIXME: Check to see if we need to convert ARGB to RGBA
        # Convert CR to a PIL Image
        img = Image.frombuffer("RGBA", (width, height), data, "raw", "BGRA", 0, 1)

        # Detect bounding boxes
        cv_img = np.array(img)
        cv_img = cv.cvtColor(cv_img, cv.COLOR_RGBA2GRAY)
        boxes = get_bounding_boxes_from_grayscale_image(cv_img)
        logger.info(boxes)

        self.parse_face_data(boxes)
        logger.info(self.bounding_boxes)

    def on_slider_value_changed(self, slider):
        self.pen_width = slider.get_value()

    def on_eraser_checkbox_toggled(self, checkbox):
        if checkbox.get_active():
            self.is_eraser = True
        else:
            self.is_eraser = False

    def redraw_mask(self, cr) -> None:
        """
        Draws masked regions on CR.  Note that the painted result are not reflected
        on canvas until after self.drawing_area.queue_draw() is called
        which triggers repainting.
        Therefore, the caller needs to call self.drawing_area.queue_draw().

        lines_list: A list of lines
        lines: Lines is defined from a point from LMB down to LMB up
               It can have multiple segments.

        is_eraser, color, width are defined for each lines.

        Undo simply pops the lines_list

        Args:
            cr (cairo context): Cairo context to paint the masked region
        """
        cr_save = cr
        original_surface = cr_save.get_target()

        # Create additional context
        surface2 = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                      self.image_width,
                                      self.image_height)
        cr = cairo.Context(surface2)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)  # Or cairo.LINE_JOIN_BEVEL

        if self.lines_list:
            cr.set_line_width(self.pen_width)  # Initial line width

            for i, lines in enumerate(self.lines_list):
                # Set color based on whether it's an eraser
                if lines.is_eraser:
                    cr.set_operator(cairo.OPERATOR_CLEAR)
                else:
                    cr.set_operator(cairo.OPERATOR_OVER)  # Default drawing
                    cr.set_source_rgba(1, 1, 1, 1)  # White color for the line. CR takes [0, 1] instead of [0, 255]

                cr.set_line_width(lines.pen_width)
                cr.move_to(*lines.points[0])

                for p in (lines.points[1:]):
                    cr.line_to(*p)
                cr.stroke()

        # Merge to the original
        cr_save.set_source_surface(surface2, 0, 0)
        cr_save.paint()

    def on_draw(self, widget, cr):
        """
        Repaints the drawing canvas cr.

        This method is triggered when a method calls self.drawing_area.queue_draw().
        The client does not directly call this method.
        """
        # Compute the transform matrix
        self.update_transform_matrix(cr)  # update transform_matrix
        cr.set_matrix(self.transform_matrix)

        # Paint base image
        Gdk.cairo_set_source_pixbuf(cr, self.pixbuf_with_base_image, 0, 0)
        cr.paint()  # render the content of pixbuf in the source buffer on the canvas

        if self.show_mask:
            self.redraw_mask(cr)

        # Draw a rectangle over the image
        if self.show_bounding_box:
            for i, face_rect in enumerate(self.bounding_boxes):
                if i == self.selected_bounding_box_index:
                    cr.set_source_rgba(*SELECTED_COLOR)
                else:
                    cr.set_source_rgba(*UNSELECTED_COLOR)
                cr.rectangle(face_rect.left,
                            face_rect.top,
                            face_rect.right - face_rect.left,
                            face_rect.bottom - face_rect.top)
                cr.fill()

    def on_button_press(self, widget, event):
        """
        Handles the LMB press event.
        """
        logger.debug(f"DEBUG physical: click pos (screen coord): x:{event.x}, y:{event.y}")
        x, y = self.screen_to_cairo_coord(event.x, event.y)

        # Create new lines data
        ld = LinesData(pen_width=self.pen_width,
                       is_eraser=self.is_eraser,
                       points=[(x, y)])
        self.lines_list.append(ld)
        self.show_mask_checkbox.set_active(True)
        self.drawing_area.queue_draw()  # refresh image canvas

    def on_motion_notify(self, widget, event):
        """
        Handles the LMB drag event.
        """
        x, y = self.screen_to_cairo_coord(event.x, event.y)

        if event.state & Gdk.ModifierType.BUTTON1_MASK:
            current_lines_data_index = len(self.lines_list) - 1
            current_lines_data = self.lines_list[current_lines_data_index]
            current_lines_data.points.append((x, y))
            self.drawing_area.queue_draw()

    def on_button_release(self, widget, event):
        """
        Handles the LMB release event
        """
        x, y = self.screen_to_cairo_coord(event.x, event.y)
        current_lines_data_index = len(self.lines_list) - 1
        current_lines_data = self.lines_list[current_lines_data_index]
        current_lines_data.points.append((x, y))
        self.drawing_area.queue_draw()

    def _process_new_image(self):
        """
        Computes various image-related values.

        The only input needed is self.pil_image

        Sets 
          pixbuf for the input image
          image width & height
          canvas width & height (drawing area)
          adjust the canvas size based on above
        
        """
        # Creates pixbuf from the original input image to be used in on_draw
        self.pixbuf_with_base_image = pil_image_to_pixbuf(self.pil_image)
        self.image_width, self.image_height = self.pil_image.size

        # Compute canvas size (DrawingArea for Cairo)
        canvas_edge_length = 768  # FIXME
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
        self.set_default_size(new_width + 200, new_height)
        self.resize(new_width + 200, new_height)

        self._adjust_matrix_redraw_canvas()

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

    def on_apply_inpainting_clicked(self, widget):
        self.detect_bounding_boxes()
        boxes = self.bounding_boxes

        pil_image = self.pil_image
        cv_mask_image = self.get_current_mask()

        if boxes is not None:
            for face_rect in boxes:
                box = (face_rect.left,
                       face_rect.top,
                       face_rect.right - face_rect.left,
                       face_rect.bottom - face_rect.top)
                pil_image = self.process_box(pil_image, cv_mask_image, box)

        self.pil_image = pil_image
        if self.output_file_path:
            self.pil_image.save(self.output_file_path)

        self.show_bounding_box_checkbox.set_active(False)
        self.show_mask_checkbox.set_active(False)
        
        self.pixbuf_with_base_image = pil_image_to_pixbuf(self.pil_image)
        self.drawing_area.queue_draw()  # refresh image canvas

        if self.save_call_back:
            self.save_call_back(self.pil_image, self.generation_information_call_back())

    def on_clear_marks_clicked(self, widget):
        self.bounding_boxes.clear()
        self.lines_list.clear()
        self.drawing_area.queue_draw()  # refresh image canvas

    def on_save_activate(self, widget):
        """
        Save menu item is selected
        """
        # Show file chooser dialog
        chooser = Gtk.FileChooserDialog(title="Save File", 
                                        parent=self, 
                                        action=Gtk.FileChooserAction.SAVE)
        # Add cancel button            
        chooser.add_buttons(Gtk.STOCK_CANCEL, 
                            Gtk.ResponseType.CANCEL, 
                            Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        response = chooser.run()
        if response == Gtk.ResponseType.OK:
            filename = chooser.get_filename()
            self.pil_image.save(filename)
        chooser.destroy()

    def on_drag_data_received(self, widget, drag_context, x, y, data, info, time):
        """Drag and Drop handler.

        data: Contains info for the dragged file name
        """
        file_path = data.get_text().strip()
        if file_path.startswith('file://'):
            file_path = file_path[7:]
        logger.info("on_drag_data_received: {file_path}")
        self.pil_image = Image.open(file_path)
        self._process_new_image()

    def process_box(self, pil_image, cv_mask_image, face) -> Image:
        """

        x
        y
        w
        h
        score
        """
        input_image = pil_image

        # input_image.save("tmp_input_image.png")  # FIXME
        # cv.imwrite("tmp_mask_image.png", cv_mask_image)  # FIXME

        # Create a temporary directory using the tempfile module
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary directory created at {temp_dir}")
            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])
            print(f"{x}, {y}, {w}, {h}")

            # Expand by buffer
            buffer = 20
            x = max(0, x-buffer)
            y = max(0, y-buffer)
            w = min(w+buffer*2, pil_image.size[0] - x)
            h = min(h+buffer*2, pil_image.size[1] - y)
     
            right = x + w
            bottom = y + h
            crop_rectangle = (x, y, right, bottom)
            cropped_image = pil_image.crop(crop_rectangle)
            # cropped_mask_image = mask_image.crop(crop_rectangle)
            print(cropped_image.size)
     
            if w > h:  # landscape
                new_h = int(h * TARGET_EDGE_LEN / w)
                new_w = TARGET_EDGE_LEN
                padding_h = TARGET_EDGE_LEN - new_h
                padding_w = 0
                padding_x = int(padding_w/2)
                padding_y = int(padding_h/2)

            else:
                new_w = int(w * TARGET_EDGE_LEN / h)
                new_h = TARGET_EDGE_LEN
                padding_w = TARGET_EDGE_LEN - new_w
                padding_h = 0
                padding_x = int(padding_w/2)
                padding_y = int(padding_h/2)

            resized_image = cropped_image.resize((new_w, new_h), resample=PIL.Image.LANCZOS)

            # Pad image
            base_image = Image.new('RGBA', (TARGET_EDGE_LEN, TARGET_EDGE_LEN), "white")
            base_image.paste(resized_image, (padding_x, padding_y))

            # 2.3 Send to image to image
            updated_face_pil_image = self.face_image_to_image(
                input_image=base_image)
            updated_face_pil_image.save("tmpface.jpg")
            
            # Crop to remove padding
            updated_face_pil_image = updated_face_pil_image.crop(
                (padding_x,  # x
                padding_y,  # y
                padding_x + new_w,  # width
                padding_y + new_h))  # height
           
            # Resize to the original dimension
            updated_face_pil_image = \
                updated_face_pil_image.resize((w, h), resample=PIL.Image.LANCZOS)

            # 2.6 Paste the updated image in the original image.
            inpainted_image = input_image.copy()
            inpainted_image.paste(updated_face_pil_image, (x, y))
            inpainted_image_cv = np.array(inpainted_image)
            inpainted_image_cv = cv.cvtColor(np.array(inpainted_image_cv), cv.COLOR_RGB2BGR)
            cv.imwrite("tmp_inpainted_image_cv.png", inpainted_image_cv, )  # FIXME
            
            # Convert both base and face to CV2 BGR
            cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

            updated_face_cv_image = cv.cvtColor(
                np.array(updated_face_pil_image), 
                cv.COLOR_RGB2BGR)

            # Apply Gaussian blur to the mask
            blurred_mask = cv.GaussianBlur(cv_mask_image, (11, 11), 0)  # Change to (21, 21)

            # Normalize the blurred mask to ensure it's in the 0-255 range
            blurred_mask = np.clip(blurred_mask, 0, 255)

            # Instead of using bitwise operations, manually interpolate between the images
            # Convert mask to float
            blurred_mask_float = blurred_mask.astype(np.float32) / 255.0
            inverse_mask_float = 1.0 - blurred_mask_float

            # Convert images to float
            original_float = cv_image.astype(np.float32)
            inpainted_float = inpainted_image_cv.astype(np.float32)

            # Interpolate
            combined_float = (inpainted_float * blurred_mask_float[..., np.newaxis]) + (original_float * inverse_mask_float[..., np.newaxis])

            # Convert back to an 8-bit image
            combined_image_cv = np.clip(combined_float, 0, 255).astype(np.uint8)

            # Convert the combined image back to PIL format
            pil_image = Image.fromarray(cv.cvtColor(combined_image_cv, cv.COLOR_BGR2RGB))
            
            # Convert the result back to a PIL image
            # pil_image = Image.fromarray(cv.cvtColor(result_image, cv.COLOR_BGR2RGB))
            return pil_image
        
    def face_image_to_image(self, input_image=None, meta_prompt=None,
                            output_dir=SPOT_FIX_TMP_DIR):
        """
        Event handler for the Generation button click

        Args:
            meta_prompt (str): Gender string of the face detected by the gender ML model
        """
        logger.info("face_image_to_image")

        if self.generation_information_call_back is not None:
            generation_info = self.generation_information_call_back()
            if generation_info is None:  # The original image may not have generation info
                generation_info = dict()              
        else:
            generation_info = dict()

        # Prompt handling
        self.positive_prompt = text_view_get_text(self.positive_prompt_field)
        self.denoising_strength = self.denoise_text.get_text()
        
        if self.positive_prompt:
            positive_prompt = self.positive_prompt
            if self.preferences["enable_positive_prompt_expansion"]:
                positive_prompt += self.preferences["positive_prompt_expansion"]
        elif generation_info is not None and "positive_prompt" in generation_info: # Priority 2. Generation
           positive_prompt = generation_info["positive_prompt"]
        else:  # use blank
            positive_prompt = ""
            if self.preferences["enable_positive_prompt_expansion"]:
                positive_prompt += self.preferences["positive_prompt_expansion"]

        # Prepend meta_prompt
        if meta_prompt:
            positive_prompt = meta_prompt + ", " + positive_prompt

        # Negative prompt
        if self.negative_prompt:
            negative_prompt = self.negative_prompt
            if self.preferences["enable_negative_prompt_expansion"]:
                negative_prompt += self.preferences["negative_prompt_expansion"]
        elif generation_info is not None and "negative_prompt" in generation_info: # Priority 2. Generation
            negative_prompt = generation_info["negative_prompt"]
        else:  # use blank
            negative_prompt = ""
            if self.preferences["enable_negative_prompt_expansion"]:
                negative_prompt += self.preferences["negative_prompt_expansion"]

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
            
        vae_path = "None" if self.preferences["vae_model"] == "None" \
            else os.path.join(
                    self.preferences["vae_model_path"], 
                    self.preferences["vae_model"])
        
        model_name = self.ldm_model_cb.get_child().get_text() # Use the model on UI
        # model_name = self.preferences["ldm_model"]
        clip_skip = str(self.preferences["clip_skip"])
        lora_models, lora_weights = generate_lora_params(self.preferences)

        # # Check to see if we have generation info to override
        # if "ldm_model" in generation_info:
        #     model_name = generation_info["ldm_model"]
        #     logger.info(f"Overriding preference model name with generation model: {model_name}")
        if "clip_skip" in generation_info:
            clip_skip = str(generation_info["clip_skip"])
        
        model_path = os.path.join(self.preferences["ldm_model_path"],
                                  model_name)
        
        if "lora_models" in generation_info:
            if generation_info["lora_models"] and len(generation_info["lora_models"]) > 0:
                l = generation_info["lora_models"].split(",")

                # if image was generated in SD1.5, enable LoRA
                if self.enable_lora:
                    model_name = self.generation_information["ldm_model"]
                    l = [os.path.join(
                        self.preferences["lora_model_path"], e.strip()) for e in l if len(e.strip()) > 0]
                    l = ",".join(l)
                else:  # if SDXL, disable LoRA for now
                    l = ""
            else:
                l = ""
            lora_models = l
            lora_weights = generation_info["lora_weights"]

        args_list = ["--prompt", positive_prompt,
                     "--negative_prompt", negative_prompt,
                     "--H", str(TARGET_EDGE_LEN),
                     "--W", str(TARGET_EDGE_LEN),
                     "--clip_skip", clip_skip,
                     "--seed", str(self.preferences["seed"]),
                     "--n_samples", str(1),
                     "--n_iter",str(1),
                     "--ckpt", model_path,
                     "--embedding_path", self.preferences["embedding_path"],
                     "--vae_ckpt", vae_path,
                     "--lora_models", lora_models,
                     "--lora_weights", lora_weights,                                            
                     "--outdir", output_dir]
        if self.preferences["safety_check"]:
            args_list.append("--safety_check")
        input_image_path = os.path.join(get_tmp_dir(), "input_image.png")
        input_image.save(input_image_path)

        args_list += [
            "--init-img", input_image_path,
            "--strength", self.denoising_strength
        ]
        generate_func=img2img_parse_options_and_generate
        
        # Start the image generation thread
        thread = threading.Thread(
            target=generate_func,
            kwargs={'args': args_list,
                    'ui_thread_instance': None})  # FIXME
        thread.start()

        thread.join()  # Wait until img2img is done.

        # Get the name of the output image
        files = os.listdir(output_dir)
        assert len(files) == 1
        file_name = os.path.join(output_dir, files[0])
        return Image.open(file_name)
    
    def update_image(self, img_data: Image, generation_parameters:str=None) -> None:
        pass

    def parse_face_data(self, faces) -> None:
        self.bounding_boxes.clear()

        for i, face in enumerate(faces):
            face_rect = BoxRect(
                face[0], 
                face[1],
                face[0]+face[2],
                face[1]+face[3]
            )
            self.bounding_boxes.append(face_rect)

    def on_image_button_release(self, widget, event):
        x, y = (event.x, event.y)
        if abs(x - self.prev_x) > 5 and  abs(y - self.prev_y) > 5:
            left = min(x, self.prev_x)
            right = max(x, self.prev_x)
            top = min(y, self.prev_y)
            bottom = max(y, self.prev_y)
            face_rect = BoxRect(left, top, right, bottom)
            self.bounding_boxes.append(face_rect)

            self.image.queue_draw()  # invalidate

    def _adjust_matrix_redraw_canvas(self):
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
        self.drawing_area.queue_draw()

    def on_vscroll(self, adjustment):
        self.translation[1] = -adjustment.get_value()
        self.drawing_area.queue_draw()


def main():
    preferences = {
        "safety_check": False,
        "watermark": False,
        "image_width": "512",
        "image_height": "768",
        "clip_skip": "2",
        "denoising_strength": "0.5",
        "batch_size": "3",
        "number_of_batches": "1",
        "ldm_model_path": "/media/pup/ssd2/recoverable_data/sd_models/Stable-diffusion",
        "ldm_model": "analogMadness_v70.safetensors",
        "ldm_inpaint_model": "majicmixRealistic_v7-inpainting.safetensors",
        "lora_model_1": "None",
        "lora_model_2": "None",
        "lora_model_3": "None",
        "lora_model_4": "None",
        "lora_model_5": "None",
        "lora_weight_1": 1.0,
        "lora_weight_2": 1.0,
        "lora_weight_3": 1.0,
        "lora_weight_4": 1.0,
        "lora_weight_5": 1.0,
        "lora_model_path": "/media/pup/ssd2/recoverable_data/sd_models/Lora",        
        "vae_model_path": "/media/pup/ssd2/recoverable_data/sd_models/VAE",
        "vae_model": "vae-ft-mse-840000-ema-pruned.ckpt",
        "embedding_path": "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
        "positive_prompt_expansion": ", highly detailed, photorealistic, 4k, 8k, uhd, raw photo, best quality, masterpiece",
        "negative_prompt_expansion": ", drawing, 3d, worst quality, low quality, disfigured, mutated arms, mutated legs, extra legs, extra fingers, badhands",
        "enable_positive_prompt_expansion": True, # False,
        "enable_negative_prompt_expansion": True, # False,
        "seed": "0"
    }

    pil_image = Image.open("../cremage_resources/512x512_human_couple.png")   # FIXME
    app = SpotInpainter(
        pil_image=pil_image, 
        output_file_path="tmp_face_fix.png",
        positive_prompt=None,
        negative_prompt=None,
        preferences=preferences)
    app.connect('destroy', Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()
