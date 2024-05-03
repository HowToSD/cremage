"""
Copyright (C) 2024, Hideyuki Inada. All rights reserved.

Scales an image
"""
import os
import sys
import tempfile
import logging
import argparse

from io import BytesIO

import numpy as np
from PIL import Image
import cv2
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")

sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
from tool_base import ToolBase
from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.misc_utils import get_tmp_file
from cremage.utils.gtk_utils import show_alert_dialog, set_pil_image_to_gtk_image
from gfpgan_wrapper import gfp_wrapper


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class ImageScaler(ToolBase):  # Subclass Window object
    def __init__(
            self,
            title:str = "Image cropper",
            **kwargs):
        super().__init__(title="Scaling Tool", **kwargs)

    def set_up_ui(self):
        if self.pil_image and self.output_file_path is None:
            raise ValueError("scaled file path is not specified when input_image is not None")

        self.set_default_size(800, 600)  # width, height
        self.set_border_width(10)

        # Create a vertical Gtk.Box
        root_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(root_box)

        # Create a MenuBar
        ## Accelerator
        accel_group = Gtk.AccelGroup()
        self.add_accel_group(accel_group)
        
        self.menubar = self.create_menu()
        root_box.pack_start(self.menubar, False, False, 0)

        # Horizontal Gtk.Box to contain the scrolled window and control elements
        container_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root_box.pack_start(container_box, True, True, 0)  # Add container_box to root_box under the menu

        # Create a ScrolledWindow
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_hexpand(True)
        scrolled_window.set_vexpand(True)
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        # Create an Image widget
        if self.pil_image is None:
            pil_image = Image.new('RGBA', (512, 768), "gray")
        else:
            pil_image = self.pil_image
        pixbuf = pil_image_to_pixbuf(pil_image)
        self.image_view = Gtk.Image.new_from_pixbuf(pixbuf)

        # Setup drag and drop for the image area
        self.image_view.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.image_view.drag_dest_add_text_targets()
        self.image_view.connect('drag-data-received', self.on_drag_data_received)

        # Add the Image to the ScrolledWindow
        scrolled_window.add(self.image_view)

        # Add the ScrolledWindow to the root_box
        container_box.pack_start(scrolled_window,
                        True,  # expand this field as the parent container expand
                        True,  # take up the initially assigned space
                        0)

        # Vertical Box for controls next to the ScrolledWindow
        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        container_box.pack_start(controls_box, False, True, 0)

        # Scale ratio controls
        scale_ratio_box = Gtk.Box(spacing=6)
        scale_ratio_label = Gtk.Label(label="Scale ratio:")
        self.scale_ratio_entry = Gtk.Entry()
        self.scale_ratio_entry.set_text("2")
        scale_ratio_box.pack_start(scale_ratio_label, False, False, 0)
        scale_ratio_box.pack_start(self.scale_ratio_entry, True, True, 0)
        controls_box.pack_start(scale_ratio_box, False, True, 0)

        # Scaling method dropdown
        scaling_method_label = Gtk.Label(label="Scaling method:")
        controls_box.pack_start(scaling_method_label, False, True, 0)
        self.scaling_method_combo = Gtk.ComboBoxText()
        self.scaling_method_combo.append_text("Lanczos")
        self.scaling_method_combo.append_text("Real ESR")
        self.scaling_method_combo.set_active(0)
        controls_box.pack_start(self.scaling_method_combo, False, True, 0)

        # Scale button
        scale_button = Gtk.Button(label="Scale")
        controls_box.pack_start(scale_button, False, True, 0)
        scale_button.connect("clicked", self.on_scale_clicked)

    def on_open_clicked(self, widget):
        super().on_open_clicked(widget)
        if self.pil_image:
            set_pil_image_to_gtk_image(self.pil_image, self.image_view)


    def on_update_clicked(self, widget):
        """
        Update caller menu item is selected
        """
        if self.output_pil_image != None:
            super().on_update_clicked(widget)


    def on_save_clicked(self, widget):
        """
        Save menu item is selected
        """
        if self.output_pil_image:
            super().on_save_clicked(widget)

    def on_scale_clicked(self, widget):
        if self.pil_image is None:
            return
        
        scale_ratio = self.scale_ratio_entry.get_text()

        try:
            scale_ratio = float(scale_ratio)
        except:
            logging.warn("Specify the scaling_ratio")
            return

        scaling_method = self.scaling_method_combo.get_active_text()
        logger.info(scaling_method)

        # Compute target image size
        width, height = self.pil_image.size
        target_width = int(width * scale_ratio)
        target_height = int(height * scale_ratio)

        if scaling_method == "Lanczos":
            cv_image = np.asarray(self.pil_image, dtype=np.uint8)[:,:,::-1]  # Convert to np array and RGB to BGR
            scaled_cv_image = cv2.resize(cv_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            self.output_pil_image = Image.fromarray(scaled_cv_image[:,:,::-1])
            if self.output_file_path:
                self.output_pil_image.save(self.output_file_path)
        elif scaling_method == "Real ESR":
            if scale_ratio not in (2, 4):
                show_alert_dialog("For ESR-GAN with GFPGAN, scaling ratio needs to be 2 or 4.")
                return
            # Create a temporary path
            if self.output_file_path is None:
                with tempfile.TemporaryFile() as temp_file:
                    print('Temporary file path:', temp_file.name)
                    self.scale_with_gfpgan(scale_ratio, temp_file.name)
                    self.output_pil_image = Image.open(temp_file.name)
                assert(os.path.exists(temp_file.name) is False)
            else:
                self.scale_with_gfpgan(scale_ratio, self.output_file_path)                
                self.output_pil_image = Image.open(self.output_file_path)
        pixbuf = pil_image_to_pixbuf(self.output_pil_image)
        self.image_view.set_from_pixbuf(pixbuf)
        if self.save_call_back:
            super().on_update_clicked(widget)  # Send output to the caller
        return

    def scale_with_gfpgan(self, scale_ratio, output_file_path=None) -> None:
        if output_file_path is None:
            ValueError("output_path is not specified.")

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-i',
            '--input',
            type=str,
            help='Input image')
        parser.add_argument(
            '-v', '--version', type=str, default='1.4', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
        parser.add_argument(
            '-s', '--upscale', type=int, default=scale_ratio, help='The final upsampling scale of the image. Default: 2')
        parser.add_argument(
            '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
        parser.add_argument(
            '--bg_tile',
            type=int,
            default=400,
            help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
        parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
        parser.add_argument('-o', '--output', type=str, default="tmpgfp.png", help='Output image path')

        # Use a temporary file to save pil image
        tmp_name = get_tmp_file(".png")
        self.pil_image.save(tmp_name)
        logger.info(f"Image temporarily saved to a temporary path: {tmp_name}")
        args = parser.parse_args(["--input", tmp_name,
                                "--output", output_file_path])  
        gfp_wrapper(args)

        if os.path.exists(tmp_name) is False:
            ValueError("Failed in creating a temp file.")
        os.remove(tmp_name)
        logger.info(f"Deleted tmp file {tmp_name}")
        if os.path.exists(tmp_name):
            ValueError("Failed in removing a temp file.")

    # drag & drop handlers
    def on_drag_data_received(self, widget, drag_context, x, y, data, info, time):
        """Drag and Drop handler.

        data: Contains info for the dragged file name
        """
        file_path = data.get_text().strip()
        if file_path.startswith('file://'):
            file_path = file_path[7:]
            self.pil_image = Image.open(file_path)
            set_pil_image_to_gtk_image(self.pil_image, self.image_view)


def main():
    app = ImageScaler(output_file_path="tmp_scaled.png")
    app.connect('destroy', Gtk.main_quit)
    app.show_all()
    Gtk.main()

if __name__ == '__main__':
    main()


