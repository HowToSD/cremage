# Video generator
import os
import sys
import logging
import time
import argparse
import json
from io import BytesIO
import tempfile
import threading
import queue
import shutil
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from functools import partial

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, GObject
import cairo

from PIL import Image
import numpy as np
import cv2 as cv
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
SDXL_MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules", "sdxl")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_ROOT, "film", "film_torch.pt")
BLANK_INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "blank_image.png")
VIDEO_INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "resources", "images", "video_input_image.png")

sys.path = [SDXL_MODULE_ROOT, MODULE_ROOT, TOOLS_ROOT] + sys.path
from sdxl_pipeline.svd_video_generator import sample
from cremage.utils.image_utils import resize_crop_pil_image
from frame_interpolation_pytorch.inference_multiple_frames import inference_multiple_frames
from cremage.utils.gtk_utils import resized_gtk_image_from_pil_image, pil_image_to_gtk_image
from cremage.utils.gtk_utils import set_pil_image_to_gtk_image, resize_with_padding
from cremage.ui.drag_and_drop_handlers import video_generator_input_image_drag_data_received
from cremage.utils.misc_utils import get_tmp_dir
from cremage.const.const import VIDEO_GENERATOR_THUMBNAIL_IMAGE_HEIGHT
from cremage.const.const import VIDEO_GENERATOR_THUMBNAIL_IMAGE_WIDTH
from cremage.ui.video_generator_input_image_view_click_handler import video_generator_input_image_view_click_handler
from cremage.status_queues.video_generation_status_queue import video_generation_status_queue
from cremage.status_queues.denoising_status_queue import denoising_status_queue

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


INPUT_FILE_NAME = "/home/pup/.cremage/outputs/1024x576_cat.png"
TARGET_FILE_NAME="cremage_svd_output_cat_pytorch.mp4"

class VideoGenerator(Gtk.Window):
    def __init__(self,
                 pil_image=None,
                 output_file_path=None,
                 save_call_back=None,
                 positive_prompt=None,
                 negative_prompt=None,
                 generation_information_call_back=None,
                 preferences=None,
                 checkpoint_path=None,
                 procedural=False,
                 status_queue=None,
                 app=None):
        """
        Args:
            pil_image: An image containing one or more faces
            pil_face_image: Face ID image
            procedural (bool): True if called from img2img. False if invoked on the UI
        """
        super().__init__(title="Video generator")

        # Apply CSS to remove padding and margin from all widgets
        css = b"""
        .custom-statusbar {
            background-color: #D3D3D3;
        }
        """
        style_provider = Gtk.CssProvider()
        style_provider.load_from_data(css)

        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            style_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.pil_image = pil_image
        self.image_cropper = None 
        self.output_file_path = output_file_path
        self.save_call_back = save_call_back
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.preferences = dict() if preferences is None else preferences
        self.generation_information_call_back = generation_information_call_back
        self.output_dir = None  # This is used for refreshing the image list, but not used for Face Fixer
        self.checkpoint_path = checkpoint_path
        self.app = app
        self.timeout_id = None  # For video frame refresh
        self.status_queue_id = None  # Video generation status queue

        # Prompt used if invoked from img2img
        self.procedural = procedural  # True if img2img. False if UI
        self.positive_prompt_procedural = positive_prompt
        self.negative_prompt_procedural = negative_prompt
        self.parent_status_queue = status_queue  # This is to update the parent and not currently used

        # Create an Image widget
        if pil_image is None:
            pil_image = Image.new('RGBA', (1024, 576), "gray")
        self.pil_image = pil_image
        self.pil_image_original = self.pil_image  # Make a copy to restore
        self.pil_image_resized =  resize_crop_pil_image(self.pil_image_original)
        # Put placeholder values - These are re-computed at the end of this method
        self.image_width, self.image_height = self.pil_image.size
        canvas_edge_length = 768
        self.canvas_width = canvas_edge_length
        self.canvas_height = canvas_edge_length
        self.set_default_size(self.canvas_width + 200, self.canvas_height)  # This will be re-computed too.
        
        self.generation_information = dict()
        if self.generation_information_call_back is not None:
            d = self.generation_information_call_back()
            if d:  # The original image may not have generation info
                self.generation_information = d

        self.set_border_width(0)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Create a drawing area for playing the generated video
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(1024, 576)
        vbox.pack_start(self.drawing_area, False, True, 0)
        self.drawing_area.connect("draw", self.on_draw)

        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)

        # Set margins for the grid
        grid.set_margin_start(10)  # Margin on the left side
        grid.set_margin_end(10)    # Margin on the right side
        grid.set_margin_top(10)    # Margin on the top
        grid.set_margin_bottom(10) # Margin on the bottom
        row = 6

        vbox.pack_start(grid, True, True, 0)

        # Image selector
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        vbox.pack_start(hbox, True, True, 0)

        filler_left = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hbox.pack_start(filler_left, True, True, 0)

        self.input_image_wrapper = Gtk.EventBox()
        self.video_generator_input_image_thumbnail = resized_gtk_image_from_pil_image(
            self.pil_image_original,
            target_width=VIDEO_GENERATOR_THUMBNAIL_IMAGE_WIDTH,
            target_height=VIDEO_GENERATOR_THUMBNAIL_IMAGE_HEIGHT)

        # Input image drag and drop support
        self.video_generator_input_image_thumbnail.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.video_generator_input_image_thumbnail.drag_dest_add_text_targets()
        self.video_generator_input_image_thumbnail.connect('drag-data-received',
                                        lambda
                                            widget,
                                            drag_context,
                                            x,
                                            y,
                                            data,
                                            info,
                                            time,
                                            app=self:
                                                video_generator_input_image_drag_data_received(
                                                    app,
                                                    widget,
                                                    drag_context,
                                                    x,
                                                    y,
                                                    data,
                                                    info,
                                                    time,
                                            ))    
        self.input_image_wrapper.add(self.video_generator_input_image_thumbnail)
        self.input_image_wrapper.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.input_image_wrapper.connect("button-press-event", lambda widget, event, app=self: video_generator_input_image_view_click_handler(app, widget, event))
        hbox.pack_start(self.input_image_wrapper, False, True, 0)

        # Cropped image
        self.cropped_image_view = resized_gtk_image_from_pil_image(
            self.pil_image_resized,
            target_width=VIDEO_GENERATOR_THUMBNAIL_IMAGE_WIDTH,
            target_height=VIDEO_GENERATOR_THUMBNAIL_IMAGE_HEIGHT)
        hbox.pack_start(self.cropped_image_view, False, True, 0)

        filler_right = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hbox.pack_start(filler_right, True, True, 0)

        # Buttons
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        vbox.pack_start(hbox, True, True, 0)

        # Filler on the left
        filler_left = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hbox.pack_start(filler_left, True, True, 0)

        # Button to crop
        self.crop_button = Gtk.Button(label="Adjust Crop")
        self.crop_button.connect('button-press-event', self.crop_handler)
        self.crop_button.set_sensitive(True)
        hbox.pack_start(self.crop_button, False, True, 0)

        # Generate button
        self.generate_button = Gtk.Button(label="Generate")
        self.generate_button.connect('button-press-event', self.generate_handler)
        hbox.pack_start(self.generate_button, False, True, 0)
    
        # Button to play the video
        self.play_button = Gtk.Button(label="Play")
        self.play_button.connect('button-press-event', self.play_handler)
        self.play_button.set_sensitive(False)  # Gray out the button
        hbox.pack_start(self.play_button, False, True, 0)

        # Button to play the video
        self.loop_checkbox = Gtk.CheckButton(label="Include Reverse Loop")
        hbox.pack_start(self.loop_checkbox, False, True, 0)

        # Filler on the right
        filler_right = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        hbox.pack_start(filler_right, True, True, 0)

        # Create a status bar
        self.statusbar = Gtk.Statusbar()
        self.status_bar_context_id = self.statusbar.get_context_id("status")
        self.statusbar.set_margin_top(0)
        self.statusbar.set_margin_bottom(0)
        self.statusbar.set_margin_start(0)
        self.statusbar.set_margin_end(0)
        self.statusbar.set_spacing(0)
        vbox.pack_end(self.statusbar, False, True, 0)

        # Add custom style class to the status bar
        self.statusbar.get_style_context().add_class("custom-statusbar")

        # Show all widgets
        self.show_all()

    def on_draw(self, widget, cr):
        if hasattr(self, 'frame'):
            height, width, channels = self.frame.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                self.frame.tobytes(),
                GdkPixbuf.Colorspace.RGB,
                False,
                8,
                width,
                height,
                width * channels
            )
            image_surface = Gdk.cairo_surface_create_from_pixbuf(pixbuf, 0)

            cr.set_source_surface(image_surface, 0, 0)
            cr.paint()

    def update_frame(self):
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    self.frame = frame
                    self.drawing_area.queue_draw()
                    return True
                else:
                    if self.timeout_id:
                        GLib.source_remove(self.timeout_id)
                        self.timeout_id = None
                    self.cap.release()
                    self.cap = None
                    return False

    def crop_update(self, cropped_pil_image):
        w, h = cropped_pil_image.size
        if w == 1024 and h == 576:
            self.pil_image_resized = cropped_pil_image
        else:
            self.pil_image_resized = cropped_pil_image.resize((1024, 576), Image.LANCZOS)

        resized_pil_image = resize_with_padding(
                self.pil_image_resized,
                target_width=VIDEO_GENERATOR_THUMBNAIL_IMAGE_WIDTH,
                target_height=VIDEO_GENERATOR_THUMBNAIL_IMAGE_HEIGHT)
        set_pil_image_to_gtk_image(resized_pil_image, self.cropped_image_view)

    def crop_handler(self, widget, event):
        print("Crop handler called")
        from image_cropper import ImageCropper

        # crop_callback = partial(self.crop_update, self)

        # Tool window
        if self.image_cropper is None:
            self.image_cropper = ImageCropper(
                pil_image=self.pil_image_original,
                output_file_path=None,
                save_call_back=None, # self.save_call_back,
                generation_information_call_back=None, # self.generation_information_call_back,
                preferences=None,
                aspect_ratio_w=1024,
                aspect_ratio_h = 576,
                callback=self.crop_update)
            self.image_cropper.connect("delete-event", self.on_image_cropper_delete)
        self.image_cropper.show_all()

    def on_image_cropper_delete(self, widget, event):
        logger.debug("Image cropper is destroyed")
        self.image_cropper = None 

    def play_handler(self, widget, event):
        # OpenCV Video Capture
        self.cap = cv.VideoCapture(self.output_file_path)

        # Timer to update frames
        self.timeout_id = GLib.timeout_add(25, self.update_frame)

    def generate_handler(self, widget, event):
        self.generate_button.set_sensitive(False)  # Gray out the button
        self.play_button.set_sensitive(False)  # Gray out the button
        # Start the image generation thread
        thread = threading.Thread(
            target=generate_func,
            kwargs={'app': self,
                    'ui_thread_instance': self})

        thread.start()
        self.status_queue_id = GLib.timeout_add(100, self.check_status_queue)
        GLib.timeout_add(100, self.check_thread, thread)


    def check_thread(self, thread):
        if thread.is_alive():
            return True  # Continue checking
        else:
            if self.status_queue_id:
                GLib.source_remove(self.status_queue_id)
                self.status_queue_id = None

            self.generate_button.set_sensitive(True)  # Re-enable the button
            self.play_button.set_sensitive(True)  # Re-enable
            self.play_handler(self.play_button, None)
            return False  # Stop checking

    def check_status_queue(self):
        global video_generation_status_queue, denoising_status_queue
        try:
            # Process all messages in video_generation_status_queue
            while not video_generation_status_queue.empty():
                status_message = video_generation_status_queue.get_nowait()
                if status_message:
                    GLib.idle_add(self.update_statusbar, status_message)
            
            # Process all messages in denoising_status_queue
            while not denoising_status_queue.empty():
                status_message = denoising_status_queue.get_nowait()
                if status_message:
                    GLib.idle_add(self.update_statusbar, status_message)

        except queue.Empty:
            pass

        return True  # This is needed for this to be called multiple times
    
    def update_statusbar(self, status_message):
        self.statusbar.push(self.status_bar_context_id, status_message)
        return False


def generate_func(app=None,
                  ui_thread_instance=None):

    start_time = time.perf_counter()
    print(f"Processing {INPUT_FILE_NAME}")

    # Set up paths
    tmp_dir = get_tmp_dir()
    tmp_input_file_path = os.path.join(tmp_dir, "video_generator_tmp_input.png")
    app.pil_image_resized.save(tmp_input_file_path)
    frame_output_path =os.path.expanduser("~/.cremage/tmp/svd/frames")

    # Remove old tmp frame files
    logger.info(f"Cleaning up old temporary frame files in {frame_output_path}")
    old_files = os.listdir(frame_output_path)
    for f in old_files:
        p = os.path.join(frame_output_path, f)
        if os.path.isfile(p):
            os.remove(p)

    print(f"Generating video frames")
    sample(
        input_path=tmp_input_file_path,
        checkpoint_path=app.checkpoint_path,
        output_path=frame_output_path,
        apply_watermark=app.preferences["watermark"],
        apply_filter=app.preferences["safety_check"],
        loop_video=app.loop_checkbox.get_active()
    )
    sample_time = time.perf_counter() - start_time
    print(f"Sample time: {sample_time}")

    video_generation_status_queue.put("Interpolating between frames")
    inference_multiple_frames(
        model_path=MODEL_PATH,
        input_dir_path=frame_output_path,
        input_file_prefix="frame",
        save_path=app.output_file_path,
        gpu=True,
        interpolation_frames=3,
        output_fps=25,
        half=False)

    end_time = time.perf_counter()
    print(f"Time elapsed: {end_time - start_time}")
    video_generation_status_queue.put(f"Completed. Time elapsed: {end_time - start_time:0.1f} seconds")


def main():
    pil_image = Image.open(BLANK_INPUT_IMAGE_PATH)
    app = VideoGenerator(
        pil_image=pil_image,
        output_file_path=os.path.join(get_tmp_dir(), "tmp_video.mp4"),
        positive_prompt=None,
        negative_prompt=None,
        preferences={"watermark":False, "safety_check":True},
        checkpoint_path="/media/pup/SAN4/recoverable_data/sd/svd_checkpoints/svd_xt_1_1.safetensors")
    app.connect('destroy', Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == "__main__":

    import os
    import sys
    import logging
    import platform

    os_name = platform.system()
    if os_name == 'Darwin':  # macOS
        gpu_device = "mps"
    else:
        gpu_device = "cuda"
    os.environ["GPU_DEVICE"] = gpu_device

    main()
