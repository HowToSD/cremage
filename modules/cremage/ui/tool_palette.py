"""
Tool palette to display multiple buttons to invoke tools.
This is displayed on a tab on the main UI.
"""
import os
import sys
import logging
from typing import Dict, Any

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [TOOLS_ROOT, MODULE_ROOT] + sys.path
from image_cropper import ImageCropper
from image_scaler import ImageScaler
from spot_inpainter import SpotInpainter
from face_fixer import FaceFixer
from graffiti_editor import GraffitiEditor
from image_segmenter import ImageSegmenter
from prompt_builder import PromptBuilder


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

PROMPT_BUILDER_INPUT_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "prompt_builder")  # FIXME. Read from config
ITEMS_PER_ROW = 8

class ToolPaletteArea():
    
    def __init__(self,
                 parent_box: Gtk.Box,
                 get_current_image_call_back=None,
                 get_tool_processed_file_path_call_back=None,
                 save_call_back=None,
                 generation_information_call_back=None,
                 preferences=None,
                 positive_prompt=None,
                 negative_prompt=None,
                 get_current_face_image_call_back=None,
                 face_model_full_path=None,
                 app=None):
        """
        Args:        
            get_tool_processed_file_path_call_back: Call this method to get the output path.
        """
        self.app = app
        self.get_current_image_call_back = get_current_image_call_back
        self.get_tool_processed_file_path_call_back = get_tool_processed_file_path_call_back
        self.save_call_back = save_call_back
        self.generation_information_call_back = generation_information_call_back
        self.preferences = preferences
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.get_current_face_image_call_back = get_current_face_image_call_back
        self.face_model_full_path = face_model_full_path
        
        tool_handlers = [self.on_crop_clicked,
                            self.on_scale_clicked,
                            self.on_spot_inpaint_clicked,
                            self.on_face_fix_clicked,
                            self.on_graffiti_editor_clicked,
                            self.on_image_segmenter_clicked,
                            self.on_prompt_builder_clicked]
        tool_names = [
            "Crop",
            "Scale",
            "Spot inpainting",
            "Face fix",
            "Graffiti editor",
            "Segmentation inpainting",
            "Visual prompt builder"]

        grid = Gtk.Grid()
        # Set margins for the grid
        grid.set_margin_start(10)  # Margin on the left side
        grid.set_margin_end(10)    # Margin on the right side
        grid.set_margin_top(10)    # Margin on the top
        grid.set_margin_bottom(10) # Margin on the bottom

        for i, tool in enumerate(tool_names):  # For 10 buttons
            button = Gtk.Button(label=f"{tool}")
            # Set button size
            button.set_size_request(24, 24)
            # Place the button in the grid
            grid.attach(button, i % ITEMS_PER_ROW, i // ITEMS_PER_ROW, 1, 1)
            button.connect('button-press-event', tool_handlers[i])

        self.image_cropper = None
        self.image_scaler = None
        self.spot_inpainter = None
        self.face_fixer = None
        self.graffiti_editor = None
        self.image_segmenter = None
        self.prompt_builder = None

        parent_box.pack_start(grid, True, True, 0)


        self.prompt_build_input_directory = PROMPT_BUILDER_INPUT_DIRECTORY

    def on_crop_clicked(self, widget, event):
        """
        Event handler for the crop button click
        """
        logger.debug("Crop clicked")

        # Tool window
        if self.image_cropper is None:
            if self.get_current_image_call_back is not None and \
                self.get_tool_processed_file_path_call_back is not None:
                self.image_cropper = ImageCropper(
                    pil_image=self.get_current_image_call_back(),
                    output_file_path=self.get_tool_processed_file_path_call_back(),
                    save_call_back=self.save_call_back,
                    generation_information_call_back=self.generation_information_call_back,
                    preferences=self.preferences)
            else:
                self.image_cropper = ImageCropper()
            self.image_cropper.connect("delete-event", self.on_image_cropper_delete)
        self.image_cropper.show_all()

    def on_scale_clicked(self, widget, event):
        """
        Event handler for the scale button click
        """
        logger.debug("Scale clicked")

        if self.image_scaler is None:
            if self.get_current_image_call_back is not None and \
                self.get_tool_processed_file_path_call_back is not None:
                self.image_scaler = ImageScaler(
                    pil_image=self.get_current_image_call_back(),
                    output_file_path=self.get_tool_processed_file_path_call_back(),
                    save_call_back=self.save_call_back,
                    generation_information_call_back=self.generation_information_call_back,
                    preferences=self.preferences)
            else:
                self.image_scaler = ImageScaler()
            self.image_scaler.connect("delete-event", self.on_image_scaler_delete)
        self.image_scaler.show_all()        

    def on_spot_inpaint_clicked(self, widget, event):
        """
        Event handler for the face detect button click
        """
        logger.debug("Spot inpainter clicked")

        if self.spot_inpainter is None:
            if self.get_current_image_call_back is not None and \
                self.get_tool_processed_file_path_call_back is not None:
                self.spot_inpainter = SpotInpainter(
                    pil_image=self.get_current_image_call_back(),
                    output_file_path=self.get_tool_processed_file_path_call_back(),
                    save_call_back=self.save_call_back,
                    positive_prompt=self.positive_prompt,
                    negative_prompt=self.negative_prompt,
                    preferences=self.preferences,
                    generation_information_call_back=self.generation_information_call_back)
            else:
                self.spot_inpainter = SpotInpainter()
            self.spot_inpainter.connect("delete-event", self.on_spot_inpainter_delete)
        self.spot_inpainter.show_all()        

    def on_face_fix_clicked(self, widget, event):
        """
        Event handler for the face fix button click
        """
        logger.debug("Face fix clicked")

        if self.face_fixer is None:
            if self.get_current_image_call_back is not None and \
                self.get_tool_processed_file_path_call_back is not None:
                self.face_fixer = FaceFixer(
                    pil_image=self.get_current_image_call_back(),
                    output_file_path=self.get_tool_processed_file_path_call_back(),
                    save_call_back=self.save_call_back,
                    positive_prompt=self.positive_prompt,
                    negative_prompt=self.negative_prompt,
                    preferences=self.preferences,
                    generation_information_call_back=self.generation_information_call_back,
                    pil_face_image=self.get_current_face_image_call_back(),
                    face_model_full_path=self.face_model_full_path)
        self.face_fixer.connect("delete-event", self.on_face_fixer_delete)
        self.face_fixer.show_all()   

    def on_graffiti_editor_clicked(self, widget, event):
        """
        Event handler for the graffiti editor button click
        """
        if self.graffiti_editor is None:
            if self.get_current_image_call_back is not None and \
                self.get_tool_processed_file_path_call_back is not None:
                self.graffiti_editor = GraffitiEditor(
                    pil_image=self.get_current_image_call_back(),
                    output_file_path=self.get_tool_processed_file_path_call_back(),
                    save_call_back=self.save_call_back,
                    preferences=self.preferences,
                    generation_information_call_back=self.generation_information_call_back)
            else:
                self.graffiti_editor = GraffitiEditor()
            self.graffiti_editor.connect("delete-event", self.on_graffiti_editor_delete)
        self.graffiti_editor.show_all()   


    def on_image_segmenter_clicked(self, widget, event):
        """
        Event handler for the face detect button click
        """
        logger.debug("Spot inpainter clicked")

        if self.image_segmenter is None:
            if self.get_current_image_call_back is not None and \
                self.get_tool_processed_file_path_call_back is not None:
                self.image_segmenter = ImageSegmenter(
                    pil_image=self.get_current_image_call_back(),
                    output_file_path=self.get_tool_processed_file_path_call_back(),
                    save_call_back=self.save_call_back,
                    preferences=self.preferences,
                    generation_information_call_back=self.generation_information_call_back)
            else:
                self.image_segmenter = ImageSegmenter()
            self.image_segmenter.connect("delete-event", self.on_image_segmenter_delete)
        self.image_segmenter.show_all()        

    def on_prompt_builder_clicked(self, widget, event):
        """
        Event handler for Visual Prompt Builder
        """
        logger.debug("Prompt builder clicked")
        self.prompt_builder = PromptBuilder(
            app=self.app,
            input_directory=self.prompt_build_input_directory
        )
        self.prompt_builder.window.connect("delete-event", self.on_prompt_builder_delete)
        self.prompt_builder.window.show_all()

    def on_image_cropper_delete(self, widget, event):
        logger.debug("Image cropper is destroyed")
        self.image_cropper = None 

    def on_image_scaler_delete(self, widget, event):
        logger.debug("Image scaler is destroyed")
        self.image_scaler = None 

    def on_spot_inpainter_delete(self, widget, event):
        logger.debug("Spot inpainter is destroyed")
        self.spot_inpainter = None 

    def on_face_fixer_delete(self, widget, event):
        logger.debug("Face fixer is destroyed")
        self.face_fixer = None 

    def on_graffiti_editor_delete(self, widget, event):
        logger.debug("Graffiti editor is destroyed")
        self.graffiti_editor = None 

    def on_image_segmenter_delete(self, widget, event):
        logger.debug("Image segmenter is destroyed")
        self.image_segmenter = None 

    def on_prompt_builder_delete(self, widget, event):
        logger.debug("Prompt builder is destroyed")
        self.prompt_builder = None 
