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
from cremage.ui.model_path_update_handler import update_ldm_model_name_value_from_ldm_model_dir
from cremage.ui.model_path_update_handler import update_sdxl_ldm_model_name_value_from_sdxl_ldm_model_dir
from cremage.utils.gtk_utils import update_combo_box

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

PROMPT_BUILDER_INPUT_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "prompt_builder")  # FIXME. Read from config
ITEMS_PER_ROW = 4

class ToolPaletteArea():
    
    def __init__(self,
                 parent_box: Gtk.Box,
                 get_current_image_call_back=None,
                 get_tool_processed_file_path_call_back=None,
                 get_tool_processed_video_call_back=None,
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
        self.get_tool_processed_video_call_back = get_tool_processed_video_call_back
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
                            self.on_prompt_builder_clicked,
                            self.on_model_mixer_clicked,
                            self.on_video_generator_clicked,
                            self.on_llm_interactor_clicked]
        tool_names = [
            "Crop",
            "Scale",
            "Spot inpainting",
            "Face fix",
            "Graffiti editor",
            "Segmentation inpainting",
            "Visual prompt builder",
            "Model mixer",
            "Video generator",
            "LLM interactor"]

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
        self.model_mixer = None
        self.video_generator = None
        self.llm_interactor = None
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
        self.spot_inpainter.set_visibility()

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

    def on_model_mixer_clicked(self, widget, event):
        """
        Event handler for model mixer
        """
        logger.debug("Model mixer clicked")
        # Do not move this to to the top as we want to lazy import
        from model_mixer import ModelMixer

        def update_model_names():
            # Rescan directory to update the list
            update_ldm_model_name_value_from_ldm_model_dir(self.app)
            update_sdxl_ldm_model_name_value_from_sdxl_ldm_model_dir(self.app)

            # Update the UI
            update_combo_box(
                self.app.fields["ldm_model"],
                self.app.ldm_model_names,
                self.app.ldm_model_names.index(self.app.preferences["ldm_model"]))

            update_combo_box(
                self.app.fields["sdxl_ldm_model"],
                self.app.sdxl_ldm_model_names,
                self.app.sdxl_ldm_model_names.index(self.app.preferences["sdxl_ldm_model"]))

        self.model_mixer = ModelMixer(
            ldm_model_dir=self.app.preferences["ldm_model_path"],
            sdxl_ldm_model_dir=self.app.preferences["sdxl_ldm_model_path"],
            vae_model_dir=self.app.preferences["vae_model_path"],
            sdxl_vae_model_dir=self.app.preferences["sdxl_vae_model_path"],
            callback=update_model_names
        )
        self.model_mixer.connect("delete-event", self.on_model_mixer_delete)
        self.model_mixer.show_all()

    def on_video_generator_clicked(self, widget, event):
        """
        Event handler for video generator
        """
        logger.debug("Video generator clicked")
        # Do not move this to to the top as we want to lazy import
        from video_generator import VideoGenerator
        
        self.video_generator = VideoGenerator(
            pil_image=self.get_current_image_call_back(),
            output_file_path=self.get_tool_processed_video_call_back(),
            positive_prompt=None,
            negative_prompt=None,
            preferences=self.app.preferences,
            checkpoint_path=os.path.join(
                self.app.preferences["svd_model_path"], "svd_xt_1_1.safetensors"))
        self.video_generator.connect("delete-event", self.on_video_generator_delete)
        self.video_generator.show_all()

    def on_llm_interactor_clicked(self, widget, event):
        """
        Event handler for LLM interactor
        """
        logger.debug("LLM interactor clicked")
        # Do not move this to to the top as we want to lazy import
        from llm_interactor import LLMInteractor
        
        self.llm_interactor = LLMInteractor(
            pil_image=self.get_current_image_call_back(),
            preferences=self.app.preferences)
        self.llm_interactor.connect("delete-event", self.on_llm_interactor_delete)
        self.llm_interactor.show_all()

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

    def on_model_mixer_delete(self, widget, event):
        logger.debug("Model mixer is destroyed")
        self.model_mixer = None 

    def on_video_generator_delete(self, widget, event):
        logger.debug("Video generator is destroyed")
        self.video_generator = None 

    def on_llm_interactor_delete(self, widget, event):
        logger.debug("LLM interactor is destroyed")
        self.llm_interactor = None 
