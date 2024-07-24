"""
Copyright (C) 2024, Hideyuki Inada. All rights reserved.

SegFormer code was adopted from:
https://huggingface.co/mattmdjaga/segformer_b2_clothes
See third_party_license.md.

Segment an image and inpaint.
"""
import os
import sys

if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

import logging
import tempfile
import threading
import shutil
from typing import Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import cv2 as cv
import PIL
from PIL import Image
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
from tool_base import ToolBase
from cremage.utils.misc_utils import get_tmp_file
from cremage.utils.gtk_utils import show_alert_dialog, set_pil_image_to_gtk_image
from cremage.utils.image_utils import pil_image_to_pixbuf, get_single_bounding_box_from_grayscale_image
from cremage.utils.gtk_utils import text_view_get_text, create_combo_box_typeahead
from cremage.utils.misc_utils import get_tmp_dir
from cremage.utils.misc_utils import generate_lora_params
from cremage.ui.model_path_update_handler import update_ldm_model_name_value_from_ldm_model_dir
from sd.options import parse_options as sd15_parse_options
from sd.img2img import generate as sd15_img2img_generate

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


TARGET_EDGE_LEN = 512  # FIXME
SPOT_FIX_TMP_DIR = os.path.join(get_tmp_dir(), "spot_fix.tmp")

SELECTED_COLOR =  (0, 0x7B/255.0, 0xFF/255.0, 0.5) # "#007BFF"
UNSELECTED_COLOR = (0xD3/255.0, 0xD3/255.0, 0xD3/255.0, 0.5)

CLASS_TO_LABEL_MAP = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf"
}

@dataclass
class BoxRect:
    left: int
    top: int
    right: int
    bottom: int


class ImageSegmenter(ToolBase):  # Subclass Window object

    def __init__(
            self,
            title:str = "Segmentation inpainting",
            **kwargs):
        super().__init__(title="Segmentation inpainting", **kwargs)

    def set_up_ui(self):

        self.show_mask = True  # To display mask (area to be inpainted) on canvas
        self.show_bounding_box = False  # To display a bounding box around the mask
        self.bounding_boxes = []
        self.selected_bounding_box_index = None
        self.prev_x = None  # LMB press position
        self_prev_y = None
        self.pen_width = 10
        self.is_eraser = False

        self.ldm_model_names = None # This is populated in update_ldm_model_name_value_from_ldm_model_dir
        self.enable_lora = False
        update_ldm_model_name_value_from_ldm_model_dir(self)

        self.generation_information = dict()
        if self.generation_information_call_back is not None:
            d = self.generation_information_call_back()
            if d:  # The original image may not have generation info
                self.generation_information = d

        # UI definition
        self.width, self.height = self.pil_image.size
        self.set_default_size(600, 800)  # width, height
        self.set_border_width(10)

        if self.pil_image and self.output_file_path is None:
            raise ValueError("Output file path is not specified when input_image is not None")

        self.segmented_raw_image = None  # raw output of the model
        self.segmented_image = None  # RGB image. Black pixels do not contain the segment
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

        #
        # Apply inpainting
        #
        apply_inpainting_button = Gtk.Button(label="Apply inpainting")
        controls_box.pack_start(apply_inpainting_button, False, True, 0)
        apply_inpainting_button.connect("clicked", self.on_apply_inpainting_clicked)

        # Segment button
        segment_button = Gtk.Button(label="Segment")
        controls_box.pack_start(segment_button, False, True, 0)
        segment_button.connect("clicked", self.on_segment_clicked)

        # Checkboxes
        self.checkboxes = list()
        for i in range(18):
            checkbox = Gtk.CheckButton(label=f"{CLASS_TO_LABEL_MAP[i]}")
            controls_box.pack_start(checkbox, False, True, 0)
            checkbox.connect("toggled", self.on_checkbox_toggled)
            self.checkboxes.append(checkbox)

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
        denoise_label = Gtk.Label()
        denoise_label.set_text("Denoising strength")
        denoise_label.set_halign(Gtk.Align.START)  # Align label to the left
        box.pack_start(denoise_label, False, False, 0)

        self.denoise_text = Gtk.Entry(text=self.preferences["denoising_strength"])
        box.pack_start(self.denoise_text, False, True, 0)

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
        cv_img = np.array(self.segmented_image).astype(np.uint8)
        cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)

        # Clip by the image size
        cv_img = cv_img[0:self.pil_image.size[1], 0:self.pil_image.size[0]]
        return cv_img

    def detect_bounding_boxes(self):
        cv_img = np.array(self.segmented_image)
        cv_img = cv.cvtColor(cv_img, cv.COLOR_RGBA2GRAY)

        box = get_single_bounding_box_from_grayscale_image(cv_img)
        # boxes = get_bounding_boxes_from_grayscale_image(cv_img)
        # logger.info(boxes)
        # self.parse_face_data(boxes)
        self.parse_face_data([box])
        
        logger.info(self.bounding_boxes)

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

        pixbuf = pil_image_to_pixbuf(pil_image)
        self.image_view.set_from_pixbuf(pixbuf)

        # Update the base_image. If the user segments again, this will be used.
        self.pil_image = pil_image

        if self.output_file_path:
            pil_image.save(self.output_file_path)

        if self.save_call_back:
            self.save_call_back(pil_image, self.generation_information_call_back())

    def on_clear_marks_clicked(self, widget):
        self.bounding_boxes.clear()
        self.lines_list.clear()
        self.drawing_area.queue_draw()  # refresh image canvas

    def process_box(self, pil_image, cv_mask_image, face) -> Image:
        """

        x
        y
        w
        h
        score
        """
        input_image = pil_image

        input_image.save(os.path.join(get_tmp_dir(), "tmp_input_image.png"))
        cv.imwrite(os.path.join(get_tmp_dir(), "tmp_mask_image.png"), cv_mask_image)

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
            updated_face_pil_image.save(os.path.join(get_tmp_dir(), "tmpface.jpg"))

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
            cv.imwrite(os.path.join(get_tmp_dir(), "tmp_inpainted_image_cv.png"), inpainted_image_cv)

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

        # FIXME
        # # Negative prompt
        # if self.negative_prompt:
        #     negative_prompt = self.negative_prompt
        #     if self.preferences["enable_negative_prompt_expansion"]:
        #         negative_prompt += self.preferences["negative_prompt_expansion"]
        # elif generation_info is not None and "negative_prompt" in generation_info: # Priority 2. Generation
        #     negative_prompt = generation_info["negative_prompt"]
        # else:  # use blank
        #     negative_prompt = ""
        #     if self.preferences["enable_negative_prompt_expansion"]:
        #         negative_prompt += self.preferences["negative_prompt_expansion"]
        negative_prompt = "low quality, worst quality"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        vae_path = "None" if self.preferences["vae_model"] == "None" \
            else os.path.join(
                    self.preferences["vae_model_path"],
                    self.preferences["vae_model"])

        model_name = self.ldm_model_cb.get_child().get_text() # Use the model on UI
        clip_skip = str(self.preferences["clip_skip"])
        lora_models, lora_weights = generate_lora_params(self.preferences)

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
        options = sd15_parse_options(args_list)
        generate_func=sd15_img2img_generate

        # Start the image generation thread
        thread = threading.Thread(
            target=generate_func,
            kwargs={'options': options,
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


    def on_checkbox_toggled(self, checkbox):
        # add color
        base_image = np.asarray(self.pil_image)
        print(base_image.shape)  # h, w, c

        segmented_image = np.zeros_like(base_image)
        # display_image = base_image.copy()
        # display_image = np.array(display_image)

        if self.segmented_raw_image is None: # 0-17
            return

        for i in range(18):
            if self.checkboxes[i].get_active():  # if checkbox is checked
                # Mark the segment with light green
                r = 200
                g = 255
                b = 200

                marked_region = (self.segmented_raw_image == i).astype(np.uint8)  # 1 or 0
                assert len(marked_region.shape) == 2
                non_zero_pixels = np.count_nonzero(marked_region)
                logger.debug(f"Number of non-zero pixels for class {i}: {non_zero_pixels}")

                # Convert to an RGB image
                marked_region = marked_region[..., np.newaxis]  # add channel
                marked_region = np.concatenate([marked_region] * 3, axis=-1)

                # Set color for the segment. Note that unselected pixels have 0s, so this only changes
                # color for the marked class.
                marked_region[:,:,0] *= r
                marked_region[:,:,1] *= g
                marked_region[:,:,2] *= b
                assert len(marked_region.shape) == 3

                segmented_image += marked_region

        a = base_image.copy()
        s = segmented_image
        w = 0.3
        a[:,:,0] = w * a[:,:,0] + (1-w) * s[:,:,0]
        a[:,:,1] = w * a[:,:,1] + (1-w) * s[:,:,1]
        a[:,:,2] = w * a[:,:,2] + (1-w) * s[:,:,2]
        base_image = Image.fromarray(a.astype(np.uint8))
        self.segmented_image = Image.fromarray(s.astype(np.uint8))
        pixbuf = pil_image_to_pixbuf(base_image)
        self.image_view.set_from_pixbuf(pixbuf)

    def on_segment_clicked(self, widget):
        processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes", local_files_only=local_files_only_value)
        model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes", local_files_only=local_files_only_value)
        inputs = processor(images=self.pil_image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=self.pil_image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        self.segmented_raw_image = pred_seg.detach().cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(self.segmented_raw_image*15)
        self.output_pil_image = pil_image
        if self.output_file_path:
            self.output_pil_image.save(self.output_file_path)

        pixbuf = pil_image_to_pixbuf(self.output_pil_image)
        self.image_view.set_from_pixbuf(pixbuf)
        if self.save_call_back:
            super().on_update_clicked(widget)  # Send output to the caller
        return

    def on_drag_data_received(self, widget, drag_context, x, y, data, info, time):
        """Drag and Drop handler.

        data: Contains info for the dragged file name
        """
        file_path = data.get_text().strip()
        if file_path.startswith('file://'):
            file_path = file_path[7:]
            self.pil_image = Image.open(file_path)
            set_pil_image_to_gtk_image(self.pil_image, self.image_view)

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
        "enable_positive_prompt_expansion": True,
        "enable_negative_prompt_expansion": True,
        "seed": "0"
    }

    pil_image = Image.open("../cremage_resources/512x512_human_couple.png")   # FIXME
    app = ImageSegmenter(
        pil_image=pil_image,
        output_file_path=os.path.join(get_tmp_dir(), "tmp_face_fix.png"),
        preferences=preferences)
    app.connect('destroy', Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()


