"""
Copyright (C) 2024, Hideyuki Inada. All rights reserved.

Algorithm
1 Detect faces
2 For each face,
2.1 resize so that the longest edge is the target edge length
2.2 keep the aspect ratio.
2.2 Pad the image so that image size will be target edge wide, target edge high.
2.3 Send to image to image
2.4 Remove the padding
2.5 Resize to the original dimension
2.6 Paste the updated image in the original image.

Face detection code for OpenCV is based on the code downloaded from
https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detect.py
Licensed under Apache License 2.0
https://github.com/opencv/opencv/blob/4.x/LICENSE

OpenCV face detection model: face_detection_yunet_2023mar.onnx
The model was downloaded from
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
Licensed under the MIT license.

See the license in the project root directory.
"""
import os
import sys
import logging
import argparse
import json
from io import BytesIO
import tempfile
import threading
import shutil
from typing import Dict, Any
from dataclasses import dataclass

from transformers import ViTImageProcessor, ViTForImageClassification  # For face classification
import numpy as np
import cv2 as cv
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
OPENCV_FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_ROOT, "opencv", "face_detection_yunet_2023mar.onnx")

from sd.img2img import img2img_parse_options_and_generate

from cremage.utils.image_utils import pil_image_to_pixbuf
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
SELECTED_COLOR =  (0, 0x7B/255.0, 0xFF/255.0, 0.5) # "#007BFF"
UNSELECTED_COLOR = (0xD3/255.0, 0xD3/255.0, 0xD3/255.0, 0.5)
FACE_FIX_TMP_DIR = os.path.join(get_tmp_dir(), "face_fix.tmp")
FACE_FIX_OUTPUT_DIR = os.path.join(FACE_FIX_TMP_DIR, "outputs")

@dataclass
class FaceRect:
    left: int
    top: int
    right: int
    bottom: int


class FaceFixer(Gtk.Window):  # Subclass Window object
    def __init__(self, pil_image=None,
                 output_file_path=None,
                 save_call_back=None,
                 positive_prompt=None,
                 negative_prompt=None,
                 generation_information_call_back=None,
                 preferences=None,
                 pil_face_image=None,
                 face_model_full_path=None):
        super().__init__(title="Face detection")

        self.face_rects = []
        self.selected_face_rect_index = None
        self.prev_x = None  # LMB press position
        self_prev_y = None
        self.output_file_path = output_file_path
        self.save_call_back = save_call_back
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.preferences = dict() if preferences is None else preferences
        self.generation_information_call_back = generation_information_call_back
        self.output_dir = None  # This is used for refreshing the image list, but not used for Face Fixer
        self.pil_face_image = pil_face_image
        self.face_model_full_path = face_model_full_path
        self.ldm_model_names = None # This is populated in update_ldm_model_name_value_from_ldm_model_dir

        update_ldm_model_name_value_from_ldm_model_dir(self)

        self.generation_information = dict()
        if self.generation_information_call_back is not None:
            d = self.generation_information_call_back()
            if d:  # The original image may not have generation info
                self.generation_information = d

        self.set_default_size(800, 600)  # width, height
        self.set_border_width(10)

        # Create a vertical Gtk.Box
        root_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(root_box)

        # Create a MenuBar
        ## Accelerator
        accel_group = Gtk.AccelGroup()
        self.add_accel_group(accel_group)

        menubar = Gtk.MenuBar()

        # File menu items
        filemenu = Gtk.Menu()
        file_item = Gtk.MenuItem(label="File")
        file_item.set_submenu(filemenu)

        # File | Exit
        exit_item = Gtk.MenuItem(label="Exit")
        exit_item.connect("activate", Gtk.main_quit)
        filemenu.append(exit_item)

        # File | Save
        save_item = Gtk.MenuItem(label="Save")  # Create save menu item
        save_item.connect("activate", self.on_save_activate)  # Connect to handler
        save_item.add_accelerator("activate", accel_group, ord('S'),
                                Gdk.ModifierType.CONTROL_MASK, Gtk.AccelFlags.VISIBLE)
        filemenu.append(save_item)  # Add save item to file menu

        menubar.append(file_item)
        root_box.pack_start(menubar, False, False, 0)

        # Horizontal Gtk.Box to contain the scrolled window and control elements
        container_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root_box.pack_start(container_box, True, True, 0)  # Add container_box to root_box under the menu

        # Create a ScrolledWindow
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_hexpand(True)
        scrolled_window.set_vexpand(True)
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        # Create an Image widget
        if pil_image is None:
            pil_image = Image.new('RGBA', (512, 768), "gray")
        self.pil_image = pil_image
        self.pil_image_original = self.pil_image  # Make a copy to restore
        pixbuf = pil_image_to_pixbuf(pil_image)

        # Create a Gtk.Image and set the Pixbuf
        # self.image = Gtk.Image.new_from_pixbuf(pixbuf)
        self.image = Gtk.DrawingArea()

        # Setup drag and drop for the image area
        self.image.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.image.add_events(Gdk.EventMask.BUTTON_RELEASE_MASK)
        self.image.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.image.drag_dest_add_text_targets()
        self.image.connect('drag-data-received', self.on_drag_data_received)

        self.image.connect("button-press-event", self.on_image_click)

        self.image.set_size_request(768, 768)
        scrolled_window.add(self.image)

        # Connect the signal for mouse click events
        self.image.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.image.connect("button-press-event", self.on_image_click)
        self.image.connect("button-release-event", self.on_image_button_release)  # LMB release

        # Connect to the draw event of the drawing area
        self.image.connect("draw", self.on_image_draw)

        # Add the ScrolledWindow to the root_box
        container_box.pack_start(scrolled_window,
                        True,  # expand this field as the parent container expand
                        True,  # take up the initially assigned space
                        0)

        # Vertical Box for controls next to the ScrolledWindow
        controls_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        container_box.pack_start(controls_box, False, True, 0)

        # Tool specific code start
        # Detection method dropdown
        detection_method_label = Gtk.Label(label="Detection method:")
        controls_box.pack_start(detection_method_label, False, True, 0)
        self.detection_method_combo = Gtk.ComboBoxText()
        self.detection_method_combo.append_text("OpenCV")
        self.detection_method_combo.append_text("InsightFace")
        self.detection_method_combo.set_active(0)
        controls_box.pack_start(self.detection_method_combo, False, True, 0)

        #
        # Auto Fix face button
        # Handles both detection and fixes
        auto_fix_button = Gtk.Button(label="Auto Fix")
        controls_box.pack_start(auto_fix_button, False, True, 0)
        auto_fix_button.connect("clicked", self.on_auto_fix_clicked)

        #
        # Detect button
        #
        detect_button = Gtk.Button(label="Detect faces")
        controls_box.pack_start(detect_button, False, True, 0)
        detect_button.connect("clicked", self.on_detect_clicked)

        #
        # Fix from marked face button
        #
        fix_marked_face_button = Gtk.Button(label="Fix marked faces")
        controls_box.pack_start(fix_marked_face_button, False, True, 0)
        fix_marked_face_button.connect("clicked", self.on_fix_marked_face_clicked)

        #
        # Delete current mark button
        #
        delete_mark_button = Gtk.Button(label="Delete mark")
        controls_box.pack_start(delete_mark_button, False, True, 0)
        delete_mark_button.connect("clicked", self.on_delete_mark_clicked)

        #
        # Clear marks button
        #
        clear_marks_button = Gtk.Button(label="Clear marks")
        controls_box.pack_start(clear_marks_button, False, True, 0)
        clear_marks_button.connect("clicked", self.on_clear_marks_clicked)

        # Disable face input
        self.disable_face_input_checkbox = Gtk.CheckButton(label="Disable face input")
        controls_box.pack_start(self.disable_face_input_checkbox, False, True, 0)
        # End of vertical control box area

        # Start of the prompt area
        # LDM model
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        model_name = self.preferences["ldm_model"]
        # Check to see if we have generation info to override
        if "ldm_model" in self.generation_information:
            model_name = self.generation_information["ldm_model"]
        ldm_label = Gtk.Label()
        ldm_label.set_text("Model")
        ldm_label.set_halign(Gtk.Align.START)  # Align label to the left
        box.pack_start(ldm_label, False, False, 0)
        
        self.ldm_model_cb = create_combo_box_typeahead(
            self.ldm_model_names,
            self.ldm_model_names.index(model_name))
        box.pack_start(self.ldm_model_cb, False, False, 0)
        root_box.pack_start(box, False, False, 0)

        # Denoising strength
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        denoising_label = Gtk.Label()
        denoising_label.set_text("Denoising strength:")
        box.pack_start(denoising_label, False, False, 0)
        self.denoising_entry = Gtk.Entry(text=str(0.3))
        box.pack_start(self.denoising_entry, False, False, 0)
        root_box.pack_start(box, False, False, 0)
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

        # Connect the key press event signal to the handler
        self.connect("key-press-event", self.on_key_press)

    def on_image_draw(self, widget, cr):
        pixbuf = pil_image_to_pixbuf(self.pil_image)
        Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
        cr.paint()

        # Draw a rectangle over the image
        for i, face_rect in enumerate(self.face_rects):
            if i == self.selected_face_rect_index:
                cr.set_source_rgba(*SELECTED_COLOR)
            else:
                cr.set_source_rgba(*UNSELECTED_COLOR)
            cr.rectangle(face_rect.left,
                         face_rect.top,
                         face_rect.right - face_rect.left,
                         face_rect.bottom - face_rect.top)
            cr.fill()

    def on_image_click(self, widget, event):
        x = event.x
        y = event.y

        self.prev_x = x
        self.prev_y = y

        # print(f"Click at ({x}, {y}) in drawing area")
        # Check to see if the inside of any box is clicked.
        self.selected_face_rect_index = None  # no box is selected
        for i, face_rect in enumerate(self.face_rects):
            if x >= face_rect.left and \
               x <= face_rect.right and \
               y >= face_rect.top and \
               y <= face_rect.bottom:
                self.selected_face_rect_index = i
                break
        self.image.queue_draw()  # refresh image canvas

    def on_key_press(self, widget, event):
        if event.state & Gdk.ModifierType.CONTROL_MASK:
            if event.keyval == Gdk.KEY_s:
                logger.info("The 's' key was pressed (with Ctrl).")
        else:
            # Ctrl is not pressed, handle other key presses as needed
            if event.keyval == Gdk.KEY_s:
                if self.output_file_path:
                    logger.info(f"Saving image as {self.output_file_path}")
                    self.pil_image.save(self.output_file_path)
                else:
                    logger.warn("detected image file path is not set")

    def on_auto_fix_clicked(self, widget):
        detection_method = self.detection_method_combo.get_active_text()
        logger.info(detection_method)

        if detection_method == "InsightFace":
            annotated_image = self.fix_with_insight_face(self.pil_image)
        elif detection_method == "OpenCV":
            annotated_image = self.fix_with_opencv(self.pil_image)
        self.pil_image = annotated_image

        generation_information_dict = self.generation_information
        if "additional_processing" not in generation_information_dict:
            generation_information_dict["additional_processing"] = list()

        generation_information_dict["additional_processing"].append(
            {"face_fixer":[]}
        )
        str_generation_params = json.dumps(generation_information_dict)
        if self.output_file_path:
            metadata = PngInfo()
            metadata.add_text("generation_data", str_generation_params)
            self.pil_image.save(self.output_file_path, pnginfo=metadata)
        self.image.queue_draw()  # refresh image canvas

        if self.save_call_back:
            self.save_call_back(self.pil_image, str_generation_params)

        return

    def mark_face_with_opencv(self, pil_image):
        parser = argparse.ArgumentParser()
        parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
        parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
        parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
        parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
        parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
        args = parser.parse_args()

        detector = cv.FaceDetectorYN.create(
            OPENCV_FACE_DETECTION_MODEL_PATH,  # model
            "",  # config
            (320, 320), # input size
            args.score_threshold,  # score threshold
            args.nms_threshold,  # nms threshold
            args.top_k # top_k
        )

        # Prepare image for detection
        img1 = pil_image.convert("RGB")  # Convert to RGB from RGBA
        img1 = np.asarray(img1, dtype=np.uint8)[:,:,::-1]  # to np and RGB to BGR
        img1Width = int(img1.shape[1]*args.scale)
        img1Height = int(img1.shape[0]*args.scale)

        img1 = cv.resize(img1, (img1Width, img1Height))
        detector.setInputSize((img1Width, img1Height))
        face_data = detector.detect(img1)

        return face_data

    def mark_face_with_insight_face(self, pil_image):
        from face_detector_insight_face import get_face_bounding_boxes

        # Prepare image for detection
        img1 = np.asarray(pil_image.convert("RGB"))
        bboxes = get_face_bounding_boxes(img1)
        return bboxes

    def fix_with_insight_face(self, pil_image: Image) -> Image:
        """
        Detects faces in the source image and annotates the image with detected faces.

        Args:
            pil_image (Image): Input image with faces
        Returns:
            Annotated image
        """
        from face_detector_insight_face import get_face_bounding_boxes
        logger.info("Using InsightFace for face detection")
        img1 = np.asarray(pil_image.convert("RGB"))
        bboxes = get_face_bounding_boxes(img1)
        return self.fix_engine(pil_image, bboxes, detection_method="InsightFace")

    def fix_with_opencv(self, pil_image: Image) -> Image:
        """
        Detects faces in the source image and annotates the image with detected faces.

        Args:
            pil_image (Image): Input image with faces
        Returns:
            Annotated image
        """
        face_data = self.mark_face_with_opencv(pil_image)

        return self.fix_engine(pil_image, face_data)

    def fix_engine(self, pil_image: Image, face_data,
                   detection_method="OpenCV") -> Image:
        """
        Detects faces in the source image and annotates the image with detected faces.

        Args:
            pil_image (Image): Input image with faces
        Returns:
            Annotated image
        """
        if face_data is not None:

            # Draw results on the input image
            self.parse_face_data(face_data, detection_method=detection_method)
            self.image.queue_draw()

            if detection_method == "OpenCV":
                faces = face_data[1]
                if faces is not None:
                    for face in faces:
                        pil_image = self.process_face(pil_image, face)
            elif detection_method == "InsightFace":
                faces = face_data
                if len(faces) > 0:
                    for face in faces:
                        # left, top, right, bottom to left, top, w, h
                        face = [face[0], face[1], face[2]-face[0], face[3]-face[1]]
                        pil_image = self.process_face(pil_image, face)

        return pil_image

    def on_detect_clicked(self, widget):

        detection_method = self.detection_method_combo.get_active_text()
        logger.info(detection_method)

        if detection_method == "InsightFace":
            face_data = self.mark_face_with_insight_face(self.pil_image)
        elif detection_method == "OpenCV":
            face_data = self.mark_face_with_opencv(self.pil_image)

        if face_data is not None:
            # Draw results on the input image
            self.parse_face_data(face_data, detection_method=detection_method)
            self.image.queue_draw()

    def on_fix_marked_face_clicked(self, widget):
        pil_image = self.pil_image

        faces = self.face_rects

        if faces is not None:
            for face_rect in faces:
                face = (face_rect.left,
                        face_rect.top,
                        face_rect.right - face_rect.left,
                        face_rect.bottom - face_rect.top)
                pil_image = self.process_face(pil_image, face)

        self.pil_image = pil_image
        generation_information_dict = self.generation_information
        if "additional_processing" not in generation_information_dict:
            generation_information_dict["additional_processing"] = list()

        generation_information_dict["additional_processing"].append(
            {"face_fixer":[]}
        )
        str_generation_params = json.dumps(generation_information_dict)
        if self.output_file_path:
            metadata = PngInfo()
            metadata.add_text("generation_data", str_generation_params)
            self.pil_image.save(self.output_file_path, pnginfo=metadata)

        self.image.queue_draw()  # refresh image canvas

        if self.save_call_back:
            self.save_call_back(self.pil_image, str_generation_params)

    def on_delete_mark_clicked(self, widget):
        if self.selected_face_rect_index is not None:
            del self.face_rects[self.selected_face_rect_index]
            self.selected_face_rect_index = None
        self.image.queue_draw()  # refresh image canvas

    def on_clear_marks_clicked(self, widget):
        self.face_rects.clear()
        self.selected_face_rect_index = None
        self.image.queue_draw()  # refresh image canvas

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
        pixbuf = pil_image_to_pixbuf(self.pil_image)
        self.image.set_from_pixbuf(pixbuf)

    def process_face(self, pil_image, face) -> Image:
        """

        x
        y
        w
        h
        score
        """
        # Gender classification
        logger.info(f"ViTImageProcessor and ViTForImageClassification connection to internet disabled : {local_files_only_value}")
        processor = ViTImageProcessor.from_pretrained('rizvandwiki/gender-classification',
                                                      local_files_only=local_files_only_value)
        model = ViTForImageClassification.from_pretrained('rizvandwiki/gender-classification',
                                                          local_files_only=local_files_only_value)

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
            print(cropped_image.size)

            # TODO: Detect race and age
            inputs = processor(images=cropped_image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_gender = model.config.id2label[predicted_class_idx]  # "male" or "female"
            logging.info(f"Predicted gender: {predicted_gender}")

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
                input_image=base_image,
                meta_prompt=predicted_gender)
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
            # pil_image.paste(updated_face_pil_image, (x, y))

            # Convert both base and face to CV2 BGR
            cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
            updated_face_cv_image = cv.cvtColor(
                np.array(updated_face_pil_image),
                cv.COLOR_RGB2BGR)

            # Compute the center of the face in the base image coordinate
            # shape[0] is face height
            # shape[1] is face width
            center_position = (x + updated_face_cv_image.shape[1] // 2,
                               y + updated_face_cv_image.shape[0] // 2)

            # Create a mask of the same size as the updated face, filled with 255 (white)
            mask = 255 * np.ones(updated_face_cv_image.shape, updated_face_cv_image.dtype)

            # Use seamlessClone to blend the updated face onto the original image
            result_image = cv.seamlessClone(
                updated_face_cv_image,
                cv_image, mask,
                # center_position, cv.MIXED_CLONE)
                center_position, cv.NORMAL_CLONE)

            # Convert the result back to a PIL image
            pil_image = Image.fromarray(cv.cvtColor(result_image, cv.COLOR_BGR2RGB))
            return pil_image

    def face_image_to_image(self, input_image=None, meta_prompt=None,
                            output_dir=FACE_FIX_TMP_DIR):  # FIXME
        """
        Event handler for the Generation button click

        Args:
            meta_prompt (str): Gender string of the face detected by the gender ML model
        """
        logger.info("face_image_to_image")

        generation_info = self.generation_information

        # Prompt handling
        # Priority 1. User-provided for face fix
        self.positive_prompt = text_view_get_text(self.positive_prompt_field)

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
            positive_prompt = "face of " + meta_prompt + ", " + positive_prompt

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

        if os.path.exists(output_dir) and output_dir.find("face_fix") >= 0:
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        vae_path = "None" if self.preferences["vae_model"] == "None" \
            else os.path.join(
                    self.preferences["vae_model_path"],
                    self.preferences["vae_model"])

        model_name = self.ldm_model_cb.get_child().get_text() # Use the model on UI
        clip_skip = str(self.preferences["clip_skip"])
        lora_models, lora_weights = generate_lora_params(self.preferences)

        # Check to see if we have generation info to override

        if "clip_skip" in generation_info:
            clip_skip = str(generation_info["clip_skip"])

        model_path = os.path.join(self.preferences["ldm_model_path"],
                                  model_name)

        if "lora_models" in generation_info:
            if generation_info["lora_models"] and len(generation_info["lora_models"]) > 0:
                l = generation_info["lora_models"].split(",")
                l = [os.path.join(
                    self.preferences["lora_model_path"], e.strip()) for e in l if len(e.strip()) > 0]
                l = ",".join(l)
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
        input_image_path = os.path.join(get_tmp_dir(), "input_image.png")
        input_image.save(input_image_path)

        args_list += [
            "--init-img", input_image_path,
            "--strength", self.denoising_entry.get_text()
        ]

        # FaceID
        if self.pil_face_image and self.disable_face_input_checkbox.get_active() is False:  # app.face_input_image_original_size:
            face_input_image_path = os.path.join(get_tmp_dir(), "face_input_image.png")
            self.pil_face_image.save(face_input_image_path)

            args_list += [
                "--face_input_img", face_input_image_path,
                "--face_model", self.face_model_full_path
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
        if os.path.exists(file_name):
            image = Image.open(file_name)
            image2 = image.copy()
            image.close()
            os.remove(file_name)  # Remove the temporary image
            image = image2
            return image
        else:
            raise ValueError(f"Invalid output file from img2img {file_name}")

    def update_image(self, img_data: Image, generation_parameters:str=None) -> None:
        pass

    def parse_face_data(self, faces, detection_method="OpenCV") -> None:
        self.face_rects.clear()

        if detection_method=="OpenCV":
            if faces is not None and faces[1] is not None:
                for i, face in enumerate(faces[1]):
                    face_rect = FaceRect(
                        face[0],
                        face[1],
                        face[0]+face[2],
                        face[1]+face[3]
                    )
                    self.face_rects.append(face_rect)
        elif detection_method == "InsightFace":
            if len(faces) > 0:
                for i, face in enumerate(faces):
                    face_rect = FaceRect(
                        face[0],
                        face[1],
                        face[2],
                        face[3]
                    )
                    self.face_rects.append(face_rect)

    def on_image_button_release(self, widget, event):
        x, y = (event.x, event.y)
        if abs(x - self.prev_x) > 5 and  abs(y - self.prev_y) > 5:
            left = min(x, self.prev_x)
            right = max(x, self.prev_x)
            top = min(y, self.prev_y)
            bottom = max(y, self.prev_y)
            face_rect = FaceRect(left, top, right, bottom)
            self.face_rects.append(face_rect)

            self.image.queue_draw()  # invalidate


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
        "vae_model_path": "/media/pup/ssd2/recoverable_data/sd_models/VAE",
        "vae_model": "vae-ft-mse-840000-ema-pruned.ckpt",
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
        "embedding_path": "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
        "positive_prompt_expansion": ", highly detailed, photorealistic, 4k, 8k, uhd, raw photo, best quality, masterpiece",
        "negative_prompt_expansion": ", drawing, 3d, worst quality, low quality, disfigured, mutated arms, mutated legs, extra legs, extra fingers, badhands",
        "enable_positive_prompt_expansion": True, # False,
        "enable_negative_prompt_expansion": True, # False,
        "seed": "0"
    }

    pil_image = Image.open("human_couple.png")   # FIXME
    app = FaceFixer(
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
