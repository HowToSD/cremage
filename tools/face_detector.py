"""
Copyright (C) 2024, Hideyuki Inada. All rights reserved.

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
from io import BytesIO
from typing import List

import numpy as np
import cv2 as cv
from PIL import Image
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
from cremage.utils.image_utils import pil_image_to_pixbuf
from cremage.utils.gtk_utils import show_alert_dialog
from gfpgan_wrapper import gfp_wrapper

OPENCV_FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_ROOT, "opencv", "face_detection_yunet_2023mar.onnx")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def annotate_face(input: np.ndarray, faces, thickness=2) -> None:
    """
    Taken from https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detect.py
    Apache License 2.0
    """
    if faces is not None and faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

def annotate_face_insight_face(input: np.ndarray, bboxes: List[np.ndarray], thickness=2) -> None:
    """
    Mark bounding boxes for each face in the input image.

    Args:
        input: Image with faces
        bboxes: Face bounding boxes. Each bbox has (left, top, right, bottom)
    """
    input = cv.cvtColor(input, cv.COLOR_RGB2BGR)  # to BGR
    for idx, face in enumerate(bboxes):

        face = face.astype(np.int32)
        print(face)
        cv.rectangle(input, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), thickness)

    input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    return input

class FaceDetector(Gtk.Window):  # Subclass Window object
    def __init__(self, pil_image=None, annotated_file_path=None,
                 save_call_back=None):
        super().__init__(title="Face detection")

        self.annotated_file_path = annotated_file_path
        self.save_call_back = save_call_back
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
        self.image = Gtk.Image.new_from_pixbuf(pixbuf)

        # Setup drag and drop for the image area
        self.image.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.image.drag_dest_add_text_targets()
        self.image.connect('drag-data-received', self.on_drag_data_received)

        # Add the Image to the ScrolledWindow
        scrolled_window.add(self.image)

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

        # Detect button
        detect_button = Gtk.Button(label="Detect")
        controls_box.pack_start(detect_button, False, True, 0)
        detect_button.connect("clicked", self.on_detect_clicked)

        # Connect the key press event signal to the handler
        self.connect("key-press-event", self.on_key_press)
            
    def on_key_press(self, widget, event):
        if event.state & Gdk.ModifierType.CONTROL_MASK:
            if event.keyval == Gdk.KEY_s:
                logger.info("The 's' key was pressed (with Ctrl).")
        else:
            # Ctrl is not pressed, handle other key presses as needed
            if event.keyval == Gdk.KEY_s:
                if self.annotated_file_path:
                    logger.info(f"Saving image as {self.annotated_file_path}")
                    self.pil_image.save(self.annotated_file_path)
                else:
                    logger.warning("detected image file path is not set")

    def on_detect_clicked(self, widget):
        detection_method = self.detection_method_combo.get_active_text()
        logger.info(detection_method)

        if detection_method == "InsightFace":
            annotated_image = self.detect_with_insight_face(self.pil_image)
        elif detection_method == "OpenCV":
            annotated_image = self.detect_with_opencv(self.pil_image)
        self.pil_image = annotated_image
        pixbuf = pil_image_to_pixbuf(self.pil_image)
        self.image.set_from_pixbuf(pixbuf)

        return

    def detect_with_insight_face(self, pil_image: Image) -> Image:
        """
        Detects faces in the source image and annotates the image with detected faces.

        Args:
            pil_image (Image): Input image with faces
        Returns:
            Annotated image
        """
        from face_detector_insight_face import get_face_bounding_boxes

        # Prepare image for detection
        img1 = np.asarray(pil_image.convert("RGB"))
        bboxes = get_face_bounding_boxes(img1)
        if len(bboxes) >= 0:
            # Draw results on the input image
            img1 = annotate_face_insight_face(img1, bboxes)
        return Image.fromarray(img1)


    def detect_with_opencv(self, pil_image: Image) -> Image:
        """
        Detects faces in the source image and annotates the image with detected faces.

        Args:
            pil_image (Image): Input image with faces
        Returns:
            Annotated image
        """
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
        faces = detector.detect(img1)
        if faces is not None:
            # Draw results on the input image
            annotate_face(img1, faces)
        return Image.fromarray(img1[:,:,::-1])  # BGR to RGB and to PIL image

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


def main():
    pil_image = Image.open("human_couple.png")   # FIXME
    app = FaceDetector(pil_image=pil_image, annotated_file_path="tmp_scaled.png")
    app.connect('destroy', Gtk.main_quit)
    app.show_all()
    Gtk.main()

if __name__ == '__main__':
    main()
