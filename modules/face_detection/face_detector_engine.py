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
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

import logging
import argparse
from io import BytesIO
from typing import List

import numpy as np
import cv2 as cv
import PIL
from PIL import Image
import tempfile
import threading
import shutil
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from transformers import ViTImageProcessor, ViTForImageClassification  # For face classification

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.misc_utils import get_tmp_dir
from cremage.utils.misc_utils import generate_lora_params
from face_detection.face_img2img import face_image_to_image

from sd.options import parse_options as sd15_parse_options
from sd.img2img import generate as sd15_img2img_generate
from sd.inpaint import generate as sd15_inpaint_generate

FACE_FIX_TMP_DIR = os.path.join(get_tmp_dir(), "face_fix.tmp")
FACE_FIX_OUTPUT_DIR = os.path.join(FACE_FIX_TMP_DIR, "outputs")
OPENCV_FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_ROOT, "opencv", "face_detection_yunet_2023mar.onnx")
from cremage.const.const import GMT_SD_1_5, GMT_SDXL

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

@dataclass
class FaceRect:
    left: int
    top: int
    right: int
    bottom: int


def parse_face_data(faces, detection_method="OpenCV") -> List:
    face_rects = list()

    if detection_method=="OpenCV":
        if faces is not None and faces[1] is not None:
            for i, face in enumerate(faces[1]):
                face_rect = FaceRect(
                    face[0],
                    face[1],
                    face[0]+face[2],
                    face[1]+face[3]
                )
                face_rects.append(face_rect)
    elif detection_method == "InsightFace":
        if len(faces) > 0:
            for i, face in enumerate(faces):
                face_rect = FaceRect(
                    face[0],
                    face[1],
                    face[2],
                    face[3]
                )
                face_rects.append(face_rect)
    return face_rects


def process_face(
                    pil_image,
                    face,
                    positive_prompt = "",
                    negative_prompt = "",
                    generator_model_type = GMT_SD_1_5,
                    model_path=None,
                    lora_models = None,
                    lora_weights = None,
                    embedding_path = None,
                    sampling_steps = None,
                    seed = 0,
                    vae_path = None,
                    sampler = None,
                    target_edge_len = None,
                    denoising_strength = None,
                    enable_face_id = False,
                    face_input_image_path = None,
                    face_model_full_path = None,
                    discretization = None,
                    discretization_sigma_min = None,
                    discretization_sigma_max = None,
                    discretization_rho = None,
                    guider = None,
                    linear_prediction_guider_min_scale = None,
                    linear_prediction_guider_max_scale = None,
                    triangle_prediction_guider_min_scale = None,
                    triangle_prediction_guider_max_scale = None,
                    sampler_s_churn = None,
                    sampler_s_tmin = None,
                    sampler_s_tmax = None,
                    sampler_s_noise = None,
                    sampler_eta = None,
                    sampler_order = None,
                    clip_skip = None
                 ) -> Image:
    """

    x
    y
    w
    h
    score
    """
    if target_edge_len is None:
        if generator_model_type == GMT_SDXL:
            target_edge_len = 1024
        else:
            target_edge_len = 512

    # Gender classification
    logger.info(f"ViTImageProcessor and ViTForImageClassification connection to internet disabled : {local_files_only_value}")
    processor = ViTImageProcessor.from_pretrained('rizvandwiki/gender-classification',
                                                    local_files_only=local_files_only_value)
    model = ViTForImageClassification.from_pretrained('rizvandwiki/gender-classification',
                                                        local_files_only=local_files_only_value)

    # Create a temporary directory using the tempfile module
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(f"Temporary directory created at {temp_dir}")
        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])
        logger.debug(f"{x}, {y}, {w}, {h}")

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
        cropped_image = cropped_image.convert("RGB")  # RGBA causes a problem with processor
        # TODO: Detect race and age
        inputs = processor(images=cropped_image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_gender = model.config.id2label[predicted_class_idx]  # "male" or "female"
        logging.info(f"Predicted gender: {predicted_gender}")

        if w > h:  # landscape
            new_h = int(h * target_edge_len / w)
            new_w = target_edge_len
            padding_h = target_edge_len - new_h
            padding_w = 0
            padding_x = int(padding_w/2)
            padding_y = int(padding_h/2)

        else:
            new_w = int(w * target_edge_len / h)
            new_h = target_edge_len
            padding_w = target_edge_len - new_w
            padding_h = 0
            padding_x = int(padding_w/2)
            padding_y = int(padding_h/2)

        resized_image = cropped_image.resize((new_w, new_h), resample=PIL.Image.LANCZOS)

        # Pad image
        base_image = Image.new('RGBA', (target_edge_len, target_edge_len), "white")
        base_image.paste(resized_image, (padding_x, padding_y))

        # 2.3 Send to image to image
        updated_face_pil_image = face_image_to_image(
            input_image=base_image,
            meta_prompt=predicted_gender,
            positive_prompt = positive_prompt,
            negative_prompt = negative_prompt,
            generator_model_type = generator_model_type,
            model_path=model_path,
            lora_models = lora_models,
            lora_weights = lora_weights,
            embedding_path = embedding_path,
            sampling_steps = sampling_steps,
            seed = seed,
            vae_path = vae_path,
            sampler = sampler,
            target_edge_len = target_edge_len,
            denoising_strength = denoising_strength,
            enable_face_id = enable_face_id,
            face_input_image_path = face_input_image_path,
            face_model_full_path = face_model_full_path,
            discretization = discretization,
            discretization_sigma_min = discretization_sigma_min,
            discretization_sigma_max = discretization_sigma_max,
            discretization_rho = discretization_rho,
            guider = guider,
            linear_prediction_guider_min_scale = linear_prediction_guider_min_scale,
            linear_prediction_guider_max_scale = linear_prediction_guider_max_scale,
            triangle_prediction_guider_min_scale = triangle_prediction_guider_min_scale,
            triangle_prediction_guider_max_scale = triangle_prediction_guider_max_scale,
            sampler_s_churn = sampler_s_churn,
            sampler_s_tmin = sampler_s_tmin,
            sampler_s_tmax = sampler_s_tmax,
            sampler_s_noise = sampler_s_noise,
            sampler_eta = sampler_eta,
            sampler_order = sampler_order,
            clip_skip = clip_skip
            )
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


def face_fix(pil_image: Image, **options) -> Image:
    method = options["detection_method"]
    if method == "InsightFace":
        del options["detection_method"]
        return fix_with_insight_face(pil_image, **options)
    elif method == "OpenCV":
        del options["detection_method"]
        return fix_with_opencv(pil_image, **options)
    else:
        raise ValueError("Invalid face detection method")

def fix_with_insight_face(pil_image: Image, **options) -> Image:
    """
    Detects faces in the source image and annotates the image with detected faces.

    Args:
        pil_image (Image): Input image with faces
    Returns:
        Annotated image
    """
    from face_detection.face_detector_insight_face import get_face_bounding_boxes
    logger.info("Using InsightFace for face detection")
    img1 = np.asarray(pil_image.convert("RGB"))
    bboxes = get_face_bounding_boxes(img1)
    return fix_engine(pil_image, bboxes, detection_method="InsightFace", **options)


def fix_with_opencv(pil_image: Image,  **options) -> Image:
    """
    Detects faces in the source image and annotates the image with detected faces.

    Args:
        pil_image (Image): Input image with faces
    Returns:
        Annotated image
    """
    face_data = mark_face_with_opencv(pil_image)

    return fix_engine(pil_image, face_data, **options)


def fix_engine(pil_image: Image,
               face_data,
               detection_method="OpenCV",
               status_queue=None,
               **options) -> Image:
    """
    Detects faces in the source image and annotates the image with detected faces.

    Args:
        pil_image (Image): Input image with faces
    Returns:
        Annotated image in PIL format.
    """
    if face_data is not None:

        # Draw results on the input image
        parse_face_data(face_data, detection_method=detection_method)
        # drawing_area.queue_draw()

        if detection_method == "OpenCV":
            faces = face_data[1]
            if faces is not None:
                for face in faces:
                    pil_image = process_face(pil_image, face, **options)
        elif detection_method == "InsightFace":
            faces = face_data
            if len(faces) > 0:
                for i, face in enumerate(faces):
                    if status_queue:
                        status_queue.put(f"Fixing face {i+1}/{len(faces)}")
                    # left, top, right, bottom to left, top, w, h
                    face = [face[0], face[1], face[2]-face[0], face[3]-face[1]]
                    pil_image = process_face(pil_image, face, **options)

    return pil_image


def mark_face_with_opencv(pil_image):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
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

    original_width = img1.shape[1]
    original_height = img1.shape[0]
    
    max_edge = max(original_height, original_width)
    if max_edge > 768:
        scale = 768 / max_edge
        scale2 = 1 / scale
        scaled = True
    else:
        scale = 1.0
        scaled = False
    img1Width = int(img1.shape[1] * scale)
    img1Height = int(img1.shape[0] * scale)

    img1 = cv.resize(img1, (img1Width, img1Height))
    detector.setInputSize((img1Width, img1Height))
    face_data = detector.detect(img1)

    # rescale coordinates
    if scaled:
        faces = face_data[1]
        for i, face in enumerate(faces):
            faces[i][0] = face[0] * scale2
            faces[i][1] = face[1] * scale2
            faces[i][2] = face[2] * scale2
            faces[i][3] = face[3] * scale2

    return face_data

def mark_face_with_insight_face(pil_image):
    from face_detection.face_detector_insight_face import get_face_bounding_boxes

    # Prepare image for detection
    img1 = np.asarray(pil_image.convert("RGB"))
    bboxes = get_face_bounding_boxes(img1)
    return bboxes

def detect_with_insight_face(pil_image: Image) -> Image:
    """
    Detects faces in the source image and annotates the image with detected faces.

    Args:
        pil_image (Image): Input image with faces
    Returns:
        Annotated image
    """
    from face_detection.face_detector_insight_face import get_face_bounding_boxes

    # Prepare image for detection
    img1 = np.asarray(pil_image.convert("RGB"))
    bboxes = get_face_bounding_boxes(img1)
    if len(bboxes) >= 0:
        # Draw results on the input image
        img1 = annotate_face_insight_face(img1, bboxes)
    return Image.fromarray(img1)


def detect_with_opencv(pil_image: Image) -> Image:
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
