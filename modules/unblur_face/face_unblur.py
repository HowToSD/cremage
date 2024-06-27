"""
Copyright 2024 Hideyuki Inada.  All rights reserved.

License: See third-party license for the license information for the portion of the third-party code used in this file.
"""
import os
import sys
import logging
from typing import Tuple, List

import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT] + sys.path
from unblur_face.cremage_model_v6 import UnblurCremageModelV6
from cremage.utils.model_downloader import download_model_if_not_exist

OPENCV_FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_ROOT, "opencv", "face_detection_yunet_2023mar.onnx")
EXTRA_PADDING = 30
TARGET_SIZE = 256
SOURCE_SIZE = 112 + EXTRA_PADDING * 2
LEFT_PADDING = (112 - 96) / 2 + EXTRA_PADDING
TOP_PADDING = EXTRA_PADDING

REFERENCE_LANDMARKS = [
    ((LEFT_PADDING + 30.2946) * TARGET_SIZE / SOURCE_SIZE, (TOP_PADDING + 51.6963) * TARGET_SIZE / SOURCE_SIZE),  # left eye
    ((LEFT_PADDING + 65.5318) * TARGET_SIZE / SOURCE_SIZE, (TOP_PADDING + 51.5014) * TARGET_SIZE / SOURCE_SIZE),  # right eye
    ((LEFT_PADDING + 48.0252) * TARGET_SIZE / SOURCE_SIZE, (TOP_PADDING + 71.7366) * TARGET_SIZE / SOURCE_SIZE),  # nose tip
    ((LEFT_PADDING + 33.5493) * TARGET_SIZE / SOURCE_SIZE, (TOP_PADDING + 92.3655) * TARGET_SIZE / SOURCE_SIZE),  # left lip corner
    ((LEFT_PADDING + 62.7299) * TARGET_SIZE / SOURCE_SIZE, (TOP_PADDING + 92.2041) * TARGET_SIZE / SOURCE_SIZE)]  # right lip corner


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def infer_unblurred_face(pil_image: Image) -> Image:
    """
    Infers an unblurred face from a 256x256x3 PIL image.

    Args:
        pil_image (Image): Input PIL image of size 256x256.

    Returns:
        Image: Unblurred PIL image.
    """
    # Download the face unblur model if not already downloaded
    model_dir = os.path.join(MODELS_ROOT, "face_unblur")
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)
    model_name = "face_unblur_v11_epoch51.pth"
    model_path = download_model_if_not_exist(
        model_dir,
        "HowToSD/face_unblur",
        model_name,
        )

    w, h = pil_image.size
    if w == 256 and h == 256:
        image = pil_image
    else:
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    image = np.array(image)
    image = torch.tensor(image).float() / 255.0
    image = rearrange(image, "h w c -> c h w")
    x = torch.unsqueeze(image, dim=0).to("cuda")  # add batch axis

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UnblurCremageModelV6()
    sd = torch.load(model_path)
    model.load_state_dict(sd['model_state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        i = 0
        y_hat = model(x)
        y_hat = torch.clamp(y_hat, 0, 1)
        image = transforms.ToPILImage()(y_hat[i].cpu())

    if w != 256 or h != 256:
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    return image

# Not currently used
# def feather_mask(mask, radius=10):
#     kernel_size = 2 * radius + 1
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     feathered_mask = cv.dilate(mask, kernel)
#     feathered_mask = cv.GaussianBlur(feathered_mask, (kernel_size, kernel_size), 0)
#     return feathered_mask / 255.0

def align_face(image: np.ndarray, results: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Aligns faces in the input image based on detected landmarks.

    Args:
        image (np.ndarray): Input image in BGR format.
        results (np.ndarray): Detected landmarks.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: List of aligned face images and their transformation matrices.
    """
    output = image.copy()
    aligned_faces = []
    transform_matrices = []

    for det in results:
        landmarks = det[4:14].astype(np.int32).reshape((5, 2))

        input_landmarks = [
            landmarks[0],  # left eye
            landmarks[1],  # right eye
            landmarks[2],  # nose tip
            landmarks[3],  # left lip corner
            landmarks[4]   # right lip corner
        ]

        input_landmarks = np.array(input_landmarks, dtype=np.float32)
        reference_landmarks = np.array(REFERENCE_LANDMARKS, dtype=np.float32)
        transform_matrix = cv.estimateAffinePartial2D(input_landmarks, reference_landmarks)[0]
        transform_matrices.append(transform_matrix)
        aligned_face = cv.warpAffine(image, transform_matrix, (256, 256))
        aligned_faces.append(aligned_face)

    return aligned_faces, transform_matrices


def mark_face_with_opencv(pil_image: Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects faces in an image using OpenCV.

    Args:
        pil_image (Image): Input PIL image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Image in BGR format and detected faces' landmarks.
    """
    scale = 1.0

    if os.path.exists(OPENCV_FACE_DETECTION_MODEL_PATH) is False:
        print("Model not found")

    detector = cv.FaceDetectorYN.create(
        OPENCV_FACE_DETECTION_MODEL_PATH,  # model
        "",  # config
        (320, 320),  # input size
        0.9,  # score threshold
        0.3,  # nms threshold
        5000  # top_k
    )

    img1 = pil_image.convert("RGB")  # Convert to RGB from RGBA
    img1 = np.asarray(img1, dtype=np.uint8)[:, :, ::-1]  # to np and RGB to BGR
    img1Width = int(img1.shape[1] * scale)
    img1Height = int(img1.shape[0] * scale)

    img1 = cv.resize(img1, (img1Width, img1Height))
    detector.setInputSize((img1Width, img1Height))
    faces = detector.detect(img1)

    return img1, np.array([]) if faces[1] is None else faces[1]


def unblur_face_image(pil_image: Image) -> Image:
    """
    Unblurs faces detected in an image.

    Args:
        pil_image (Image): Input PIL image.

    Returns:
        Image: Output PIL image with unblurred faces.
    """
    # Detect faces in the image
    cv_img, face_data = mark_face_with_opencv(pil_image)

    if face_data.shape == (0,):
        logger.info("No face was found")
        return pil_image
    else:
        aligned_faces, transform_matrices = align_face(cv_img, face_data)
        for i, (img, transform_matrix) in enumerate(zip(aligned_faces, transform_matrices)):
            pil_image = Image.fromarray(img[:,:,::-1])
            sharpened_face = infer_unblurred_face(pil_image)
            sharpened_face = np.asarray(sharpened_face)[:,:,::-1]
            # Compute the inverse transformation matrix
            inverse_transform_matrix = cv.invertAffineTransform(transform_matrix)

            # Warp the grayscale face back to the original position
            warped_face = cv.warpAffine(
                sharpened_face, # source
                inverse_transform_matrix,
                (cv_img.shape[1], cv_img.shape[0]))  # target height, target width

            # Ensure mask and images are of the same type and size
            mask = (warped_face > 0).astype(np.uint8) * 255

            # Erode the mask slightly to cover the 1-pixel border
            kernel = np.ones((5, 5), np.uint8)
            eroded_mask = cv.erode(mask, kernel, iterations=1)

            # Copy the warped grayscale face back into the original image using the eroded mask
            cv_img = cv.bitwise_and(cv_img, cv.bitwise_not(eroded_mask))
            cv_img = cv.bitwise_or(cv_img, cv.bitwise_and(warped_face, eroded_mask))

        cv_img = cv_img[:,:,::-1]
        return Image.fromarray(cv_img)
