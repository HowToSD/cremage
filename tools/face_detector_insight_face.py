# Generates aligned face data from single image file.
# It requires insightface open source package and torch.
#  FaceAnalysis loads 5 models.
# Code taken from (licensed under MIT license)
# https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
import os
import sys
import logging
import re
from typing import Optional
from argparse import ArgumentParser

import numpy as np
import cv2
import torch  # This import statement seems to be required to make CUDAExecutionProvider work
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from typing import Any, List
from PIL import Image

assert insightface.__version__>='0.7'

# Duplicated imports, but do not remove until check assessing the impact on CUDA provider
import cv2
import numpy as np 
from skimage import transform as trans

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
RESOURCES_ROOT = os.path.join(PROJECT_ROOT, "resources")

sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Code taken from (licensed under MIT license) start
# https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    """
    Computes the transformation matrix to align source face landmarks to the standard
    arcface reference coordinates.

    Parameters
    ----------
    lmk : np.ndarray
        Source face landmarks as an array of shape (5, 2), where each row corresponds
        to a landmark with x and y coordinates.
    image_size : int, optional
        The size of the image to which the landmarks will be mapped. Must be a multiple
        of 112 or 128. The destination coordinates will be scaled accordingly.
    mode : str, optional
        The method used for transformation. Currently, only 'arcface' is supported.

    Returns:
    -------
    np.ndarray
        A 2x3 affine transformation matrix that maps the input landmarks (lmk) to the
        coordinates defined in arcface_dst.

    Notes
    -----
    The function asserts that the image size must be a multiple of 112 or 128, as the
    scaling ratio depends on these dimensions. The transformation matrix includes
    scaling based on the image size, translation, and possibly rotation and flipping,
    but it maintains the aspect ratio of the landmarks.

    Examples
    --------
    >>> lmk = np.array([[x1, y1], [x2, y2], ..., [x5, y5]])
    >>> transform_matrix = estimate_norm(lmk)
    >>> transformed_image = cv2.warpAffine(src_image, transform_matrix, (image_size, image_size))
    """

    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    # Compute scale ratio
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)   # source to target
    M = tform.params[0:2, :]   # 3x3 mat to 2x3 mat
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    """
    1. Compute the matrix to transform face landmarks to match
       the target landmark positions.
    2. Apply the matrix to transform landmarks.
       End result is the aligned face where landmarks are close to the 
       target positions.
    """
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

# Code taken from (licensed under MIT license) end
# https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py


def get_aligned_faces(
        rgb_cv_image: np.ndarray) -> np.ndarray:
    """
    Extracts faces from an image and aligns each face.

    Args:
        rgb_cv_image (np.ndarray): RGB-converted cv image. Note default is BGR so
                                   conversion has to be done before calling this method.
    Returns:
    -------
      Aligned face images in ndarray in [number_of_faces, 256, 256, 3].
      If a face is not found, it will return an empty array (0, ).
    """
    face_analyzer = FaceAnalysis(name='buffalo_l',
                       providers=['CUDAExecutionProvider'])
    
    # ctx_id=0 parameter specifies that the first GPU device should be used
    # size of the image input for the face detector; in this case, it's set to 640x640 pixels.
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))  

    source_faces = face_analyzer.get(rgb_cv_image)  
    images = list()
    if len(source_faces) == 0:
        logger.warn(f"No face found")
        return np.asarray(images)
    for face in source_faces:
        print(face)
        print(face.kps)
        aligned_face_img = norm_crop(rgb_cv_image, face.kps, image_size=256, mode='arcface')        
        images.append(aligned_face_img)

    faces = np.asarray(images)
    return faces


def get_face_bounding_boxes(
        rgb_cv_image: np.ndarray) -> List[np.ndarray]:
    """
    Extracts faces from an image and detects a bounding box for each face.

    Args:
        rgb_cv_image (np.ndarray): RGB-converted cv image. Note default is BGR so
                                   conversion has to be done before calling this method.
    Returns:
    -------
        List of bounding boxes for each face found in the input image.
        Bounding box is an numpy ndarray which contains left, top, right, bottom of
        a face.
    """
    face_analyzer = FaceAnalysis(name='buffalo_l',
                       providers=['CUDAExecutionProvider'])
    
    # ctx_id=0 parameter specifies that the first GPU device should be used
    # size of the image input for the face detector; in this case, it's set to 640x640 pixels.
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))  

    source_faces = face_analyzer.get(rgb_cv_image)  
    bbox_list = [f.bbox for f in source_faces]    
    return bbox_list

def main():
    import time

    input_file_path = os.path.join(RESOURCES_ROOT, "images", "real1.jpg")
    if(os.path.exists(input_file_path) is False):
        raise ValueError(f"{input_file_path} does not exist")
    
    source_img = cv2.imread(input_file_path)
    source_img = source_img[:,:,::-1]  # bgr to rgb

    bboxes = get_face_bounding_boxes(source_img)
    print(bboxes)


if __name__ == "__main__":
    main()