"""
Taken from the runwayml github repo. See LICENSE at the root of the project for license information.
"""
import numpy as np
from PIL import Image
from imwatermark import WatermarkEncoder
import cv2

wm = "StableDiffusionV1-Inpainting"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))


def put_watermark(img):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

