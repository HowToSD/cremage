import os
import sys
import unittest

import torch
import numpy as np
import cv2
import PIL

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.ml_utils import scale_pytorch_images


class TestMlUtil(unittest.TestCase):
    def test_scale_pytorch_images(self):
        images = torch.rand((8, 3, 768, 512)).to("cuda")  # float32, b, c, h, w
        images = scale_pytorch_images(images,
                                      width=1024,
                                      height=1536,
                                      interpolation=cv2.INTER_LANCZOS4)

        self.assertTrue(images.shape == (8, 3, 1536, 1024))


if __name__ == '__main__':
    unittest.main()
