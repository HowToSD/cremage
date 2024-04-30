import os
import sys
import unittest

import numpy as np
import PIL
from PIL import Image

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..") 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.image_utils import display_pil_image_from_mask_pil_image
from cremage.utils.image_utils import resize_with_padding, bbox_for_multiple_of_64


class TestImageUtil(unittest.TestCase):
    def test_display_from_mask(self):
        
        img = np.array([
            [[0, 0, 0], [0, 130, 0]],  # black, white
            [[0, 1, 0], [255, 255, 255]]   # black, white
        ], dtype=np.uint8)

        img = PIL.Image.fromarray(img)
        img = display_pil_image_from_mask_pil_image(img)
        actual = np.asarray(img)
        
        expected = np.array([
            [[0, 0, 0, 0], [255, 255, 255, 255]],
            [[0, 0, 0, 0], [255, 255, 255, 255]]], dtype=np.uint8)
        
        self.assertTrue(np.allclose(actual, expected))
  
    def test_display_from_mask_alpha(self):
        
        img = np.array([
            [[0, 0, 0, 0], [255, 255, 255, 255]],  # black transparent, white opaque
            [[0, 0, 0, 255], [255, 255, 255, 0]]   # black opaque, white transparent
        ], dtype=np.uint8)

        img = PIL.Image.fromarray(img)
        img = display_pil_image_from_mask_pil_image(img)
        actual = np.asarray(img)    
        expected = np.array([
            [[0, 0, 0, 0], [255, 255, 255, 255]],
            [[0, 0, 0, 0], [255, 255, 255, 255]]], dtype=np.uint8)  # black transparent

        self.assertTrue(np.allclose(actual, expected))
  
    def test_resize_with_padding(self):
        h = 240
        w = 120
        image = (np.random.rand(h, w, 3)*255).astype(np.uint8)
        image = Image.fromarray(image)
        scaled = resize_with_padding(image, target_width=768, target_height=120)
        self.assertTrue(scaled.size == (768, 120))

    def test_resize_with_padding_odd(self):
        h = 241
        w = 120
        image = (np.random.rand(h, w, 3)*255).astype(np.uint8)
        image = Image.fromarray(image)
        scaled = resize_with_padding(image, target_width=768, target_height=120)
        self.assertTrue(scaled.size == (768, 120))

    def test_resize_with_padding_odd2(self):
        h = 240
        w = 121
        image = (np.random.rand(h, w, 3)*255).astype(np.uint8)
        image = Image.fromarray(image)
        scaled = resize_with_padding(image, target_width=768, target_height=120)
        self.assertTrue(scaled.size == (768, 120))

    def test_resize_with_padding_odd3(self):
        h = 240
        w = 121
        image = (np.random.rand(h, w, 3)*255).astype(np.uint8)
        image = Image.fromarray(image)
        scaled = resize_with_padding(image, target_width=125, target_height=240)
        self.assertTrue(scaled.size == (125, 240))

    def test_resize_w_with_padding_bbox(self):
        h = 240
        w = 121
        image = (np.random.rand(h, w, 3)*255).astype(np.uint8)
        image = Image.fromarray(image)
        scaled, bbox = resize_with_padding(image, target_width=125, target_height=240, return_bbox=True)
        self.assertTrue(scaled.size == (125, 240))
        self.assertTrue(bbox == (2, 0, 2 + w, 0 + h))

    def test_resize_w_with_padding_bbox_odd(self):
        h = 240
        w = 120
        image = (np.random.rand(h, w, 3)*255).astype(np.uint8)
        image = Image.fromarray(image)
        scaled, bbox = resize_with_padding(image, target_width=125, target_height=240, return_bbox=True)
        self.assertTrue(scaled.size == (125, 240))
        # 3 padding pixels get assigned to the left x out of 5 width padding pixels
        self.assertTrue(bbox == (3, 0, 3 + w, 0 + h))

    def test_resize_h_with_padding_bbox(self):
        h = 236
        w = 125
        image = (np.random.rand(h, w, 3)*255).astype(np.uint8)
        image = Image.fromarray(image)
        scaled, bbox = resize_with_padding(image, target_width=125, target_height=240, return_bbox=True)
        self.assertTrue(scaled.size == (125, 240))
        self.assertTrue(bbox == (0, 2, 0 + w, 2 + h))

    def test_resize_h_with_padding_bbox_odd(self):
        h = 235
        w = 125
        image = (np.random.rand(h, w, 3)*255).astype(np.uint8)
        image = Image.fromarray(image)
        scaled, bbox = resize_with_padding(image, target_width=125, target_height=240, return_bbox=True)
        self.assertTrue(scaled.size == (125, 240))
        # 3 padding pixels get assigned to the top h out of 5 height padding pixels
        self.assertTrue(bbox == (0, 3, 0 + w, 3 + h))

    def test_bbox_for_multiple_of_64_1(self):
        self.assertTrue(bbox_for_multiple_of_64(0, 0) == (0, 0))
        self.assertTrue(bbox_for_multiple_of_64(0, 1) == (0, 64))
        self.assertTrue(bbox_for_multiple_of_64(1, 0) == (64, 0))
        self.assertTrue(bbox_for_multiple_of_64(1, 1) == (64, 64))
        self.assertTrue(bbox_for_multiple_of_64(63, 63) == (64, 64))
        self.assertTrue(bbox_for_multiple_of_64(64, 64) == (64, 64))
        self.assertTrue(bbox_for_multiple_of_64(65, 65) == (128, 128))

if __name__ == '__main__':
    unittest.main()
