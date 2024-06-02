import os
import sys
import json
import unittest
import tempfile

import torch
import numpy as np
import PIL
from PIL import Image
from imwatermark import WatermarkEncoder

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.image_utils import display_pil_image_from_mask_pil_image
from cremage.utils.image_utils import resize_with_padding, bbox_for_multiple_of_64
from cremage.utils.image_utils import save_torch_tensor_as_image
from cremage.utils.image_utils import save_torch_tensor_as_image_with_watermark

class TestImageUtil(unittest.TestCase):
    def test_save_torch_tensor_as_image_with_watermark(self):

        wm = "Cremage"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        img = torch.rand((3, 768, 512)).float()
        img = img.to("cuda")
        with tempfile.TemporaryDirectory() as temp_test_dir:
            files = os.listdir(temp_test_dir)
            meta = {"h": 768, "w": 512}
            img2 = save_torch_tensor_as_image_with_watermark(
                temp_test_dir,
                img,
                meta,
                file_number=0,
                wm_encoder=wm_encoder)
            files2 = os.listdir(temp_test_dir)
            self.assertTrue(len(files2) - len(files) == 1)

    def test_save_torch_tensor_as_image(self):
        img = torch.rand((3, 768, 512)).float()
        img = img.to("cuda")
        with tempfile.TemporaryDirectory() as temp_test_dir:
            temp_test_file = tempfile.NamedTemporaryFile(
                dir=temp_test_dir, suffix=".png")
            meta = {"h": 768, "w": 512}
            img2 = save_torch_tensor_as_image(
                temp_test_file.name,
                img,
                meta)
    
            self.assertTrue(os.path.exists(temp_test_file.name) is True)
            img3 = Image.open(temp_test_file.name)

            for img in [img2, img3]:
                w, h = img.size
                self.assertTrue(w == 512 and h == 768)
                self.assertTrue("generation_data" in img.info)
                generation_data = json.loads(img.info["generation_data"])
                self.assertTrue(generation_data["h"] == 768)
                self.assertTrue(generation_data["w"] == 512)

        assert(os.path.exists(temp_test_file.name) is False)

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
