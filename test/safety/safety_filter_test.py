import os
import sys
import json
import unittest
import tempfile

import torch
import numpy as np
import PIL
from PIL import Image
from einops import rearrange

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path

from safety.safety_filter import SafetyFilter

TEST_IMAGE_1_PATH = os.path.realpath(os.path.join(PROJECT_ROOT, "..", "cremage_resources", "safety_filter_test_1.png"))
TEST_IMAGE_2_PATH = os.path.realpath(os.path.join(PROJECT_ROOT, "..", "cremage_resources", "safety_filter_test_2.png"))

def cosine_similarity_for_images(img_1: np.ndarray, img_2: np.ndarray) -> float:
        """
        Compute cosine similarity of two sets of images.

        Args:
            img_1 (np.ndarray): Images in C, H, W format. Values are in [0, 1]
            img_2 (np.ndarray): Images in C, H, W format. Values are in [0, 1]
        Returns:
            Cosine similarity of two sets of images.
        """
        # Flatten the images
        input_vector = img_1.flatten()
        assert len(input_vector.shape) == 1
        output_vector = img_2.flatten()
        # Normalize the vectors
        input_norm = np.linalg.norm(input_vector)
        output_norm = np.linalg.norm(output_vector)
        if input_norm == 0 or output_norm == 0:
            raise ValueError("One of the imags yielded a zero vector.")
        normalized_input = input_vector / input_norm
        normalized_output = output_vector / output_norm
        cosine_similarity = np.dot(normalized_input, normalized_output)
        return cosine_similarity

class TestSafetyFilter(unittest.TestCase):

    def test_safety_filter_numpy_1(self):
        sf = SafetyFilter()
        img = Image.open(TEST_IMAGE_1_PATH)
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, 0)
        checked_imgs, concepts = sf(numpy_images=img)
        self.assertTrue(img.shape == checked_imgs.shape)
        self.assertFalse(concepts[0])
        img = np.squeeze(img)
        checked_imgs = np.squeeze(checked_imgs)
        cosine_similarity = cosine_similarity_for_images(img, checked_imgs)
        self.assertTrue(cosine_similarity > 0.999)

    def test_safety_filter_numpy_2(self):
        sf = SafetyFilter()
        img = Image.open(TEST_IMAGE_2_PATH)
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, 0)
        checked_imgs, concepts = sf(numpy_images=img)
        self.assertTrue(img.shape == checked_imgs.shape)
        self.assertTrue(concepts[0])
        checked_imgs = np.squeeze(checked_imgs)
        cosine_similarity = cosine_similarity_for_images(img, checked_imgs)
        self.assertTrue(cosine_similarity < 0.8, f"Cosine similarity was {cosine_similarity}")


    def test_safety_filter_torch_1(self):
        sf = SafetyFilter()
        img = Image.open(TEST_IMAGE_1_PATH)
        img = np.asarray(img) / 255.0
        img_np = np.expand_dims(img, 0)
        img_bhwc = torch.tensor(img_np)
        img_bchw = rearrange(img_bhwc, "b h w c -> b c h w")
        checked_imgs, concepts = sf(torch_images=img_bchw)
        self.assertTrue(img_bchw.shape == checked_imgs.shape, f"checked_imgs.shape is {checked_imgs.shape}")
        self.assertFalse(concepts[0])
        checked_imgs = rearrange(checked_imgs, "b c h w -> b h w c").detach().cpu().numpy()
        img = np.squeeze(img)
        checked_imgs = np.squeeze(checked_imgs)
        cosine_similarity = cosine_similarity_for_images(img, checked_imgs)
        self.assertTrue(cosine_similarity > 0.999)

    def test_safety_filter_torch_1(self):
        sf = SafetyFilter()
        img = Image.open(TEST_IMAGE_2_PATH)
        img = np.asarray(img) / 255.0
        img_np = np.expand_dims(img, 0)
        img_bhwc = torch.tensor(img_np)
        img_bchw = rearrange(img_bhwc, "b h w c -> b c h w")
        checked_imgs, concepts = sf(torch_images=img_bchw)
        self.assertTrue(img_bchw.shape == checked_imgs.shape, f"checked_imgs.shape is {checked_imgs.shape}")
        self.assertTrue(concepts[0])
        checked_imgs = rearrange(checked_imgs, "b c h w -> b h w c").detach().cpu().numpy()
        img = np.squeeze(img)
        checked_imgs = np.squeeze(checked_imgs)
        cosine_similarity = cosine_similarity_for_images(img, checked_imgs)
        self.assertTrue(cosine_similarity < 0.8)





if __name__ == '__main__':
    unittest.main()
