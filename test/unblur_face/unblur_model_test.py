import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from unblur_face.cremage_model_v6 import UnblurCremageModelV6

class TestUnblurFaceModel(unittest.TestCase):
    def test_model(self):
        # Instantiate the model
        model = UnblurCremageModelV6().to("cuda")

        # Define input tensor with appropriate dimensions
        input_tensor = torch.randn(1, 3, 256, 256).to("cuda")

        # Perform forward pass
        out = model(input_tensor).to("cuda")

        self.assertTrue(out.shape == (1, 3, 256, 256))


if __name__ == '__main__':
    unittest.main()