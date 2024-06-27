import os
import sys
import unittest

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path

from unblur_face.mha import MultiHeadSelfAttention

class TestMHA(unittest.TestCase):
    def test_mha(self):

        # Example usage
        batch_size = 8
        seq_length = 77
        embed_size = 768
        heads = 8

        x = torch.randn(batch_size, seq_length, embed_size).to("cuda")
        attention = MultiHeadSelfAttention(embed_size, heads).to("cuda")
        out = attention(x, x, x)
        self.assertTrue(out.shape == (8, 77, 768))

if __name__ == '__main__':
    unittest.main()