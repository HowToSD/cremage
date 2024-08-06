"""
Util test

Copyright (c) 2024 Hideyuki Inada. All rights reserved.
"""
import os
import sys
import logging
from typing import Dict, Union
import unittest

import torch
from omegaconf import ListConfig, OmegaConf


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
SDXL_MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules", "sdxl") 
sys.path = [SDXL_MODULE_ROOT, MODULE_ROOT] + sys.path

from sgm.util import append_dims

class TestUtil(unittest.TestCase):

    def test_append_dims(self):
        a = torch.tensor([1.0])
        print(a)
        b = append_dims(a, 4)  # tensor([[[[1.]]]]), shape: torch.Size([1, 1, 1, 1])
        self.assertTrue(b.dim() == 4)


if __name__ == '__main__':
    unittest.main()