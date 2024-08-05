"""
Guiders test

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

from sgm.modules.diffusionmodules.guiders import VanillaCFG

class TestGuider(unittest.TestCase):

    def test_torch_chunk(self):
        # Split on batch-axis 8 -> 4 and 4.
        # Reverse of concat
        x = torch.rand((8, 4, 64, 64))
        u, c = x.chunk(2)
        self.assertTrue(u.shape == (4, 4, 64, 64))
        self.assertTrue(c.shape == (4, 4, 64, 64))

    def test_vanilla_instantiation(self):
        x = torch.rand((2, 4, 64, 64))
        g = VanillaCFG(scale=7.5)
        self.assertTrue(g.scale == 7.5)

    def test_prepare_inputs(self):
        """
        Test merging uc and c tensors.
        Merging is done by concanating the tensors with the same key in uc and c.
        Result is a dict with the same keys, but concatenated values.
        """
        x = torch.rand((2, 4, 64, 64))  # noisy image
        uc = dict()
        c = dict()
        uc["crossattn"] = torch.rand((2, 77, 768))  # negative prompt embedding
        c["crossattn"] = torch.rand((2, 77, 768))  # positive prompt embedding
        g = VanillaCFG(scale=7.5)
        s = torch.tensor([0.1, 0.2])
        x_out, s_out, c_out = g.prepare_inputs(x, s, c, uc)
        self.assertTrue(x_out.shape == (4, 4, 64, 64))
        self.assertTrue(s_out.shape == (4,))
        self.assertTrue(c_out["crossattn"].shape == (4, 77, 768))

    def test_call(self):
        """
        Test merging uc and c tensors.
        Merging is done by concanating the tensors with the same key in uc and c.
        Result is a dict with the same keys, but concatenated values.
        """
        bs = 4
        x = torch.rand((bs, 4, 64, 64))  # noisy image
        g = VanillaCFG(scale=7.5)
        s = torch.tensor([0.1, 0.2])
        x_out = g(x, s)
        half_bs = bs / 2
        self.assertTrue(x_out.shape == (half_bs, 4, 64, 64))


    def test_prepare_and_call(self):
        """
        Test combined execution of prepare and call.
        """
        bs = 2
        x = torch.rand((bs, 4, 64, 64))  # noisy image
        uc = dict()
        c = dict()
        uc["crossattn"] = torch.rand((bs, 77, 768))  # negative prompt embedding
        c["crossattn"] = torch.rand((bs, 77, 768))  # positive prompt embedding
        g = VanillaCFG(scale=7.5)
        s = torch.tensor([0.1, 0.2])
        x, s, c_out = g.prepare_inputs(x, s, c, uc)  # double batch size
        def dummy_model(x, s):
            return x
        x = dummy_model(x, s)
        x_out = g(x, s)  # halve batch size
        self.assertTrue(x_out.shape == (bs, 4, 64, 64))


if __name__ == '__main__':
    unittest.main()