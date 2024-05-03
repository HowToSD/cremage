import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
LDM_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs/ldm/configs")
sys.path = [MODULE_ROOT] + sys.path
from k_diffusion.external import CompVisDenoiser
from k_diffusion.sampling import get_sigmas_karras

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


class TestKDiffusion(unittest.TestCase):
    def test_get_sigmas_karras(self):
        n = 20
        sigma_min = 0.0316
        sigma_max = 0.9976
        sigmas = get_sigmas_karras(n, sigma_min, sigma_max)
        print(sigmas)

    def test_get_sigmas_karras2(self):
        n = 20
        sigma_min = 0.1
        sigma_max = 10.
        sigmas = get_sigmas_karras(n, sigma_min, sigma_max)
        print(sigmas)

if __name__ == '__main__':
    unittest.main()
