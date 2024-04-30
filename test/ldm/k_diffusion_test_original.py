# Cremage note: FIXME. Currently test cases broken maybe due to the refactoring of other code.
# FIXME
import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..") 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
LDM_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs/ldm/configs")
sys.path = [MODULE_ROOT] + sys.path
from k_diffusion.external import CompVisDenoiser
from ldm.models.diffusion.ldm_wrapper_for_k_diffusion import LDMWrapperForKDiffusion
from ldm.models.diffusion.k_diffusion_samplers import EulerAncestralSampler

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


class TestKDiffusion(unittest.TestCase):
    def test_sigmas(self):
        current_config = OmegaConf.load(f"{LDM_CONFIG_DIR}/stable-diffusion/v1-inference.yaml")
        current_config = current_config.model
        current_config["params"]["cond_stage_config"]["params"]["lora_ranks"] = []
        current_config["params"]["cond_stage_config"]["params"]["lora_weights"] = []
        ldm_model = instantiate_from_config(current_config).to("cuda")

        c = torch.rand((2, 77, 768)).to("cuda").half()  # Positive text embedding
        uc = torch.rand((2, 77, 768)).to("cuda").half()  # Negative text embedding
        cfg = 7.5
        ldm_wrapper_model = LDMWrapperForKDiffusion(ldm_model, c, uc, cfg)
        denoiser = CompVisDenoiser(ldm_wrapper_model, False).to("cuda")  # quantize
        actual = denoiser.get_sigmas(5).cpu()  # returns n + 1, so if n = 5, returns 6 sigmas.
        expected = torch.tensor([14.6146, 4.0861, 1.6156, 0.6951, 0.0292, 0.0000])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-4))

    def test_k_diffusion_sampler(self):
        current_config = OmegaConf.load(f"{LDM_CONFIG_DIR}/stable-diffusion/v1-inference.yaml")
        current_config = current_config.model
        current_config["params"]["cond_stage_config"]["params"]["lora_ranks"] = []
        current_config["params"]["cond_stage_config"]["params"]["lora_weights"] = []
        ldm_model = instantiate_from_config(current_config).to("cuda")
        c = torch.rand((2, 77, 768)).to("cuda").half()  # Positive text embedding
        uc = torch.rand((2, 77, 768)).to("cuda").half()  # Negative text embedding
        cfg = 7.5
        sampler = EulerAncestralSampler(ldm_model)
        x, _ = sampler.sample(20, 
                    batch_size=2,
                    shape=[4, 64, 96],
                    conditioning=c,
                    unconditional_conditioning=uc,
                    unconditional_guidance_scale=cfg)
        self.assertTrue(x.shape == (2, 4, 64, 96))
    
    def test_k_diffusion_add_noise(self):
        current_config = OmegaConf.load(f"{LDM_CONFIG_DIR}/stable-diffusion/v1-inference.yaml")
        current_config = current_config.model
        current_config["params"]["cond_stage_config"]["params"]["lora_ranks"] = []
        current_config["params"]["cond_stage_config"]["params"]["lora_weights"] = []
        ldm_model = instantiate_from_config(current_config).to("cuda")
        c = torch.rand((2, 77, 768)).to("cuda").half()  # Positive text embedding
        uc = torch.rand((2, 77, 768)).to("cuda").half()  # Negative text embedding
        cfg = 7.5
        sampler = EulerAncestralSampler(ldm_model)
        x0 = torch.rand((2, 4, 64, 64)).to("cuda")
        t = torch.tensor([10, 10]).to("cuda")  # Simulate denoising ratio = 0.5
        noisy_image = sampler.stochastic_encode(
            x0,
            t,
            sampling_steps=20)
        self.assertTrue(noisy_image.shape == (2, 4, 64, 64))

    def test_k_diffusion_sampler_img2img(self):
        current_config = OmegaConf.load(f"{LDM_CONFIG_DIR}/stable-diffusion/v1-inference.yaml")
        current_config = current_config.model
        current_config["params"]["cond_stage_config"]["params"]["lora_ranks"] = []
        current_config["params"]["cond_stage_config"]["params"]["lora_weights"] = []
        ldm_model = instantiate_from_config(current_config).to("cuda")
        c = torch.rand((2, 77, 768)).to("cuda").half()  # Positive text embedding
        uc = torch.rand((2, 77, 768)).to("cuda").half()  # Negative text embedding
        cfg = 7.5
        x0 = torch.rand((2, 4, 64, 96)).to("cuda")
        sampler = EulerAncestralSampler(ldm_model)
        x, _ = sampler.sample(20,
                    batch_size=2,
                    shape=[4, 64, 96],
                    conditioning=c,
                    unconditional_conditioning=uc,
                    unconditional_guidance_scale=cfg,
                    x0=x0,
                    denoising_steps=1)  # minimum
        self.assertTrue(x.shape == (2, 4, 64, 96))

        x, _ = sampler.sample(20,
                    batch_size=2,
                    shape=[4, 64, 96],
                    conditioning=c,
                    unconditional_conditioning=uc,
                    unconditional_guidance_scale=cfg,
                    x0=x0,
                    denoising_steps=8)
        self.assertTrue(x.shape == (2, 4, 64, 96))

        x, _ = sampler.sample(20,
                    batch_size=2,
                    shape=[4, 64, 96],
                    conditioning=c,
                    unconditional_conditioning=uc,
                    unconditional_guidance_scale=cfg,
                    x0=x0,
                    denoising_steps=20) # maximum
        self.assertTrue(x.shape == (2, 4, 64, 96))


if __name__ == '__main__':
    unittest.main()
