import os
import sys

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from ldm.models.diffusion.ddim import DDIMSampler
from cldm.ddim_hacked import DDIMSampler as DDIMControlNetSampler
from ldm.models.diffusion.k_diffusion_samplers import EulerSampler
from ldm.models.diffusion.k_diffusion_samplers import EulerAncestralSampler
from ldm.models.diffusion.k_diffusion_samplers import HeunSampler
from ldm.models.diffusion.k_diffusion_samplers import Dpm2Sampler
from ldm.models.diffusion.k_diffusion_samplers import Dpm2AncestralSampler
from ldm.models.diffusion.k_diffusion_samplers import LmsSampler
from ldm.models.diffusion.k_diffusion_samplers import Dpmpp2sAncestralSampler
from ldm.models.diffusion.k_diffusion_samplers import DpmppSdeSampler
from ldm.models.diffusion.k_diffusion_samplers import Dpmpp2mSampler
from ldm.models.diffusion.k_diffusion_samplers import Dpmpp2mSdeSampler
from ldm.models.diffusion.k_diffusion_samplers import Dpmpp3mSdeSampler


sampler_name_list = [
    "DDIM",
    "Euler",
    "Euler A",
    "Heun",
    "DPM2",
    "DPM2 A",
    "LMS",
    "DPM++ 2S A",
    "DPM++ SDE",
    "DPM++ 2M",
    "DPM++ 2M SDE",
    "DPM++ 3M SDE"
]

def instantiate_sampler(sampler_name:str, model, use_control_net: bool):
    if sampler_name == "DDIM":
        if use_control_net:
            sampler = DDIMControlNetSampler(model)
        else:
            sampler = DDIMSampler(model)
    elif sampler_name == "Euler":
        sampler = EulerSampler(model)
    elif sampler_name == "Euler A":
        sampler = EulerAncestralSampler(model)
    elif sampler_name == "Heun":
        sampler = HeunSampler(model)
    elif sampler_name == "DPM2":
        sampler = Dpm2Sampler(model)
    elif sampler_name == "DPM2 A":
        sampler = Dpm2AncestralSampler(model)
    elif sampler_name == "LMS":
        sampler = LmsSampler(model)
    elif sampler_name == "DPM++ 2S A":
        sampler = Dpmpp2sAncestralSampler(model)
    elif sampler_name == "DPM++ SDE":
        sampler = DpmppSdeSampler(model)
    elif sampler_name == "DPM++ 2M":
        sampler = Dpmpp2mSampler(model)
    elif sampler_name == "DPM++ 2M SDE":
        sampler = Dpmpp2mSdeSampler(model)
    elif sampler_name == "DPM++ 3M SDE":
        sampler = Dpmpp3mSdeSampler(model)
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")
    
    return sampler
