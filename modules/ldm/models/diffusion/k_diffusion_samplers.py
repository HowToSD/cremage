""""
Note on the structure of how K-diffusion code is integrated into Cremage.

K-diffusion module contains various sampling methods.
Each sampling method accepts the model argument, and internally calls the model to
predict the noise.
However, the model argument is not the LatentDiffusionModel that is used
in Cremage or CompViz. Instead, it expects CompVisDenoiser.
CompVisDenoiser is designed to wrap another model. So here is the hierarchy:

Custom sampler wraps:
    CompVisDenoiser wraps:
        LDMWrapperForKDiffusion wraps:
            LatentDiffusionModel

For each sampling step, LDMWrapperForKDiffusion's forward method is called by CompVisDenoiser.
So you need to map this to LDM's apply model method.

In addition, K-diffusion contains code to do the following:
1) Compute sigmas from the model's alphas. Sigmas are the noise level used during
   reverse diffusion.
2) Convert sigmas to t so that we can pass t to our LDM model.
3) Compute sigmas for fewer sampling steps. For example, DDPM contains 1000 steps.
   But Euler-Ancestral typically contains much fewer steps like 20. Instead of you
   computing the right t or sigmas, K-diffusion does this, so you do not need to compute
   the interpolated values yourself.
"""
import os
import sys
from functools import partial
import logging
import re
from typing import List, Tuple
import contextlib

import torch
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path = [MODULE_ROOT] + sys.path

from k_diffusion.external import DiscreteEpsDDPMDenoiser, CompVisDenoiser
from k_diffusion.sampling import get_sigmas_karras
from k_diffusion.sampling import get_sigmas_vp
from k_diffusion.sampling import sample_euler
from k_diffusion.sampling import sample_euler_ancestral
from k_diffusion.sampling import sample_heun
from k_diffusion.sampling import sample_dpm_2
from k_diffusion.sampling import sample_dpm_2_ancestral
from k_diffusion.sampling import sample_lms
from k_diffusion.sampling import sample_dpmpp_2s_ancestral
from k_diffusion.sampling import sample_dpmpp_sde
from k_diffusion.sampling import sample_dpmpp_2m
from k_diffusion.sampling import sample_dpmpp_2m_sde
from k_diffusion.sampling import sample_dpmpp_3m_sde
from ldm.models.diffusion.ldm_wrapper_for_k_diffusion import LDMWrapperForKDiffusion
from ldm.modules.diffusionmodules.util import extract_into_tensor

class KDiffusionSamplerBase(object):
    def __init__(self,
                 model,
                 sigma_min = 0.0316386,
                 sigma_max = 14.5521805,
                 beta_d=19.9,
                 beta_min=0.1,
                 eps_s=1e-3):
        self.ldm_model = model
        self.ddpm_num_timesteps = model.num_timesteps  # e.g. 1000
        alphas_cumprod = self.ldm_model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        # How these were computed
        # DiscreteEpsDDPMDenoiser class from K-diffusion contains the following formula:
        # (1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        # An SD model I tried has 
        # The first value in alphas_cumprod was 0.9990
        # The last value alphas_cumprod was 0.0047
        # Applying the formula, you will get:
        # 14.5521805
        # 0.0316386, 
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # For sigma_vp. Values are from get_sigmas_vp method's default values
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.eps_s = eps_s
        
        self.device = os.environ.get("GPU_DEVICE", "cpu")

        # This is var.to("cuda") or var.cpu(), but ensure that the device
        # is the same as ldm_model
        if torch.cuda.is_available():
            to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.ldm_model.device)
        else:
            to_torch = lambda x: x.clone().detach().to(torch.float16).to(self.ldm_model.device)

        # Copy LDM model's attributes to this class
        # Note that are 7 alpha-related attributes
        self.register_buffer('betas', to_torch(self.ldm_model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.ldm_model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if torch.cuda.is_available():
                if attr.device != torch.device("cuda"):
                    attr = attr.to(torch.device("cuda"))
            else:
                if attr.device != torch.device(os.environ.get("GPU_DEVICE", "cpu")):
                    attr = attr.half().to(os.environ.get("GPU_DEVICE", "cpu"))
        setattr(self, name, attr)

    @torch.no_grad()
    def compute_sigmas(self,
                   n:int):
        """
        Compute sigma schedule.

        Args:
            n (int): Number of sampling steps
        """
        return None
    
    @torch.no_grad()
    def _sample_common_prep(self,
               S,  # Number of denoising steps (e.g. 20, 30, 40, 50)
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        This method is called at the beginning of the sample method.

        TODO: Remove unused parameters and add docstring.
        """
        C, H, W = shape
        size = (batch_size, C, H, W)
        if x0 is None:
            self.x = torch.randn(size, device=os.environ.get("GPU_DEVICE", "cpu"))
        else:
            self.x = x0

        # Instantiate the top-level model that is called by each sampler
        # Note that this model takes x and sigma (and not x and t)
        # This is level 2
        self.compviz_wrapper_model = CompVisDenoiser(self.ldm_model, False).to(os.environ.get("GPU_DEVICE", "cpu"))  # quantize       

        # Top level wrapper model that deals with sampler
        self.ldm_wrapper_model = LDMWrapperForKDiffusion(self.compviz_wrapper_model,
                                           conditioning,
                                           unconditional_conditioning,
                                           unconditional_guidance_scale)

        # Compute sigmas (noise level used in denoising)
        # This may be overridden by a sampler
        self.sigmas = self.compute_sigmas(S)  # Compute sigmas
        
        if "denoising_steps" in kwargs:  # partial denoising (img2img)
            t = kwargs["denoising_steps"]
            # Pick last t+1 sigmas
            self.sigmas = self.sigmas[-(t+1):]
            logger.debug("")
            print(f"t:{t}")
            assert self.sigmas.shape[0] == t + 1

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):

        self._sample_common_prep(
               S=S,
               batch_size=batch_size,
               shape=shape,
               conditioning=conditioning,
               callback=callback,
               normals_sequence=normals_sequence,
               img_callback=img_callback,
               quantize_x0=quantize_x0,
               eta=eta,
               mask=mask,
               x0=x0,
               temperature=temperature,
               noise_dropout=noise_dropout,
               score_corrector=score_corrector,
               corrector_kwargs=corrector_kwargs,
               verbose=verbose,
               x_T=x_T,
               log_every_t=log_every_t,
               unconditional_guidance_scale=unconditional_guidance_scale,
               unconditional_conditioning=unconditional_conditioning,
               **kwargs,
        )

        if torch.cuda.is_available():
            precision_context = torch.autocast(device_type=os.environ.get("GPU_DEVICE", "cpu"))
        else:
            precision_context = contextlib.nullcontext()
        with precision_context:
            x, aux = self.do_sample()

        return x, aux
    
    @torch.no_grad()
    def do_sample(self):
        return self.x, None

    @torch.no_grad()
    def stochastic_encode(self, x0, t, sampling_steps, noise=None):
        """
        Add noise to the image using the forward diffusion algorithm.
        The formula is
        x_t = sqrt(cum_prod_alpha_t) * x_0 + sqrt(1 - cum_prod_alpha_t) * N

        Args:
            x0: Original image
            t: Time step to specify to control the amount of noise to add.
               This is used as an index to a cumprod alpha array.
               t is the index in the sampler's time steps [0, n] to specify
               the number of steps to add noise.
               For example, if Euler Ancestral is used for 20 steps of sampling,
               t needs to point to a number in [0, 19].
            sampling_steps (int): Number of steps used for full denoising for the sampler.

        Note:
        This method is adopted from modules/ldm/models/diffusion/ddim.py.
        Internally, t is mapped to an index in DDPM timestep [0, 999].
        TODO: Depending on the image to image denoiser which is expected to be used
        after this method, there may be a mismatch between the noise added
        using this method and the noise the denoiser is expected to remove.
        For example, alpha used in this method assumes that the timestep
        intervals are evenly spaced, but there is no guarantee for that.
        In addition, alpha is picked from discretized 1000 values instead
        of interpolated value, which may result in a value that is slightly off.
        """
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod

        if noise is None:
            noise = torch.randn_like(x0)

        t = (t * 1000.0 / sampling_steps).long()

        # Apply forward diffusion formula to add noise
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

class EulerSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return self.compviz_wrapper_model.get_sigmas(n).to(self.device)

    @torch.no_grad()
    def do_sample(self):
        x = sample_euler(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class EulerAncestralSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return self.compviz_wrapper_model.get_sigmas(n).to(self.device)

    @torch.no_grad()
    def do_sample(self):
        x = sample_euler_ancestral(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class HeunSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return self.compviz_wrapper_model.get_sigmas(n).to(self.device)

    @torch.no_grad()
    def do_sample(self):
        x = sample_heun(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class Dpm2Sampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return get_sigmas_karras(n, self.sigma_min, self.sigma_max, device=self.device)    
    @torch.no_grad()
    def do_sample(self):
        x = sample_dpm_2(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class Dpm2AncestralSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return get_sigmas_karras(n, self.sigma_min, self.sigma_max, device=self.device)
    @torch.no_grad()
    def do_sample(self):
        x = sample_dpm_2_ancestral(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class LmsSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return self.compviz_wrapper_model.get_sigmas(n).to(self.device)

    @torch.no_grad()
    def do_sample(self):
        x = sample_lms(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class Dpmpp2sAncestralSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return get_sigmas_karras(n, self.sigma_min, self.sigma_max, device=self.device)
    @torch.no_grad()
    def do_sample(self):
        x = sample_dpmpp_2s_ancestral(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class DpmppSdeSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return get_sigmas_karras(n, self.sigma_min, self.sigma_max, device=self.device)
    @torch.no_grad()
    def do_sample(self):
        x = sample_dpmpp_sde(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class Dpmpp2mSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return get_sigmas_karras(n, self.sigma_min, self.sigma_max, device=self.device)
    @torch.no_grad()
    def do_sample(self):
        x = sample_dpmpp_2m(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class Dpmpp2mSdeSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return get_sigmas_karras(n, self.sigma_min, self.sigma_max, device=self.device)
    @torch.no_grad()
    def do_sample(self):
        x = sample_dpmpp_2m_sde(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None

class Dpmpp3mSdeSampler(KDiffusionSamplerBase):

    @torch.no_grad()
    def compute_sigmas(self, n):
        return get_sigmas_karras(n, self.sigma_min, self.sigma_max, device=self.device)
    @torch.no_grad()
    def do_sample(self):
        x = sample_dpmpp_3m_sde(self.ldm_wrapper_model, self.x, self.sigmas) # , extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        return x, None
