"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""
import os
import sys
import logging
from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (get_ancestral_step,
                                                        linear_multistep_coeff,
                                                        to_d, to_neg_log_sigma,
                                                        to_sigma)
from ...util import append_dims, default, instantiate_from_config

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..","..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.status_queues.denoising_status_queue import denoising_status_queue

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = os.environ.get("GPU_DEVICE", "cpu"),
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        """
        Prepare data before sampling iteration.

        Compute sigmas for sampling steps
        Multiply x with a number that is slightly larger tha sigma[0].
        Set uc to c if uc is not defined
        Compute s_in, which is a list of 1's.

        Docstring was added by Cremage.

        Args:
            x: Noisy image in latent space
            cond: Conditional dict (e.g. including text embedding for positive prompt)
                  Dictionary contains two keys:
                    "crossattn": shape (bs, 77, 2048)
                    "vector": shape (bs, 2816)
            uc: Conditional dict (e.g. including text embedding for negative prompt)
            num_steps: User-specified sampling steps (e.g. 20)
        """
        # Cremage note
        # Compute sigmas based on the number of steps and discretization type
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )

        # Cremage note
        # If uc is not defined, use c as uc
        uc = default(uc, cond)

        # Cremage note
        # Adjust x using the initial sigma value
        # The multiplier is sqrt(1 + sigma_0^2)
        # This results in x being scaled by a value slightly larger than sigma_0
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        # Cremage note
        # Create rank 1 (1d array) on the same device
        # For bs=2, [1, 1]
        #     bs=1, [1]
        # s_in is used to populate an array for the batch with a single sigma value.
        # e.g. if sigma is 14, then multiplying with s_in will yield:
        # [14, 14, ..., 14]
        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        """
        Combines guider adjustment and model inference for denoising.
        Specifically, it does the following:
        1. Guider adjustment of model input prior to the model inference
           Specifically, this concatenates uc and c along batch-axis
           so that model outputs two sets of images: one based on uc another based on c.
           x is extended by 2x to match the new bs.
        2. Model inference
        3. Guider adjustment of model output (applies x = x_uc + cfg * (x_c - x_uc))

        Docstring was added by Cremage.

        Args:
            x: Noisy image in latent space
            denoiser: Model
            sigma: Sigma
            cond: Conditional dict (e.g. including text embedding for positive prompt)
                  Dictionary contains two keys:
                    "crossattn": shape (bs, 77, 2048)
                    "vector": shape (bs, 2816)
            uc: Conditional dict (e.g. including text embedding for negative prompt)
        """        
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            logger.debug("### Sampling setting ###")
            logger.debug(f"Sampler: {self.__class__.__name__}")
            logger.debug(f"Discretization: {self.discretization.__class__.__name__}")
            logger.debug(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            denoising_status_queue.put(f"Sampling {i+1} / {num_sigmas}")
            print(f"Sampling {i+1} / {num_sigmas}")
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )
        denoising_status_queue.put(f"Sampling completed")
        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        # Cremage note: Convert sigma to t
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]

        # Cremage note: Compte t delta
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()  # Cremage note: e**x - 1

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        # Cremage note
        # 1. Combine cond + uc using guider. bs becomes x2
        #    cond is a dict containing crossattn and vector keys
        #    crossattn value shape: (bs, 77, 2048)
        #    vector value shape:    (bs, 2816)
        # 2. Calls the model with x, sigma, combined cond
        # 3. Using guider, applies x = x_uc + cfg * (x_c - x_uc).
        #    bs becomes original length
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        # Cremage note
        # Now compute x from denoised (raw model output + cfg adjustment)
        # denoised value should be untouched in below code.

        # Cremage note
        # Convert sigma to t
        #   t = -log(sigma)
        #   You can also get sigma from t:
        #   sigma = exp(-t)
        # h: t_next - t
        # r = h_last / h
        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)

        # Cremage note
        # get_mult returns either:
        # m0, m1
        #   or
        # m0, m1, m2, m3  (if prev sigma is not None)
        # m0 = sigma_next / sigma_current
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        # Cremage note:
        # Consider this as the key of the denoising step
        #   less noisy image = noisy image - noise
        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        # Cremage note: self.get_sigma_gen is range(). Do not get confused
        # This is just written this way to display tqdm.
        for i in self.get_sigma_gen(num_sigmas):  # Cremage: For 20 steps, num_sigmas is 21
            x, old_denoised = self.sampler_step(
                old_denoised,

                # Cremage: Pass three sigmas
                #   prev, current, next
                # s_in * means replicate a single sigma value to fill the array for the batch
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                
                denoiser,  # Cremage: Model
                x,
                cond,
                uc=uc,
            )

        return x
