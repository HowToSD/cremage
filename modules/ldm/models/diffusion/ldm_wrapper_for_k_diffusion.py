"""
LatentDiffusionModel (LDM) wrapper for K-diffusion.
See note in k_diffusion_sampler.py
"""
import os
import sys
import logging
from typing import List, Tuple

import torch

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path = [MODULE_ROOT] + sys.path
from k_diffusion.external import CompVisDenoiser

class LDMWrapperForKDiffusion(torch.nn.Module):

    def __init__(self,
                 compviz_wrapper_model:torch.nn.Module,
                #  c:torch.tensor,
                #  unconditional_conditioning:torch.tensor,
                c,
                unconditional_conditioning,
                unconditional_guidance_scale:float):
        """
        Latent diffusion model wrapper to be used with K-Diffusion.
        Adapted from p_sample_ddim method for DDIMSampler (modules/ldm/models/diffusion/ddim.py).

        Args:
            model (LatentDiffusion): Reference to the main LatentDiffusion model instance.
                Note that this model is not a UNet model itself, but it contains a reference to the
                actual UNet model internally.
            c (torch.tensor): CLIP Text embedding for positive prompt
            unconditional_conditioning (torch.tensor): CLIP Text embedding for negative prompt
            unconditional_guidance_scale (float): CFG
        """
        super().__init__()
        self.compviz_model = compviz_wrapper_model
        self.alphas_cumprod = compviz_wrapper_model.inner_model.alphas_cumprod
        self.ddpm_num_timesteps = compviz_wrapper_model.inner_model.num_timesteps
        self.c = c
        self.unconditional_conditioning = unconditional_conditioning
        self.unconditional_guidance_scale = unconditional_guidance_scale

    def apply_model(self, x, t, **kwargs):
        """
        Calls the UNet model to predict the noise in x.

        Args:
            x: Noisy image in latent space.
            t: Time step.
        """
        c = self.c
        unconditional_conditioning = self.unconditional_conditioning
        unconditional_guidance_scale = self.unconditional_guidance_scale

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # e_t = self.ldm_model.apply_model(x, t, c)

            # Change to call CompVizDenoiser model
            e_t = self.compviz_model(x, t, c)  # this wrapper -> compviz -> sd ldm

        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                # Make sure that uc is also a dict
                assert isinstance(unconditional_conditioning, dict)

                c_in = dict()  # Create a result dict
                for k in c:
                    if isinstance(c[k], list):
                        # If the value is a list, then concatenate each element
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:  # If the value is a tensor, then just concatenate uc and c
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])

            # Pass both negative and positive prompt embedding and get inference result.
            # Split the prediction to the noise for negative and the noise for positive.
            # e_t_uncond, e_t = self.ldm_model.apply_model(x_in, t_in, c_in).chunk(2)

            # Change to call CompVizDenoiser model
            # e_t_uncond, e_t = self.compviz_model(x_in, t_in, **c_in).chunk(2)
            c_in = {"cond":{"c_crossattn": [c_in]}}
            e_t_uncond, e_t = self.compviz_model(x_in, t_in, **c_in).chunk(2)

            # Update the noise prediction
            # This is a vector that originates from e_t_uncond.
            # This vector is pointed toward e_t from there.
            # The length of the vector is determined by CFG.
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        return e_t

    def forward(self, *args, **kwargs):
        """
        This is called from the sampler"""
        return self.apply_model(*args, **kwargs)
