import os
import logging
import sys
import torch

from sgm.modules.diffusionmodules.discretizer import Discretization

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

class Img2ImgDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 1.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization: Discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        logger.debug(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        logger.debug("prune index:", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        logger.debug(f"sigmas after pruning: ", sigmas)
        return sigmas


class Txt2NoisyDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas

    Examples for 30 sampling steps with 0.15 refiner strength (Added by Cremage):
    Sigmas before pruning:
      [14.6146, 11.9484,  9.9172,  8.3028,  6.9739,  5.9347,  5.0878,  4.3728,
        3.7997,  3.3211,  2.9183,  2.5671,  2.2765,  2.0260,  1.8024,  1.6129,
        1.4458,  1.2931,  1.1606,  1.0410,  0.9324,  0.8299,  0.7380,  0.6524,
        0.5693,  0.4924,  0.4179,  0.3417,  0.2653,  0.1793,  0.0000]

    Prune index: 3

    Sigmas after pruning (note [0.2653,  0.1793,  0.0000] are gone):
       14.6146, 11.9484,  9.9172,  8.3028,  6.9739,  5.9347,  5.0878,  4.3728,
         3.7997,  3.3211,  2.9183,  2.5671,  2.2765,  2.0260,  1.8024,  1.6129,
         1.4458,  1.2931,  1.1606,  1.0410,  0.9324,  0.8299,  0.7380,  0.6524,
         0.5693,  0.4924,  0.4179,  0.3417]

    params:
        discretization: Sigma generator object instance (Note added by Cremage).
          Calling this object returns sigmas.
        strength: float between 0.0 and 1.0. 0.0 means full sampling (all sigmas are returned)
    """

    def __init__(
        self, discretization: Discretization, strength: float = 0.0, original_steps=None
    ):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)  # Cremage: Noise generator
        logger.debug(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1

        # Cremage note
        # if self.strength is 0.2 and steps is 30,
        # strength * steps = 6 steps
        # -1 to convert to an index
        # Now, we also want to put min() so that this won't exceed
        # the total number of steps (e.g. the user puts 2 which would result in 60 steps
        # or index 59).
        # Then we want to make sure that it's >= 0.
        # So overall, this is equivalent of clip(0, steps-1)
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        
        # Cremage note
        # Remove the beginning, but we will flip the array after this
        # so we are removing the final part of sigmas (e.g. last 6 steps)
        # to finish denoising
        sigmas = sigmas[prune_index:] 
        logger.debug("prune index:", prune_index)
        sigmas = torch.flip(sigmas, (0,))
        logger.debug(f"sigmas after pruning: ", sigmas)
        return sigmas
