import os
import sys

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path


hires_fix_upscaler_name_list = [
    "None",  # Disable upscaler
    "Latent",  # Upscale in latent scape using Torch interpolate
    "Lanczos"  # Upscale in pixel space
]
