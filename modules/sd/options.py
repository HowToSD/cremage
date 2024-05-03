import os
import argparse

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
LDM_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs/ldm/configs")
LDM_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/ldm")


def parse_options(arguments=None):
    """
    Parse options.

    Note: If option contains more than 1 word, use an underscore to connect words.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="realistic photo of a turtle",
        help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="",
        help="the negative prompt to avoid in the image"
    )
    parser.add_argument(
        "--init-img",
        type=str,
        default="puppy.png",  # FIXME
        nargs="?",
        help="path to the input image"
    )   
    parser.add_argument(
        "--mask-img",
        type=str,
        default="mask.png",  # FIXME
        nargs="?",
        help="path to the input image"
    )   
    parser.add_argument(
        "--face_input_img",
        type=str,
        default="face_input.png",  # FIXME
        nargs="?",
        help="path to the face input image"
    )   
    parser.add_argument(
        "--face_model",
        type=str,
        default="None",
        nargs="?",
        help="path to the face model"
    )   
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="",
        help="Directory name of the text embedding files",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=1,
        help="Uses an intermediate CLIP transformer block output if you specify a number greater than 1. Specify 2 for anime. 12 is the maximum which is the output of the first transformer block",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="DDIM",
        help="Name of the sampler",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="Number of batches",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,  # Use 5.0 for img2img
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--strength",  # denoising strength
        type=float,
        # default=0.75,
        default=0.85,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )   
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=f"{LDM_CONFIG_DIR}/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--control_net_config",
        type=str,
        default=f"{LDM_CONFIG_DIR}/stable-diffusion/cldm_v15.yaml",  # CONTROLNET changes
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--inpaint_config",
        type=str,
        default=f"{LDM_CONFIG_DIR}/stable-diffusion/inpainting.yaml",
        help="path to config which constructs inpaint model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=f"{LDM_MODEL_DIR}/v1-5-pruned.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--inpaint_ckpt",
        type=str,
        default=f"{LDM_MODEL_DIR}/sd-v1-5-inpainting.ckpt",
        help="path to checkpoint of an inpainting model",
    )    
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default=f"None",  # Use the VAE from the main model
        help="path to checkpoint of the vae model",
    ) 
    parser.add_argument(
        "--control_models",
        type=str,
        default=f"",
        help="List of comma-separated full paths of ControlNets to load. An empty string if not used",
    )
    parser.add_argument(
        "--control_weights",
        type=str,
        default=f"",
        help="List of comma-separated weight to be applied to each ControlNet",
    )
    parser.add_argument(
        "--control_image_path",
        type=str,
        default=f"",
        help="List of comma-separated control image path for each ControlNet",
    )    
    parser.add_argument(
        "--lora_models",
        type=str,
        default=f"",
        help="List of comma-separated full paths of LoRAs to load. An empty string if not used",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=f"",
        help="List of comma-separated weight to be applied to each LoRA",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"  # Use float16 as the base and upcast float32 when needed
    )
    parser.add_argument(
        "--safety_check",
        action="store_true",
        help="enable safety check for generated image"
    )   
    parser.add_argument(
        "--watermark",
        action="store_true",
        help="enable safety check for generated image"
    )   
    parser.add_argument(
        "--save_memory",
        action="store_true",
        help="move CLIP and VAE models to CPU during diffusing to save GPU memory",
        default=True  # TODO. Provide a way to disable this
    )    
    parser.add_argument(
        "--hires_fix_upscaler",
        type=str,
        help="Upscaler to use for hires fix. Specify None to disable hires fix",
        default="None"
    )
    parser.add_argument(
        "--auto_face_fix",
        action="store_true",
        help="Apply face fix automatically for each generated image",
    )
 
    opt = parser.parse_args(arguments)
    return opt
