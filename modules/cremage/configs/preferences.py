import os
from dataclasses import dataclass
from omegaconf import OmegaConf

CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "config.yaml")

@dataclass
class Config:
    safety_check: bool
    watermark: bool
    image_width: int
    image_height: int
    clip_skip: int
    denoising_strength: float
    batch_size: int
    number_of_batches: int
    ldm_model_path: str
    ldm_model: str
    ldm_inpaint_model: str
    vae_model_path: str
    vae_model: str
    control_model_path: str
    control_model: str
    sampler: str
    sampling_steps: int
    cfg: float
    lora_model_path: str
    lora_model_1: str
    lora_model_2: str
    lora_model_3: str
    lora_model_4: str
    lora_model_5: str
    lora_weight_1: float
    lora_weight_2: float
    lora_weight_3: float
    lora_weight_4: float
    lora_weight_5: float
    embedding_path: str
    positive_prompt_expansion: str
    negative_prompt_expansion: str
    enable_positive_prompt_expansion: bool
    enable_negative_prompt_expansion: bool
    enable_hf_internet_connection: bool
    seed: int
    hires_fix_upscaler: str
    auto_face_fix: bool
    hide_k_diffusion_samplers: bool

def load_user_config():

    if os.path.exists(CONFIG_FILE_PATH) is False:
        # Set the default config values
        config = {
            "safety_check": True,
            "watermark": False,
            "image_width": 512,
            "image_height": 512,
            "clip_skip": 1,
            "batch_size": 1,
            "number_of_batches": 1,
            "denoising_strength": 0.7,
            "ldm_model_path": "models/ldm",
            "ldm_model": "v1-5-pruned.ckpt",
            "ldm_inpaint_model": "sd-v1-5-inpainting.ckpt",
            "vae_model_path": "models/vae",
            "control_model_path": "models/control_net",
            "control_model": "None",
            "embedding_path": "models/embeddings",
            "vae_model": "vae-ft-mse-840000-ema-pruned.ckpt",
            "sampler": "DDIM",
            "sampling_steps": 50,
            "cfg": 7.5,
            "hires_fix_upscaler": "None",
            "lora_model_path": "models/loras",
            "lora_model_1": "None",
            "lora_model_2": "None",
            "lora_model_3": "None",
            "lora_model_4": "None",
            "lora_model_5": "None",
            "lora_weight_1": 1.0,
            "lora_weight_2": 1.0,
            "lora_weight_3": 1.0,
            "lora_weight_4": 1.0,
            "lora_weight_5": 1.0,
            "positive_prompt_expansion": ", highly detailed, photorealistic, 4k, 8k, uhd, highres, raw photo, best quality, masterpiece",
            "negative_prompt_expansion": ", worst quality, low quality, lowres",
            "enable_positive_prompt_expansion": True,
            "enable_negative_prompt_expansion": True,
            "enable_hf_internet_connection": True,
            "seed": -1,
            "auto_face_fix": False,
            "hide_k_diffusion_samplers": True
        }
        OmegaConf.save(config, CONFIG_FILE_PATH)

    config = OmegaConf.structured(Config)

    # Load and merge the YAML file
    yaml_config = OmegaConf.load(CONFIG_FILE_PATH)
    config = OmegaConf.merge(config, yaml_config)
    return config

def save_user_config(config):
    OmegaConf.save(config, CONFIG_FILE_PATH)
