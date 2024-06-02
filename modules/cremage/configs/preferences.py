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
    sdxl_ldm_model_path: str
    sdxl_ldm_model: str
    refiner_sdxl_ldm_model: str
    sdxl_ldm_inpaint_model: str
    sdxl_vae_model_path: str
    sdxl_vae_model: str
    refiner_sdxl_vae_model: str
    discretization: str
    discretization_sigma_min: float
    discretization_sigma_max: float
    discretization_rho: float
    sampler: str
    sdxl_sampler: str
    sampler_s_churn: float
    sampler_s_tmin: float
    sampler_s_tmax: float
    sampler_s_noise: float
    sampler_eta: float
    sampler_order: int
    sampling_steps: int
    cfg: float
    guider: str
    linear_prediction_guider_min_scale: float
    linear_prediction_guider_max_scale: float
    triangle_prediction_guider_min_scale: float
    triangle_prediction_guider_max_scale: float
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
    sdxl_lora_model_path: str
    sdxl_lora_model_1: str
    sdxl_lora_model_2: str
    sdxl_lora_model_3: str
    sdxl_lora_model_4: str
    sdxl_lora_model_5: str
    sdxl_lora_weight_1: float
    sdxl_lora_weight_2: float
    sdxl_lora_weight_3: float
    sdxl_lora_weight_4: float
    sdxl_lora_weight_5: float
    sdxl_use_refiner: bool
    sdxl_refiner_strength: float
    refiner_sdxl_lora_model_1: str
    refiner_sdxl_lora_model_2: str
    refiner_sdxl_lora_model_3: str
    refiner_sdxl_lora_model_4: str
    refiner_sdxl_lora_model_5: str
    refiner_sdxl_lora_weight_1: float
    refiner_sdxl_lora_weight_2: float
    refiner_sdxl_lora_weight_3: float
    refiner_sdxl_lora_weight_4: float
    refiner_sdxl_lora_weight_5: float
    embedding_path: str
    sdxl_embedding_path: str
    positive_prompt_pre_expansion: str
    negative_prompt_pre_expansion: str
    enable_positive_prompt_pre_expansion: bool
    enable_negative_prompt_pre_expansion: bool
    positive_prompt_expansion: str
    negative_prompt_expansion: str
    enable_positive_prompt_expansion: bool
    enable_negative_prompt_expansion: bool
    enable_hf_internet_connection: bool
    seed: int
    hires_fix_upscaler: str
    hires_fix_scale_factor: float
    auto_face_fix: bool
    hide_k_diffusion_samplers: bool
    face_strength: float
    generator_model_type: str

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
            "sdxl_embedding_path": "models/embeddings_sdxl",
            "vae_model": "vae-ft-mse-840000-ema-pruned.ckpt",
            "sdxl_ldm_model_path": "models/ldm",
            "sdxl_ldm_model": "None",
            "refiner_sdxl_ldm_model": "None",            
            "sdxl_ldm_inpaint_model": "None",
            "sdxl_vae_model_path": "models/vae",
            "sdxl_vae_model": "None",
            "refiner_sdxl_vae_model": "None",
            "discretization": "LegacyDDPMDiscretization",
            "discretization_sigma_min": 0.0292,
            "discretization_sigma_max": 14.6146,
            "discretization_rho": 3.0,
            "guider": "VanillaCFG",
            "linear_prediction_guider_min_scale": 1.0,  # [1.0, 10.0]
            "linear_prediction_guider_max_scale": 1.5,
            "triangle_prediction_guider_min_scale": 1.0,  # [1.0, 10.0]
            "triangle_prediction_guider_max_scale": 2.5,  # [1.0, 10.0]
            "sampler": "DDIM",
            "sdxl_sampler": "DPMPP2M",
            "sampler_s_churn": 0.0,
            "sampler_s_tmin": 0.0,
            "sampler_s_tmax": 999.0,
            "sampler_s_noise": 1.0,
            "sampler_eta": 1.0,
            "sampler_order": 4,
            "sampling_steps": 50,
            "cfg": 7.5,
            "hires_fix_upscaler": "None",
            "hires_fix_scale_factor": 1.5,
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
            "sdxl_lora_model_path": "models/loras",
            "sdxl_lora_model_1": "None",
            "sdxl_lora_model_2": "None",
            "sdxl_lora_model_3": "None",
            "sdxl_lora_model_4": "None",
            "sdxl_lora_model_5": "None",
            "sdxl_lora_weight_1": 1.0,
            "sdxl_lora_weight_2": 1.0,
            "sdxl_lora_weight_3": 1.0,
            "sdxl_lora_weight_4": 1.0,
            "sdxl_lora_weight_5": 1.0,
            "sdxl_use_refiner": False,
            "sdxl_refiner_strength": 0,
            "refiner_sdxl_lora_model_1": "None",
            "refiner_sdxl_lora_model_2": "None",
            "refiner_sdxl_lora_model_3": "None",
            "refiner_sdxl_lora_model_4": "None",
            "refiner_sdxl_lora_model_5": "None",
            "refiner_sdxl_lora_weight_1": 1.0,
            "refiner_sdxl_lora_weight_2": 1.0,
            "refiner_sdxl_lora_weight_3": 1.0,
            "refiner_sdxl_lora_weight_4": 1.0,
            "refiner_sdxl_lora_weight_5": 1.0,
            "positive_prompt_pre_expansion": "score_9, score_8_up, score_7_up, score_6_up, score_4_up, rating_safe, source_anime, ",
            "negative_prompt_pre_expansion": "score_4, score_5, score_6, ",
            "enable_positive_prompt_pre_expansion": False,
            "enable_negative_prompt_pre_expansion": False,
            "positive_prompt_expansion": ", highly detailed, photorealistic, 4k, 8k, uhd, highres, raw photo, best quality, masterpiece",
            "negative_prompt_expansion": ", worst quality, low quality, lowres",
            "enable_positive_prompt_expansion": True,
            "enable_negative_prompt_expansion": True,
            "enable_hf_internet_connection": True,
            "seed": -1,
            "auto_face_fix": False,
            "hide_k_diffusion_samplers": True,
            "face_strength": 1.0,
            "generator_model_type": "SD 1.5"
        }
        OmegaConf.save(config, CONFIG_FILE_PATH)

    config = OmegaConf.structured(Config)

    # Load and merge the YAML file
    yaml_config = OmegaConf.load(CONFIG_FILE_PATH)
    config = OmegaConf.merge(config, yaml_config)
    return config

def save_user_config(config):
    OmegaConf.save(config, CONFIG_FILE_PATH)
