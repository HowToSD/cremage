# Video frames generation code using SVD.
# Currently this code generates 25 frames
# Based scripts/sampling/simple_video_sample.py from stability ai.
# Check project root for license
import os
import sys
import logging
import math
from glob import glob
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
from einops import repeat
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove

SDXL_MODULE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.realpath(os.path.join(SDXL_MODULE_ROOT, "configs"))
SAMPLING_CONFIG_DIR = os.path.realpath(os.path.join(SDXL_MODULE_ROOT, "scripts", "sampling", "configs"))
PROJECT_ROOT = os.path.realpath(os.path.join(SDXL_MODULE_ROOT, "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules") 
sys.path = [SDXL_MODULE_ROOT, MODULE_ROOT] + sys.path
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from .vram_mode import load_model, unload_model, set_lowvram_mode, initial_svd_model_load
from cremage.status_queues.video_generation_status_queue import video_generation_status_queue

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def _show_warning(version, H, W, motion_bucket_id, fps_id):
    if (H, W) != (576, 1024) and "sv3d" not in version:
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )
    if (H, W) != (576, 576) and "sv3d" in version:
        print(
            "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
        )
    if motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")


def sample(
    input_path: str = None,  # Can either be image file or folder with image files
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "svd_xt_1_1",
    fps_id: int = 6,
    motion_bucket_id: int = 75, # 127
    cond_aug: float = 0,  # 0.02
    seed: int = 42,  # 23,
    decoding_t: int = 1,
    device: str = "cuda",
    output_path: Optional[str] = None,
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
    checkpoint_path=None,
    apply_filter=True,
    apply_watermark=False
):
    """
    Generates a single sample conditioned on an image `input_path` or
    multiple images, one for each image file in folder `input_path`.

    Note: Original decoding_t was 14, but now is is set to 1 to reduce
    the possibility of OOM.

    For motion bucket id discussion, see https://www.reddit.com/r/StableDiffusion/comments/1azn2lx/stable_video_diffusion_v11_is_pretty_good_at/
    It is switched to 75 from 127.
    """
    if version == "svd_xt_1_1":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_path = default(output_path, "outputs/svd/")
        model_config = os.path.join(SAMPLING_CONFIG_DIR, "svd_xt_1_1.yaml")

        if os.path.exists(output_path) is False:
            logger.info(f"Creating {output_path}")
            os.makedirs(output_path)

    else:
        raise ValueError(f"Version {version} does not exist.")

    if apply_filter:
        # from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
        # filter = DeepFloydDataFiltering(verbose=False, device=device)
        from safety.safety_filter import SafetyFilter
        filter = SafetyFilter()
    else:
        filter = None

    video_generation_status_queue.put(f"Loading the video model...")
    model = load_video_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
        checkpoint_path=checkpoint_path
    )
    set_lowvram_mode(True)  # Set to low VRAM to reduce VRAM consumption
    initial_svd_model_load(model.first_stage_model)
    initial_svd_model_load(model.conditioner)
    initial_svd_model_load(model.model)
    video_generation_status_queue.put(f"Loaded the video model.")

    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_img_path in all_img_paths:
        with Image.open(input_img_path) as image:
            if image.mode == "RGBA":
                input_image = image.convert("RGB")
            else:
                input_image = image
            w, h = image.size  # 1024, 576

            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                input_image = input_image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

            input_image = input_image.convert("RGB")

        image = np.array(input_image).astype(np.uint8)
        image = torch.tensor(image).float() / 255.0
        image = rearrange(image, "h w c -> c h w")
        image = image * 2.0 - 1.0  # Cremage note: [0, 1] to [0, 2] to [-1, 1]

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        _show_warning(version, H, W, motion_bucket_id, fps_id)

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],  # num_frames is set to 14: Cremage
                    T=num_frames,
                    device=device,
                )

                load_model(model.conditioner)
                load_model(model.first_stage_model)
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )
                unload_model(model.conditioner)
                unload_model(model.first_stage_model)

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                # Denoise
                load_model(model.model)
                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )
                video_generation_status_queue.put(f"Sampling starting ...")
                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                video_generation_status_queue.put(f"Sampling completed ...")
                model.en_and_decode_n_samples_a_time = decoding_t

                # VAE
                unload_model(model.model)
                load_model(model.first_stage_model)
                video_generation_status_queue.put(f"Converting images to pixel space using VAE ...")
                logger.info("Converting images to pixel space using VAE ...")
                samples_x = model.decode_first_stage(samples_z)
                video_generation_status_queue.put(f"VAE completed.")
                logger.info("VAE completed.")
                unload_model(model.first_stage_model)

                if "sv3d" in version:
                    samples_x[-1:] = value_dict["cond_frames_without_noise"]
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                logger.info(f"Samples.shape: {samples.shape}") # [25, 3, 576, 1024]

                # Output
                base_count = len(glob(os.path.join(output_path, "*.mp4")))

                imageio.imwrite(
                    os.path.join(output_path, f"{base_count:06d}.jpg"), input_image
                )

                if apply_watermark:
                    samples = embed_watermark(samples)

                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                if apply_filter:
                    vid, _ = filter(vid)

                for i, img in enumerate(vid):
                    img = Image.fromarray(img)
                    img.save(os.path.join(output_path, f"frame_{i:06d}.jpg"))

                video_generation_status_queue.put(f"Video frames generated.")


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_video_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
    checkpoint_path = None
):
    config = OmegaConf.load(config)

    # Override checkpoint path
    config.model.params.ckpt_path = checkpoint_path
    if os.path.exists(checkpoint_path) is False:
        logger.error(f"{checkpoint_path} is not found. Aborting")
        return None

    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    return model
