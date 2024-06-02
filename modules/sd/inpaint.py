"""
Based on inpaint_st.py in https://github.com/runwayml/stable-diffusion and adapted for Cremage.
"""
import os
import sys
import logging
import time
import random
import json
import contextlib

import numpy as np
import cv2 as cv
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from omegaconf import OmegaConf
from einops import repeat
import torch
from pytorch_lightning import seed_everything

from .set_up_path import PROJECT_ROOT
LDM_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/ldm")
LDM_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs/ldm/configs")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from sd.options import parse_options
from sd.image_generator import chunk
from sd.image_generator import load_model_from_config
from cremage.ui.update_image_handler import update_image
from cremage.utils.generation_status_updater import StatusUpdater
from cremage.utils.image_utils import bbox_for_multiple_of_64
from cremage.utils.image_utils import resize_with_padding
from ip_adapter.ip_adapter_faceid import generate_face_embedding_from_image

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

MAX_SIZE = 640


def make_batch_sd(
        image,
        mask,
        txt,
        negative_prompt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)  # bhwc to bchw
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0  # [0, 255] to [0, 2] to [-1, 1]

    mask = np.array(mask.convert("L"))  # to grayscale by using "L"uminance
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]   # Rank 2 [h, w] to rank 4 [b, c, h, w]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)  # Still [-1, 1] but black mask is now 0. Black masked part won't be painted

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "negative_prompt":  num_samples * [negative_prompt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch


def generate(opt, ui_thread_instance=None, status_queue=None):

    if opt.seed == 0:
        seed = random.getrandbits(32)
    else:
        seed = opt.seed
    seed_everything(seed)

    config = OmegaConf.load(f"{opt.inpaint_config}")

    # Process FaceID
    load_face_id = False
    if os.path.exists(opt.face_input_img) and os.path.exists(opt.face_model):
        logger.info("Face input image is specified and the face model is found. Generating face embedding")
        face_embedding, uc_face_embedding = generate_face_embedding_from_image(
            opt.face_input_img,
            opt.face_model,
            batch_size=opt.n_samples)
        logger.info(f"Generated face_embedding. Shape: {face_embedding.shape}")
        load_face_id = True
    elif os.path.exists(opt.face_input_img) and os.path(opt.face_model) is False:
        logger.info("Face input image is specified but the face model is not found. Ignoring face image")

    # Load the main LDM model
    print("Loading the main model ...")
    model = load_model_from_config(config, opt, inpainting=True)
    print("Loaded the main model")

    # Resize image
    # Spec
    # You need to pass the multiples of 64.
    # The user may not have specified h or w that comply with this requirement.
    # So we need to scale the image the the smallest w and h that is still
    # bigger than opt.W and opt.H and add padding.

    original_width, original_height = opt.W, opt.H
    padded_w, padded_h = bbox_for_multiple_of_64(opt.W, opt.H)
    padding_used = False  # True means that we need to crop after generation
    bbox_to_crop = None
    # if padded_w != opt.W or padded_h != opt.H:
    #     padding_used = True
        # # Compute left top to crop after the image is generated
        # crop_x = (padded_w - opt.W) // 2 + (padded_w - opt.W) % 2
        # crop_y = (padded_h - opt.H) // 2 + (padded_h - opt.H) % 2
        # opt.W, opt.H = padded_w, padded_h
        # bbox_to_crop = (crop_x, crop_y, crop_x + original_width, crop_y + original_height)

    image = Image.open(opt.init_img)
    w, h = image.size

    # if w != opt.W or h != opt.H:
    if padded_w != w or padded_h != h:
        padding_used = True
        # image = image.resize((padded_w, padded_h), resample=PIL.Image.LANCZOS)
        image, bbox_to_crop = resize_with_padding(
                    image,
                    target_width=padded_w,
                    target_height=padded_h, return_bbox=True)

    original_image = image  # Make a copy to produce the final result

    # Load mask and resize as needed
    mask = Image.open(opt.mask_img)  # white area will be painted!
    w, h = mask.size
    # if w != opt.W or h != opt.H:
    #     mask = mask.resize((opt.W, opt.H), resample=PIL.Image.LANCZOS)
    if padded_w != w or padded_h != h:
        # image = image.resize((padded_w, padded_h), resample=PIL.Image.LANCZOS)
        mask, _ = resize_with_padding(
                    mask,
                    target_width=padded_w,
                    target_height=padded_h, return_bbox=True)
    opt.W, opt.H = padded_w, padded_h

    sampler = DDIMSampler(model)  # FIXME

    prompt = opt.prompt # prompt, 
    negative_prompt = opt.negative_prompt

    scale = opt.scale # scale, 
    sampling_steps = opt.sampling_steps # sampling_steps, 
    num_samples=opt.n_samples 
    w=opt.W 
    h=opt.H
  
    device = torch.device(os.environ.get("GPU_DEVICE", "cpu"))
    model = sampler.model

    if opt.seed == -1:
        seed = random.getrandbits(32)
    else:
        seed = opt.seed
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    sample_path = outpath
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    if torch.cuda.is_available():
        precision_context = torch.autocast(device_type=os.environ.get("GPU_DEVICE", "cpu"))
    else:
        precision_context = contextlib.nullcontext()

    with torch.no_grad():
        with precision_context:
            batch = make_batch_sd(
                image,
                mask,
                txt=prompt,
                negative_prompt=negative_prompt,
                device=device,
                num_samples=num_samples)

            # Generate text embedding for positive prompt using CLIP
            c = model.cond_stage_model.encode(
                batch["txt"],
                embedding_dir=opt.embedding_path,
                clip_skip=opt.clip_skip)

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # Generate text embedding for negative prompt using CLIP
            uc_cross = model.get_unconditional_conditioning(
                num_samples,
                batch["negative_prompt"][0:1],
                embedding_dir=opt.embedding_path,
                clip_skip=opt.clip_skip)
           
            # Negative embedding (uc) and positive embedding(c) can have
            # a size mismatch if embedding is concatenated along the first axis.
            # e.g. 308 vs 77 for tensor shape [2, 308, 768] vs [2, 77, 768]
            # To cope with this, we will be using the filler embedding 
            filler = model.get_learned_conditioning(opt.n_samples * [""],
                                                    embedding_dir=opt.embedding_path,
                                                    clip_skip=opt.clip_skip)
            
            # Calculate the difference in size along axis 1
            diff = uc_cross.shape[1] - c.shape[1]
            assert diff % 77 == 0
            diff = diff // 77
            if diff < 0:  # uc is smaller
                diff = -diff
                # Repeat the filler tensor the required number of times and concatenate it with uc
                repeated_filler = filler.repeat(1, diff, 1)  # The 'diff' times repetition is along the second axis
                uc_cross = torch.cat((uc_cross, repeated_filler), axis=1)
            else:  # c is smaller or they are equal
                repeated_filler = filler.repeat(1, diff, 1)  # Repeat along the second axis
                c = torch.cat((c, repeated_filler), axis=1)
            assert uc_cross.shape[1] == c.shape[1]

            if load_face_id:  # Append face embedding. New shape should be (1, 81, 768)
                c = torch.concat((c, face_embedding), axis=1)
                uc_cross = torch.concat((uc_cross, uc_face_embedding), axis=1)
                assert(c.shape[1] % 77 == 4 and c.shape[2] == 768)
                assert(uc_cross.shape[1] % 77 == 4 and uc_cross.shape[2] == 768)

            cond={"c_concat": [c_cat], "c_crossattn": [c]}
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]

            status_update_func = None
            if status_queue:
                status_updater = StatusUpdater(sampling_steps, 1, status_queue)
                status_update_func = status_updater.status_update

            samples_cfg, intermediates = sampler.sample(
                    sampling_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
                    callback=status_update_func
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)
            if opt.safety_check:
                from cremage.utils.safety_check import check_safety
                result, has_nsfw_concept = check_safety(result)
            result = result*255
            

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]

    if opt.watermark:
        from cremage.utils.watermark import put_watermark
        result = [put_watermark(img) for img in result]
    
    for i, img in enumerate(result):
        time_str = time.time()

        generation_parameters = {
            "time": time.time(),
            "positive_prompt": opt.prompt,
            "negative_prompt": opt.negative_prompt,
            "ldm_inpaint_model": os.path.basename(opt.inpaint_ckpt),
            "vae_model": os.path.basename(opt.vae_ckpt),
            "sampler": "DDIM",  # FIXME
            "sampling_iterations": opt.sampling_steps,  # FIXME
            "image_height": opt.H,
            "image_width": opt.W,
            "clip_skip": opt.clip_skip,
            "seed": seed # FIXME
        }

        str_generation_params = json.dumps(generation_parameters)
        metadata = PngInfo()
        metadata.add_text("generation_data", str_generation_params)

        # Now only use the masked part of the image
        inpainted_image = img

        # We have now
        # original_image
        # mask (white section was painted, black section was untouched)
        # inpainted_image
        # We want to overlay inpainted_image & mask on original image
        # Convert PIL images to OpenCV format
        original_image_cv = cv.cvtColor(np.array(original_image), cv.COLOR_RGBA2BGR)
        inpainted_image_cv = cv.cvtColor(np.array(inpainted_image), cv.COLOR_RGBA2BGR)       
        mask_cv = cv.cvtColor(np.array(mask), cv.COLOR_RGB2GRAY)

        use_seamless_clone = False
        if use_seamless_clone:
            # Find contours
            contours, _ = cv.findContours(mask_cv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Find the largest contour based on area and calculate its centroid
            largest_contour = max(contours, key=cv.contourArea)
            M = cv.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # Fallback to the center of the image if calculation is not possible
                cX, cY = mask_cv.shape[1] // 2, mask_cv.shape[0] // 2

            center = (cX, cY)

            # Perform the seamless cloning
            try:
                output = cv.seamlessClone(inpainted_image_cv, original_image_cv, mask_cv, center, cv.NORMAL_CLONE)
                # Convert the result back to PIL format if needed
                output_image = Image.fromarray(cv.cvtColor(output, cv.COLOR_BGR2RGB))
            except:
                logger.warning("Seamless clone failed. Switching back to simple paste")
                # In case mask is not binary (0 or 255), binarize it.
                mask_binary = cv.threshold(mask_cv, 127, 255, cv.THRESH_BINARY)[1]

                # Create an inverse mask
                inverse_mask_binary = cv.bitwise_not(mask_binary)

                # Use the masks to grab parts of the images
                inpainted_part = cv.bitwise_and(inpainted_image_cv, inpainted_image_cv, mask=mask_binary)
                original_part = cv.bitwise_and(original_image_cv, original_image_cv, mask=inverse_mask_binary)

                # Combine the parts
                combined_image_cv = cv.add(inpainted_part, original_part)

                # Convert the combined image back to a PIL image
                output_image = Image.fromarray(cv.cvtColor(combined_image_cv, cv.COLOR_BGR2RGB))
        else:  # do not use seamless clone. FIXME. Clean up
            # mask_binary = cv.threshold(mask_cv, 127, 255, cv.THRESH_BINARY)[1]

            # # Create an inverse mask
            # inverse_mask_binary = cv.bitwise_not(mask_binary)

            # # Use the masks to grab parts of the images
            # inpainted_part = cv.bitwise_and(inpainted_image_cv, inpainted_image_cv, mask=mask_binary)
            # original_part = cv.bitwise_and(original_image_cv, original_image_cv, mask=inverse_mask_binary)

            # # Combine the parts
            # combined_image_cv = cv.add(inpainted_part, original_part)

            # # Convert the combined image back to a PIL image
            # output_image = Image.fromarray(cv.cvtColor(combined_image_cv, cv.COLOR_BGR2RGB))

            # Apply Gaussian blur to the mask
            blurred_mask = cv.GaussianBlur(mask_cv, (11, 11), 0)  # Change to (21, 21)

            # Normalize the blurred mask to ensure it's in the 0-255 range
            blurred_mask = np.clip(blurred_mask, 0, 255)

            # Instead of using bitwise operations, manually interpolate between the images
            # Convert mask to float
            blurred_mask_float = blurred_mask.astype(np.float32) / 255.0
            inverse_mask_float = 1.0 - blurred_mask_float

            # Convert images to float
            original_float = original_image_cv.astype(np.float32)
            inpainted_float = inpainted_image_cv.astype(np.float32)

            # Interpolate
            combined_float = (inpainted_float * blurred_mask_float[..., np.newaxis]) + (original_float * inverse_mask_float[..., np.newaxis])

            # Convert back to an 8-bit image
            combined_image_cv = np.clip(combined_float, 0, 255).astype(np.uint8)

            # Convert the combined image back to PIL format
            output_image = Image.fromarray(cv.cvtColor(combined_image_cv, cv.COLOR_BGR2RGB))

            # Remove padding
            if padding_used and bbox_to_crop:
                output_image = output_image.crop(bbox_to_crop)

            output_image.save(
                os.path.join(sample_path,
                            f"{base_count:05}_{time_str}.png"),
                            pnginfo=metadata)

        if ui_thread_instance:
            update_image(ui_thread_instance,
                         output_image,
                        generation_parameters=str_generation_params)

        base_count += 1



    print(f"Images generated in {sample_path}")
    return result


def inpaint_parse_options_and_generate(args=None,
                                       ui_thread_instance=None,
                                       status_queue=None):
    opt = parse_options(args)    
    generate(opt, ui_thread_instance=ui_thread_instance, status_queue=status_queue)


if __name__ == "__main__":
    inpaint_parse_options_and_generate()
