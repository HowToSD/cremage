"""
Generate button handler for the main UI.
"""
import os
import logging
import sys
import threading
import queue

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [PROJECT_ROOT, MODULE_ROOT] + sys.path
from sd.txt2img import parse_options_and_generate
from sd.img2img import img2img_parse_options_and_generate
from sd.inpaint import inpaint_parse_options_and_generate

from cremage.const.const import *
from cremage.utils.gtk_utils import text_view_get_text
from cremage.utils.misc_utils import override_args_list, generate_lora_params
from cremage.utils.misc_utils import join_directory_and_file_name
from cremage.utils.prompt_history import update_prompt_history
from cremage.ui.ui_to_preferences import copy_ui_field_values_to_preferences

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def generate_handler(app, widget, event) -> None:
    """
    Event handler for the Generation button click

    Args:
        app (Gtk.Window): The application window instance
        widget: The button clicked
    """
    # Copy UI fields to preferences
    copy_ui_field_values_to_preferences(app)

    textbuffer = app.positive_prompt.get_buffer()
    positive_prompt = textbuffer.get_text(textbuffer.get_start_iter(), textbuffer.get_end_iter(), False)
    positive_prompt_before_expansion = positive_prompt

    textbuffer = app.negative_prompt.get_buffer()
    negative_prompt = textbuffer.get_text(textbuffer.get_start_iter(), textbuffer.get_end_iter(), False)
    negative_prompt_before_expansion = negative_prompt

    # Grab expansion value irrespective of enabling status
    positive_prompt_expansion = app.preferences["positive_prompt_expansion"]
    negative_prompt_expansion = app.preferences["negative_prompt_expansion"]
    positive_prompt_pre_expansion = app.preferences["positive_prompt_pre_expansion"]
    negative_prompt_pre_expansion = app.preferences["negative_prompt_pre_expansion"]

    # Only update the prompt if expansion is enabled
    if app.preferences["enable_positive_prompt_pre_expansion"]:
        positive_prompt = app.preferences["positive_prompt_pre_expansion"] + " " + positive_prompt
        logger.debug(f"Positive prompt after pre_expansion: {positive_prompt}")

    if app.preferences["enable_negative_prompt_pre_expansion"]:
        negative_prompt = app.preferences["negative_prompt_pre_expansion"] + " " + negative_prompt
        logger.debug(f"Negative prompt after pre_expansion: {negative_prompt}")

    if app.preferences["enable_positive_prompt_expansion"]:
        positive_prompt += " " + app.preferences["positive_prompt_expansion"]
        logger.debug(f"Positive prompt after expansion: {positive_prompt}")

    if app.preferences["enable_negative_prompt_expansion"]:
        negative_prompt += " " + app.preferences["negative_prompt_expansion"]
        logger.debug(f"Negative prompt after expansion: {negative_prompt}")

    # Read generation preference from UI
    generator_model_type = app.preferences["generator_model_type"]
    sampling_steps = str(app.preferences["sampling_steps"])
    unconditional_guidance_scale = str(app.preferences["cfg"])
    image_height = str(app.preferences["image_height"])
    image_width = str(app.preferences["image_width"])
    clip_skip = str(app.preferences["clip_skip"])
    seed = str(app.preferences["seed"])
    batch_size = str(app.preferences["batch_size"])
    number_of_batches = str(app.preferences["number_of_batches"])
    denoising_strength = str(app.preferences["denoising_strength"])  # hires-fix and img2img
    control_model_path = join_directory_and_file_name(app.preferences["control_model_path"], app.preferences["control_model"])
    auto_face_fix = app.preferences["auto_face_fix"]

    control_models = control_model_path  # FIXME. Support multiple ControlNet
    control_weights = str(1.0)  # FIXME. Support weight

    hires_fix_upscaler = app.preferences["hires_fix_upscaler"]
    hires_fix_scale_factor = app.preferences["hires_fix_scale_factor"]

    if app.preferences["generator_model_type"] == "SD 1.5":
        embedding_path = app.preferences["embedding_path"]
        sampler = app.preferences["sampler"]
        ldm_path = join_directory_and_file_name(app.preferences["ldm_model_path"], app.preferences["ldm_model"])
        ldm_inpaint_path = join_directory_and_file_name(app.preferences["ldm_model_path"], app.preferences["ldm_inpaint_model"])
        vae_path = join_directory_and_file_name(app.preferences["vae_model_path"], app.preferences["vae_model"])

        # LoRA
        lora_dict = dict()
        for k, v in app.preferences.items():
            if k.startswith("lora_model"):
                lora_dict[k] = v
            elif k.startswith("lora_weight"):
                lora_dict[k] = v
        lora_dict["lora_model_path"] = app.preferences["lora_model_path"]

        # Comma-separated full path to LoRAs and weights
        # e.g. "model/loras/foo.safetensors,model/loras/bar.safetensors"
        #      "1.0,0.9"
        lora_models, lora_weights = generate_lora_params(lora_dict)
    elif app.preferences["generator_model_type"] == "SDXL":
        embedding_path = app.preferences["sdxl_embedding_path"]
        sampler = app.preferences["sdxl_sampler"]+"Sampler"
        ldm_path = join_directory_and_file_name(app.preferences["sdxl_ldm_model_path"], app.preferences["sdxl_ldm_model"])
        refiner_ldm_path = join_directory_and_file_name(app.preferences["sdxl_ldm_model_path"], app.preferences["refiner_sdxl_ldm_model"])
        ldm_inpaint_path = join_directory_and_file_name(app.preferences["sdxl_ldm_model_path"], app.preferences["sdxl_ldm_inpaint_model"])
        vae_path = join_directory_and_file_name(app.preferences["sdxl_vae_model_path"], app.preferences["sdxl_vae_model"])
        refiner_vae_path = join_directory_and_file_name(app.preferences["sdxl_vae_model_path"], app.preferences["refiner_sdxl_vae_model"])
        
        # SDXL LoRA
        lora_dict = dict()
        for k, v in app.preferences.items():
            if k.startswith("sdxl_lora_model"):
                lora_dict[k] = v
            elif k.startswith("sdxl_lora_weight"):
                lora_dict[k] = v
        lora_dict["sdxl_lora_model_path"] = app.preferences["sdxl_lora_model_path"]

        # Comma-separated full path to LoRAs and weights
        # e.g. "model/loras/foo.safetensors,model/loras/bar.safetensors"
        #      "1.0,0.9"
        lora_models, lora_weights = generate_lora_params(lora_dict, sdxl=True)

        # SDXL Refiner LoRA
        lora_dict = dict()
        for k, v in app.preferences.items():
            if k.startswith("refiner_sdxl_lora_model"):
                lora_dict[k] = v
            elif k.startswith("refiner_sdxl_lora_weight"):
                lora_dict[k] = v
        lora_dict["sdxl_lora_model_path"] = app.preferences["sdxl_lora_model_path"]
        refiner_sdxl_lora_models, refiner_sdxl_lora_weights = \
            generate_lora_params(lora_dict, refiner=True)

        if app.preferences["sdxl_use_refiner"]:
            refiner_strength = app.preferences["sdxl_refiner_strength"]
        else:
            refiner_strength = 0.0

    # Common for both sd1.5 and sdxl
    args_list = ["--prompt", positive_prompt,
                    "--negative_prompt", negative_prompt,
                    "--H", image_height,
                    "--W", image_width,
                    "--sampler", sampler,
                    "--sampling_steps", sampling_steps,
                    "--scale", unconditional_guidance_scale,
                    "--clip_skip", clip_skip,
                    "--seed", seed,
                    "--n_samples", batch_size,   # Batch size: 3 for 768x512 will result in OOM. 2 works
                    "--n_iter", number_of_batches,
                    "--ckpt", ldm_path,
                    "--embedding_path", embedding_path,
                    "--vae_ckpt", vae_path,
                    "--hires_fix_upscaler", hires_fix_upscaler,
                    "--hires_fix_scale_factor", str(hires_fix_scale_factor),
                    "--control_models", control_models,
                    "--control_weights", control_weights,
                    "--control_image_path", app.control_net_image_file_path,
                    "--lora_models", lora_models,
                    "--lora_weights", lora_weights,
                    "--outdir", app.output_dir,
                    "--embedding_images_dir", app.embedding_images_dir,
                    "--strength", denoising_strength]
    if app.preferences["safety_check"]:
        args_list.append("--safety_check")
    if app.preferences["watermark"]:
        args_list.append("--watermark")
    if auto_face_fix:
        args_list.append("--auto_face_fix")

    # FaceID
    if app.face_input_image_original_size and app.disable_face_input_checkbox.get_active() is False:
        face_input_image_path = app.face_input_image_file_path

        args_list += [
            "--face_input_img", face_input_image_path,
            "--face_model", join_directory_and_file_name(
                app.preferences["control_model_path"], FACE_MODEL_NAME),
            "--face_strength", str(app.preferences["face_strength"])
        ]

    if generator_model_type == "SD 1.5":
        if app.generation_mode in (MODE_IMAGE_TO_IMAGE, MODE_INPAINTING):
            # Save current image_input to a file
            # Note that we are saving the full-size image here.
            # Resize to match the noisy image resolution is done
            # in the generation prep code.
            input_image_path = os.path.join(app.tmp_dir, "input_image.png")
            app.input_image_original_size.save(input_image_path)

            args_list += [
                "--init-img", input_image_path,
            ]
            generate_func = img2img_parse_options_and_generate

            if app.generation_mode == MODE_INPAINTING:
                # Save current image_input to a file
                args_list += [
                    "--mask-img", app.mask_image_path,
                    "--inpaint_ckpt", ldm_inpaint_path
                ]
                generate_func = inpaint_parse_options_and_generate
        else:  # txt2img
            generate_func = parse_options_and_generate

        # Override args_list if override checkbox is checked
        if app.override_checkbox.get_active():
            info = text_view_get_text(app.generation_information)
            logger.info(f"Using the generation settings from the image instead of UI")
            args_list = override_args_list(args_list, info, app.preferences)
        else:
            logger.debug("Not overriding the generation setting")

        update_prompt_history(
            positive_prompt_before_expansion,
            negative_prompt_before_expansion,
            positive_prompt_expansion,
            negative_prompt_expansion,
            positive_prompt_pre_expansion,
            negative_prompt_pre_expansion)

        status_queue = queue.Queue()
        # Start the image generation thread
        thread = threading.Thread(
            target=generate_func,
            kwargs={'args': args_list,
                    'ui_thread_instance': app,
                    'status_queue': status_queue})
    # end if sd 1.5

    elif generator_model_type == "SDXL":

        from sdxl.sdxl_pipeline.sdxl_image_generator import parse_options_and_generate as sdxl_parse_options_and_generate

        if app.generation_mode in (MODE_IMAGE_TO_IMAGE, MODE_INPAINTING):
            # Save current image_input to a file
            # Note that we are saving the full-size image here.
            # Resize to match the noisy image resolution is done
            # in the generation prep code.
            input_image_path = os.path.join(app.tmp_dir, "input_image.png")
            app.input_image_original_size.save(input_image_path)

            args_list += [
                "--init-img", input_image_path,
            ]
            generation_type = "img2img"

            if app.generation_mode == MODE_INPAINTING:
                # Save current image_input to a file
                args_list += [
                    "--mask-img", app.mask_image_path,
                    "--inpaint_ckpt", ldm_inpaint_path
                ]
                generation_type = "inpaint"
        else:  # txt2img
            generation_type = "txt2img"
        
        args_list += [
            "--refiner_strength", str(refiner_strength),
            "--refiner_sdxl_ckpt", refiner_ldm_path,
            "--refiner_sdxl_vae_ckpt", refiner_vae_path,
            "--refiner_sdxl_lora_models", refiner_sdxl_lora_models,
            "--refiner_sdxl_lora_weights", refiner_sdxl_lora_weights,

            "--discretization", app.preferences["discretization"],
            "--discretization_sigma_min", str(app.preferences["discretization_sigma_min"]),
            "--discretization_sigma_max", str(app.preferences["discretization_sigma_max"]),
            "--discretization_rho", str(app.preferences["discretization_rho"]),
            "--guider", app.preferences["guider"],
            "--linear_prediction_guider_min_scale", str(app.preferences["linear_prediction_guider_min_scale"]),
            "--linear_prediction_guider_max_scale", str(app.preferences["linear_prediction_guider_max_scale"]),
            "--triangle_prediction_guider_min_scale", str(app.preferences["triangle_prediction_guider_min_scale"]),
            "--triangle_prediction_guider_max_scale", str(app.preferences["triangle_prediction_guider_max_scale"]),

            "--sampler_s_churn", str(app.preferences["sampler_s_churn"]),
            "--sampler_s_tmin", str(app.preferences["sampler_s_tmin"]),
            "--sampler_s_tmax", str(app.preferences["sampler_s_tmax"]),
            "--sampler_s_noise", str(app.preferences["sampler_s_noise"]),
            "--sampler_eta", str(app.preferences["sampler_eta"]),
            "--sampler_order", str(app.preferences["sampler_order"])
        ]

        generate_func = sdxl_parse_options_and_generate

        # Override args_list if override checkbox is checked
        if app.override_checkbox.get_active():
            info = text_view_get_text(app.generation_information)
            logger.info(f"Using the generation settings from the image instead of UI")
            args_list = override_args_list(args_list, info, app.preferences,
                                           sdxl=True)
        else:
            logger.debug("Not overriding the generation setting")

        update_prompt_history(
            positive_prompt_before_expansion,
            negative_prompt_before_expansion,
            positive_prompt_expansion,
            negative_prompt_expansion,
            positive_prompt_pre_expansion,
            negative_prompt_pre_expansion)

        status_queue = queue.Queue()
        # Start the image generation thread
        thread = threading.Thread(
            target=generate_func,
            kwargs={'args': args_list,
                    'generation_type': generation_type,
                    'ui_thread_instance': app,
                    'status_queue': status_queue})
    # end if sdxl

    thread.start()

    GLib.timeout_add(100, update_ui, app, status_queue)


def update_ui(app, status_queue) -> bool:
    """
    Updates the main UI with image generation status.
    This method is invoked during each iteration in sampling.

    Args:
        app (Gtk.Window): The main application.
        status_queue (queue): The queue to receive status update from the generation thread.
    """
    try:
        message = status_queue.get_nowait()
        # app_window.set_title(message)
        # print(message)
        GLib.idle_add(app.generation_status.set_text, message)

        if message == "Done":
            return False
    except queue.Empty:
        pass
    return True