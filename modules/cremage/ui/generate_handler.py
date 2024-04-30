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

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [PROJECT_ROOT, MODULE_ROOT] + sys.path
from txt2img import parse_options_and_generate
from img2img import img2img_parse_options_and_generate
from inpaint import inpaint_parse_options_and_generate
from cremage.const.const import *
from cremage.utils.gtk_utils import text_view_get_text
from cremage.utils.misc_utils import override_args_list, generate_lora_params
from cremage.utils.misc_utils import join_directory_and_file_name
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
    logger.debug(positive_prompt)
    if app.preferences["enable_positive_prompt_expansion"]:
        positive_prompt += app.preferences["positive_prompt_expansion"]
        logger.debug(f"Positive prompt after expansion: {positive_prompt}")

    textbuffer = app.negative_prompt.get_buffer()
    negative_prompt = textbuffer.get_text(textbuffer.get_start_iter(), textbuffer.get_end_iter(), False)
    logger.debug(negative_prompt)
    if app.preferences["enable_negative_prompt_expansion"]:
        negative_prompt += app.preferences["negative_prompt_expansion"]
        logger.debug(f"Negative prompt after expansion: {negative_prompt}")

    # Read generation preference from UI
    sampler = app.preferences["sampler"]
    sampling_steps = str(app.preferences["sampling_steps"])
    unconditional_guidance_scale = str(app.preferences["cfg"])
    image_height = str(app.preferences["image_height"])
    image_width = str(app.preferences["image_width"])
    clip_skip = str(app.preferences["clip_skip"])
    seed = str(app.preferences["seed"])
    batch_size = str(app.preferences["batch_size"])
    number_of_batches = str(app.preferences["number_of_batches"])
    denoising_strength = str(app.preferences["denoising_strength"])  # hires-fix and img2img
    ldm_path = join_directory_and_file_name(app.preferences["ldm_model_path"], app.preferences["ldm_model"])
    ldm_inpaint_path = join_directory_and_file_name(app.preferences["ldm_model_path"], app.preferences["ldm_inpaint_model"])
    vae_path = join_directory_and_file_name(app.preferences["vae_model_path"], app.preferences["vae_model"])
    control_model_path = join_directory_and_file_name(app.preferences["control_model_path"], app.preferences["control_model"])
    auto_face_fix = app.preferences["auto_face_fix"]

    control_models = control_model_path  # FIXME. Support multiple ControlNet
    control_weights = str(1.0)  # FIXME. Support weight

    hires_fix_upscaler = app.preferences["hires_fix_upscaler"]

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

    args_list = ["--prompt", positive_prompt,
                    "--negative_prompt", negative_prompt,
                    "--safety_check", app.preferences["safety_check"],
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
                    "--embedding_path", app.preferences["embedding_path"],
                    "--vae_ckpt", vae_path,
                    "--hires_fix_upscaler", hires_fix_upscaler,
                    "--control_models", control_models,
                    "--control_weights", control_weights,
                    "--control_image_path", app.control_net_image_file_path,
                    "--lora_models", lora_models,
                    "--lora_weights", lora_weights,
                    "--outdir", app.output_dir,
                    "--strength", denoising_strength]
    if auto_face_fix:
        args_list.append("--auto_face_fix")
        
    # FaceID
    if app.face_input_image_original_size and app.disable_face_input_checkbox.get_active() is False:
        face_input_image_path = app.face_input_image_file_path

        args_list += [
            "--face_input_img", face_input_image_path,
            "--face_model", join_directory_and_file_name(
                app.preferences["control_model_path"], FACE_MODEL_NAME)
        ]

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
    else:
        generate_func = parse_options_and_generate

    # Override args_list if override checkbox is checked
    if app.override_checkbox.get_active():
        info = text_view_get_text(app.generation_information)
        logger.info(f"Using the generation settings from the image instead of UI")
        args_list = override_args_list(args_list, info, app.preferences)
    else:
        logger.debug("Not overriding the generation setting")

    status_queue = queue.Queue()
    # Start the image generation thread
    thread = threading.Thread(
        target=generate_func,
        kwargs={'args': args_list,
                'ui_thread_instance': app,
                'status_queue': status_queue})
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