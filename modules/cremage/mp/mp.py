import os
import logging
import sys
import multiprocessing
import threading
import time

from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [PROJECT_ROOT, MODULE_ROOT] + sys.path

from sd.txt2img import generate as sd15_txt2img_generate
from sd.img2img import generate as sd15_img2img_generate
from sd.inpaint import generate as sd15_inpaint_generate
from sdxl.sdxl_pipeline.sdxl_inpaint import generate as sdxl_inpaint_generate
from sdxl.sdxl_pipeline.sdxl_image_generator import generate as sdxl_generate
from cremage.utils.image_utils import deserialize_pil_image

from cremage.const.const import *

MP_MESSAGE_TYPE_EXIT = "exit"
MP_MESSAGE_TYPE_INFERENCE = "inference"

ui_to_ml_queue = None
ml_to_ui_queue = None

def ml_process(ui_to_ml_queue, ml_to_ui_queue):
    while True:
        try:
            message = ui_to_ml_queue.get_nowait()
            message_type = message["type"]
            if message_type == MP_MESSAGE_TYPE_EXIT :
                print("received exit command")
                break

            elif message_type == MP_MESSAGE_TYPE_INFERENCE:
                generator_model_type = message["generator_model_type"]
                mode = message["mode"]
                kw = message["parameters"]

                if generator_model_type == GMT_SD_1_5:
                    print("received ldm command")
                    kw["status_queue"] = ml_to_ui_queue
                    if mode == MODE_TEXT_TO_IMAGE:
                        sd15_txt2img_generate(**kw)

                    elif mode == MODE_IMAGE_TO_IMAGE:
                        sd15_img2img_generate(**kw)

                    elif mode == MODE_INPAINTING:
                        sd15_inpaint_generate(**kw)

                    else:
                        raise ValueError(f"Invalid generation mode: {mode}")

                elif generator_model_type == GMT_SDXL:
                    print("received sdxl ldm command")

                    if mode == MODE_TEXT_TO_IMAGE:
                        sdxl_generate(options=kw["options"],
                                    generation_type="txt2img",
                                    ui_thread_instance=kw['ui_thread_instance'],
                                    status_queue=ml_to_ui_queue)

                    elif mode == MODE_IMAGE_TO_IMAGE:
                        sdxl_generate(options=kw["options"],
                                    generation_type="img2img",
                                    ui_thread_instance=kw['ui_thread_instance'],
                                    status_queue=ml_to_ui_queue)

                    elif mode == MODE_INPAINTING:
                        kw["options"]["status_queue"] = ml_to_ui_queue
                        sdxl_inpaint_generate(**kw["options"])

                    else:
                        raise ValueError(f"Invalid generation mode: {mode}")

                elif generator_model_type == GMT_SD_3:
                    from sd3.txt2img import generate as sd3_txt2image_generate
                    kw["options"]["status_queue"] = ml_to_ui_queue
                    sd3_txt2image_generate(**kw["options"])

                elif generator_model_type == GMT_STABLE_CASCADE:
                    from stable_cascade.txt2img import generate as stable_cascade_txt2image_generate
                    kw["options"]["status_queue"] = ml_to_ui_queue
                    stable_cascade_txt2image_generate(**kw["options"])

                elif generator_model_type == GMT_PIXART_SIGMA:
                    from pixart_sigma.txt2img import generate as pixart_sigma_txt2image_generate
                    kw["options"]["status_queue"] = ml_to_ui_queue
                    pixart_sigma_txt2image_generate(**kw["options"])

                elif generator_model_type == GMT_HUNYUAN_DIT:
                    from hunyuan_dit.txt2img import generate as hunyuan_dit_txt2image_generate
                    kw["options"]["status_queue"] = ml_to_ui_queue
                    hunyuan_dit_txt2image_generate(**kw["options"])

                elif generator_model_type == GMT_FLUX_1_SCHNELL:
                    from flux.txt2img import generate as flux_txt2image_generate
                    kw["options"]["status_queue"] = ml_to_ui_queue
                    flux_txt2image_generate(**kw["options"])

                elif generator_model_type == GMT_KANDINSKY_2_2:
                    kw["options"]["status_queue"] = ml_to_ui_queue
                    if mode == MODE_TEXT_TO_IMAGE:
                        from kandinsky.txt2img import generate as k2_2_txt2img_generate
                        k2_2_txt2img_generate(**kw["options"])
                    else:
                        kw["options"]["input_image"] = deserialize_pil_image(kw["options"]["input_image"])
                        if mode == MODE_IMAGE_TO_IMAGE:
                            from kandinsky.img2img import generate as k2_2_img2img_generate
                            k2_2_img2img_generate(**kw["options"])
                        elif mode == MODE_INPAINTING:
                            kw["options"]["mask_image"] = deserialize_pil_image(kw["options"]["mask_image"])
                            from kandinsky.inpaint import generate as k2_2_inpaint_generate
                            k2_2_inpaint_generate(**kw["options"])

            # Send a response back to the UI process
            # ml_to_ui_queue.put(f"Processed: {message}")
        except multiprocessing.queues.Empty:
            # Sleep briefly to avoid high CPU usage
            time.sleep(0.001)

def init_mp():
    global ui_to_ml_queue
    global ml_to_ui_queue

    # Queues for communication
    ui_to_ml_queue = multiprocessing.Queue()
    ml_to_ui_queue = multiprocessing.Queue()

    # Start the ML process
    ml_process_instance = multiprocessing.Process(target=ml_process, args=(ui_to_ml_queue, ml_to_ui_queue))
    ml_process_instance.start()

    return ui_to_ml_queue, ml_to_ui_queue