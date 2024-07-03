"""
Interacts with LLM to analyze images.

Copyright (C) 2024, Hideyuki Inada. All rights reserved.
"""
import os
import sys
import re
import logging
from typing import Dict, Any, Tuple

from PIL import Image
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TOOLS_ROOT = os.path.join(PROJECT_ROOT, "tools")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT, TOOLS_ROOT] + sys.path
from cremage.utils.gtk_utils import text_view_get_text, text_view_set_text, resized_gtk_image_from_pil_image
from cremage.utils.misc_utils import get_tmp_dir
from cremage.ui.image_box import ImageBox

LLM_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
MAX_PROMPT_LENGTH = 4096
NUM_PREV = 20
IMAGE_SIZE = 384

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

def extract_response(content):
    # Reverse the content string
    reversed_content = content[::-1]

    # Find the first occurrence of the reversed [/INST] tag
    # using .*? instead of .* to make it non-greedy so that it will find the first
    # TSNI instead of the last TSNI.
    match = re.search(r"(.*?)\]TSNI\/\[", reversed_content, re.DOTALL)

    if match:
        reversed_response = match.group(1)
        response = reversed_response[::-1].strip()
    else:
        response = ""
    return response


class LLMInteractor(Gtk.Window):
    def __init__(self,
                 pil_image=None,
                 save_call_back=None,
                 generation_information_call_back=None,
                 preferences=None,
                 status_queue=None):
        """
        Args:
            pil_image: An image containing one or more faces
            procedural (bool): True if called from img2img. False if invoked on the UI
        """
        # Delay import until class is instantiated
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        super().__init__(title="LLM Interacter")
        self.pil_image = pil_image
        self.save_call_back = save_call_back
        self.preferences = dict() if preferences is None else preferences
        self.generation_information_call_back = generation_information_call_back
        self.output_dir = None  # This is used for refreshing the image list, but not used for Face Fixer

        self.llm_model = LlavaNextForConditionalGeneration.from_pretrained(
            LLM_MODEL_ID,
            load_in_4bit=True,
            local_files_only=local_files_only_value)
        self.llm_processor = LlavaNextProcessor.from_pretrained(
            LLM_MODEL_ID,
            local_files_only=local_files_only_value)

        self.status_queue = status_queue

        # Create an Image widget
        if pil_image is None:
            pil_image = Image.new('RGBA', (IMAGE_SIZE, IMAGE_SIZE), "gray")
        self.pil_image = pil_image

        self.generation_information = dict()
        if self.generation_information_call_back is not None:
            d = self.generation_information_call_back()
            if d:  # The original image may not have generation info
                self.generation_information = d

        # UI and menu definition
        self.set_border_width(10)

        # Create a vertical Gtk.Box
        root_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(root_box)

        # Create a MenuBar
        ## Accelerator
        accel_group = Gtk.AccelGroup()
        self.add_accel_group(accel_group)

        menubar = Gtk.MenuBar()

        # File menu items
        filemenu = Gtk.Menu()
        file_item = Gtk.MenuItem(label="File")
        file_item.set_submenu(filemenu)

        # File | Exit
        exit_item = Gtk.MenuItem(label="Exit")
        exit_item.connect("activate", Gtk.main_quit)
        filemenu.append(exit_item)
        menubar.append(file_item)
        root_box.pack_start(menubar, False, False, 0)

        # Main UI definition
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        root_box.pack_start(vbox, True, True, 0)

        # Image
        self.gtk_image_box = ImageBox(
                                self.pil_image,
                                target_width=IMAGE_SIZE,
                                target_height=IMAGE_SIZE,
                                callback=lambda e: setattr(self, 'pil_image', e))

        vbox.pack_start(self.gtk_image_box, False, True, 0)

        self.use_image_check_box = Gtk.CheckButton(label="Add image to input for LLM")
        self.use_image_check_box.set_active(True)

        vbox.pack_start(self.use_image_check_box, False, True, 0)

        # ListBox for chat history with a ScrolledWindow
        self.conversation = Gtk.ListBox()
        self.scrolled_window = Gtk.ScrolledWindow()
        vbox.pack_start(self.scrolled_window, True, True, 0)
        self.scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.scrolled_window.add(self.conversation)

        # Box to wrap the prompt and the send button
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        vbox.pack_start(self.hbox, True, True, 0)

        self.scrolled_window_prompt = Gtk.ScrolledWindow()
        self.hbox.pack_start(self.scrolled_window_prompt, True, True, 0)
        self.scrolled_window_prompt.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        # User prompt
        user_input_frame = Gtk.Frame()
        self.scrolled_window_prompt.add(user_input_frame)
        user_input_frame.set_shadow_type(Gtk.ShadowType.IN)  # Gives the impression of an inset (bordered) widget
        user_input_frame.set_border_width(1)  # Adjust the user_input_frame border width if necessary
        self.user_input = Gtk.TextView()
        self.user_input.set_wrap_mode(Gtk.WrapMode.WORD)
        self.user_input.set_size_request(400, 25)
        user_input_frame.add(self.user_input)
 
        # Send button
        vbox_filler_wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.hbox.pack_start(vbox_filler_wrapper, False, False, 0)
        self.send_button = Gtk.Button(label="Send")
        vbox_filler_wrapper.pack_start(self.send_button, False, False, 0)
        self.send_button.set_size_request(24, -1)
        self.send_button.connect("clicked", self.on_send_clicked)
        vbox_filler = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox_filler_wrapper.pack_start(vbox_filler, True, True, 0)

        # Clear context button
        vbox_filler_wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.hbox.pack_start(vbox_filler_wrapper, False, False, 0)
        self.clear_context_button = Gtk.Button(label="Clear context")
        vbox_filler_wrapper.pack_start(self.clear_context_button, False, False, 0)
        self.clear_context_button.set_size_request(24, -1)
        self.clear_context_button.connect("clicked", self.on_clear_context_clicked)
        vbox_filler = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox_filler_wrapper.pack_start(vbox_filler, True, True, 0)

        # Prompt history
        self.content = list()
        self.raw_response_list = list()

    def on_clear_context_clicked(self, button):
        self.content = list()
        self.raw_response_list = list()

    def on_send_clicked(self, button):
        user_prompt = text_view_get_text(self.user_input)

        if self.use_image_check_box.get_active() and self.pil_image:
            prompt = f"[INST] <image>\n{user_prompt}  [/INST]"
        else:
            prompt = f"[INST] {user_prompt}  [/INST]"

        historical_prompt_list = self.raw_response_list[-NUM_PREV:]
        if len(self.raw_response_list) >= NUM_PREV:
            self.raw_response_list.pop(0)  # remove the oldest

        historical_prompt_list.append(prompt)
        prompt = "\n\n".join(historical_prompt_list)

        # Check if the prompt is too long
        # If so, discard the old prompt(s)
        max_length = MAX_PROMPT_LENGTH  # See https://github.com/haotian-liu/LLaVA/issues/1095 for more info
        empty_list = False
        while len(prompt) > max_length:
            if len(historical_prompt_list) == 0:
                empty_list = True
                break
            dropped = historical_prompt_list.pop(0)
            print(f"Prompt length {len(prompt)} is exceeding max length {max_length}")
            print(f"Dropping: {dropped}")
            prompt = "\n\n".join(historical_prompt_list)

        if empty_list:  # truncate the current prompt as it's too long
            if self.use_image_check_box.get_active():
                prompt = f"[INST] <image>\n{user_prompt[:max_length]}  [/INST]"
            else:
                prompt = f"[INST] {user_prompt[:max_length]}  [/INST]"

        print("\n$$$ User prompt start (actual input for LLM) $$$")
        print(prompt)
        print("$$$ User prompt end $$$\n")
        content = generate_caption(self.llm_model, self.llm_processor,
                                   prompt, self.pil_image if self.use_image_check_box.get_active() else None)

        print("\n*** RAW RESPONSE start ***")
        print(content)
        print("*** RAW RESPONSE end ***\n")

        response = extract_response(content)

        print(f"\n--- Extracted last response ---")
        print(f"{response}")
        print("--- Extracted last response end ---\n")

        response_list_entry = f"[INST] {user_prompt} [/INST] " + response
        print(f"\n+++ New response_list entry start +++")
        print(f"{response_list_entry}")
        print("+++ New response list entry end +++\n")
        self.raw_response_list.append(response_list_entry)

        if self.use_image_check_box.get_active():
            self.content.append(
                {"speaker": "user", "text": user_prompt, "image": self.pil_image}
            )
        else:
            self.content.append(
                {"speaker": "user", "text": user_prompt}
            )
        self.content.append( 
            {"speaker": "agent", "text": response})

        update_list_box(self.content, self.conversation, self.scrolled_window)
        text_view_set_text(self.user_input, "")


def update_list_box(content, list_box: Gtk.ListBox, scrolled_window):
    """
    Updates the Gtk.ListBox with the given content.

    Args:
        content: List[Dict[str, str]]:  Content: List
        Each list contains a dict.
            "speaker": "agent" or "user"
            "text": text string of the speaker
            "image": image thumbnail data to be displayed to the right of the text of the user.
        
        Example:
            user: Describe a scene (image of the beach in summer)
            agent: The image shows a photo of a beach during summer.

            Then the list box shows:
                Describe a scene (image of the beach in summer) (thumbnail image here)
            The image shows a photo of a beach during summer.
    """
    # Clear the current content of the list box
    for row in list_box.get_children():
        list_box.remove(row)
    
    for item in content:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        
        # Create a label for the text
        label = Gtk.Label(label=item["text"], xalign=0)
        label.set_xalign(0 if item["speaker"] == "agent" else 1)
        label.set_line_wrap(True)  # Enable word wrap
        label.set_line_wrap_mode(Gtk.WrapMode.WORD)  # Set wrap mode to word
        label.set_selectable(True)

        # Set label color based on the speaker
        if item["speaker"] == "agent":
            label.override_color(Gtk.StateFlags.NORMAL, Gdk.RGBA(0.1, 0.1, 0.0, 1))
        else:
            label.override_color(Gtk.StateFlags.NORMAL, Gdk.RGBA(0.4, 0.5, 0.5, 1))

        if item["speaker"] == "user":
            # Right-align: text first, then image
            box.pack_start(label, True, True, 0)
            if "image" in item and item["image"]:
                image = resized_gtk_image_from_pil_image(
                                item["image"],
                                target_width=128,
                                target_height=128)

                box.pack_end(image, False, False, 0)

        else:
            # Left-align: image first, then text
            box.pack_start(label, True, True, 0)
            if "image" in item and item["image"]:
                image = resized_gtk_image_from_pil_image(
                                item["image"],
                                target_width=128,
                                target_height=128)
                box.pack_start(image, False, False, 0)
        
        # Add the box to the list box
        list_box_row = Gtk.ListBoxRow()
        list_box_row.add(box)
        list_box.add(list_box_row)
    
    # Show all widgets in the list box
    list_box.show_all()

    # Scroll to the bottom to show the last added rows
    def scroll_to_bottom():
        adjustment = scrolled_window.get_vadjustment()
        adjustment.set_value(adjustment.get_upper() - adjustment.get_page_size())
        scrolled_window.set_vadjustment(adjustment)
        return False  # Stop the idle_add

    GLib.idle_add(scroll_to_bottom)


def generate_caption(model, processor, prompt, pil_image):
    """
    Generates a caption using LLM.

    Args:
        model: LLM model instance
        processor: Tokenizer instance
        prompt: Prompt. An example:
           prompt = "[INST] <image>\nDetermine if this image was created by a professional artist or an amateur. [/INST]"
        pil_image: PIL format image
    """
    # Tokenizer
    if pil_image:
        inputs = processor(text=prompt, images=[pil_image], return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=4096)
    output = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    return output


def main():

    preferences = {
        "safety_check": False,
        "watermark": False,
        "image_width": "512",
        "image_height": "768",
        "clip_skip": "2",
        "denoising_strength": "0.5",
        "batch_size": "3",
        "number_of_batches": "1",
        "ldm_model_path": "/media/pup/ssd2/recoverable_data/sd_models/Stable-diffusion",
        "ldm_model": "analogMadness_v70.safetensors",
        "ldm_inpaint_model": "majicmixRealistic_v7-inpainting.safetensors",
        "vae_model_path": "/media/pup/ssd2/recoverable_data/sd_models/VAE",
        "vae_model": "vae-ft-mse-840000-ema-pruned.ckpt",
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
        "lora_model_path": "/media/pup/ssd2/recoverable_data/sd_models/Lora",
        "embedding_path": "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
        "positive_prompt_expansion": ", highly detailed, photorealistic, 4k, 8k, uhd, raw photo, best quality, masterpiece",
        "negative_prompt_expansion": ", drawing, 3d, worst quality, low quality, disfigured, mutated arms, mutated legs, extra legs, extra fingers, badhands",
        "enable_positive_prompt_expansion": True, # False,
        "enable_negative_prompt_expansion": True, # False,
        "seed": "0"
    }

    pil_image = Image.open("../cremage_resources/check.png")
    app = LLMInteractor(
        pil_image=pil_image,
        preferences=preferences)
    app.connect('destroy', Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()
