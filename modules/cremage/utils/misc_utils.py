"""
Miscellaneous utility functions
"""
import os
import sys
import platform
import subprocess
import time
import logging
import shutil
from typing import List, Dict, Any, Tuple
import json

TMP_DIR = os.path.join(os.path.expanduser("~"), ".cremage", "tmp")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def open_os_directory(directory_path:str) -> None:
    """
    Opens the OS directory using the native file viewer.

    Args:
        directory_path (str): The directory to open.
    """
    # Detect the operating system
    os_name = platform.system()

    # Open the directory using the appropriate command
    try:
        if os_name == 'Windows':
            subprocess.run(['explorer', directory_path], check=True)
        elif os_name == 'Darwin':  # macOS
            subprocess.run(['open', directory_path], check=True)
        elif os_name == 'Linux':
            subprocess.run(['xdg-open', directory_path], check=True)
        else:
            print(f"Unsupported OS: {os_name}")
    except Exception as e:
        print(f"Failed to open directory: {e}")


def get_tmp_dir():
    if os.path.exists(TMP_DIR) is False:
        os.makedirs(TMP_DIR)
    return TMP_DIR


def clean_tmp_dir():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
        logger.info("fCleaned up {TMP_DIR}")


def get_tmp_image_file_name(extension=".png"):
    return str(time.time()) + "_processed.png"


def join_directory_and_file_name(dir_path: str, file_name: str):
    """
    Returns the full path from the input directory and the file name.

    Args:
        dir_path (str): The full path of the directory.
        file_name (str): The file name that can contain "None".
    Returns:
        "None" if the file name is "None", empty or None.
    """
    return "None" if file_name == "None" or file_name == "" or file_name is None else os.path.join(dir_path, file_name)  



def str_to_detected_data_type(s: str):
    """
    Converts a string to its detected data type.
    
    The function attempts to convert the input string to a boolean, integer, or float,
    in that order, based on its content. If none of these conversions are applicable,
    the input string is returned as is.
    
    Args:
        s (str): The input string to be converted.

    Returns:
        bool, int, float, or str: The converted value of the input string into its 
        detected data type, or the original string if no conversion is applicable.
        
    Examples:
        >>> str_to_detected_data_type("True")
        True
        >>> str_to_detected_data_type("3.14")
        3.14
        >>> str_to_detected_data_type("100")
        100
        >>> str_to_detected_data_type("example")
        "example"
    """
    # Directly return the Boolean values if applicable
    if s.upper() == "TRUE":
        return True
    elif s.upper() == "FALSE":
        return False
    
    # Attempt to convert the string to a float or int
    try:
        output = float(s)
        # If the string representation of the number doesn't contain a decimal point,
        # it is more accurately represented as an integer
        if "." not in s:
            output = int(output)
        return output
    except ValueError:
        # Return the original string if it cannot be converted to a number
        return s

def override_args_list(args_list: List[str],
                       generation_string: str,
                       preferences_dict: Dict[str, Any]):
    """

    Example dict reconstructed from generation_string:
    {
        "time": 1711483882.8759885,
        "positive_prompt": "cute puppy",
        "negative_prompt": "scary",
        "ldm_model": "foo.safetensors",
        "vae_model": "vae-ft-mse-840000-ema-pruned.ckpt", 
        "sampler": "DDIM", 
        "sampling_iterations": 50, 
        "image_height": 768, 
        "image_width": 512, 
        "clip_skip": 1,
        "seed": 1462286278}
    """
    # TODO: Support other parameters. 
    #   1st priority: height, width, vae, 
    #   2nd: seed, 
    #   3rd: sampler, sampling iterations
    retval = args_list

    try:
        print(generation_string)
        override_dict = json.loads(generation_string)

    except:
        logging.warn("Failed to parse generation_string. Override ignored")
        return retval

    retval = args_list.copy()

    for i in range(int(len(args_list) / 2)) :
        j = i * 2
        if args_list[j] == "--prompt" and "positive_prompt" in override_dict:
            retval[j+1] = override_dict["positive_prompt"]
        if args_list[j] == "--negative_prompt" and "negative_prompt" in override_dict:
            retval[j+1] = override_dict["negative_prompt"]
        if args_list[j] == "--clip_skip" and "clip_skip" in override_dict:
            retval[j+1] = str(override_dict["clip_skip"])
        if args_list[j] == "--ckpt" and "ldm_model" in override_dict:
            retval[j+1] = os.path.join(
                preferences_dict["ldm_model_path"], override_dict["ldm_model"])
        if args_list[j] == "--lora_models" and "lora_models" in override_dict:
            # Convert each relative path to full path
            if override_dict["lora_models"]:
                l = override_dict["lora_models"].split(",")
                l = [os.path.join(
                    preferences_dict["lora_model_path"], e.strip()) for e in l if len(e.strip()) > 0]
                l = ",".join(l)
            else:
                l = ""
            retval[j+1] = l
        if args_list[j] == "--lora_weights" and "lora_weights" in override_dict:
            retval[j+1] = override_dict["lora_weights"]

    return retval


def generate_lora_params(preferences: Dict[str, Any]) -> Tuple[str, str]:
    """
    Generate lora_models and lora_weights string parameters from preferences dictionary.
    """
    # Comma-separated LoRA paths and weight list
    lora_model_params = [
        "lora_model_1",
        "lora_model_2",
        "lora_model_3",
        "lora_model_4",
        "lora_model_5",
    ]
    lora_weight_params = [
        "lora_weight_1",
        "lora_weight_2",
        "lora_weight_3",
        "lora_weight_4",
        "lora_weight_5",
    ]
    lora_models = []        
    lora_weights = []        
    for e in lora_model_params:
        if preferences[e] is not None and preferences[e] != "None":
            model_path = os.path.join(
                preferences["lora_model_path"],
                preferences[e])
            print(model_path)
        else:
            model_path = ""
        lora_models.append(model_path)

    for e in lora_weight_params:
        lora_weights.append(
            str(preferences[e]) if preferences[e] else "")

    lora_models = ",".join(lora_models)
    lora_weights = ",".join(lora_weights)

    return (lora_models, lora_weights)
