"""
Miscellaneous utility functions
"""
import os
import sys
import re
import argparse
import time
import platform
import subprocess
import time
import logging
import shutil
from typing import List, Dict, Any, Tuple
import json
import psutil

TMP_DIR = os.path.join(os.path.expanduser("~"), ".cremage", "tmp")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def extract_embedding_filenames(prompt:str) -> List[str]:
    """
    Extracts one or more embedding file names from a prompt.

    Args:
        prompt(str): A positive or negative prompt.
    Returns:
        A list of embedding file names. If no embedding file name is detected,
        an empty list is returned.
    """
    if prompt is None:
        return []
    
    # Define the regex pattern for the special token <embedding:embedding_file_name>
    # The file name is a sequence of any characters except ">"
    pattern = r'<embedding:([^>]+)>'

    # Find all matches of the pattern in the prompt
    matches = re.findall(pattern, prompt)
    
    return matches


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


def get_tmp_file(extension:str="") -> str:
    """
    Returns a temporary file name.

    Args:
        extention(str): The optional file extension.
    Returns:
        A temporary file name in a temporary directory.
    """
    file_name = str(time.time()) + extension
    return os.path.join(get_tmp_dir(), file_name)


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

def override_options(opt: argparse.Namespace,
                       generation_string: str,
                       preferences_dict: Dict[str, Any],
                       sdxl=False):
    """
    Override options object with previous generation parameters.
    """
    parsed_args = opt

    # Deal with a string where the user decided to add a new line
    # in text view.
    # In this case, generation string contains \n as opposed to \\n
    # which unedited generation string contains.
    # A single backslash \n causes JSON parse error, so you need
    # to add the second backslash.
    # To do so, we first define the regex pattern to match \n not preceded by \
    # The pattern uses a negative lookbehind assertion to ensure that the \n is not part of \\n
    # (?<!...) is the syntax for negative lookbehind
    # (?<!\\) ensures that the position is not preceded by a single backslash
    pattern = r'(?<!\\)\n'

    # Then replace \n with \\n using the regex pattern
    generation_string = re.sub(pattern, r'\\n', generation_string)

    try:
        print(generation_string)
        override_dict = json.loads(generation_string)
    except:
        logging.warn("Failed to parse generation_string. Override ignored")
        return opt

    for arg, value in vars(parsed_args).items():
        if arg == "prompt" and "positive_prompt" in override_dict:
              setattr(parsed_args, arg, override_dict["positive_prompt"])
        if arg == "negative_prompt" and "negative_prompt" and "negative_prompt" in override_dict:
            setattr(parsed_args, arg, override_dict["negative_prompt"])
        if arg == "clip_skip" and "clip_skip" in override_dict:
            setattr(parsed_args, arg, int(override_dict["clip_skip"]))
        if arg == "scale" and "cfg" in override_dict:
            setattr(parsed_args, arg, float(override_dict["cfg"]))
        if arg == "sampler" and "sampler" in override_dict:
            sampler = str(override_dict["sampler"])
            if sdxl:
                sampler += "Sampler"
            setattr(parsed_args, arg, sampler)
        if arg == "sampling_steps" and "sampling_iterations" in override_dict:
            setattr(parsed_args, arg, int(override_dict["sampling_iterations"]))
        if arg == "H" and "image_height" in override_dict:
            setattr(parsed_args, arg, int(override_dict["image_height"]))
        if arg == "W" and "image_width" in override_dict:
            setattr(parsed_args, arg, int(override_dict["image_width"]))
        if arg == "ckpt" and "ldm_model" in override_dict:
            setattr(parsed_args, arg, os.path.join(
                preferences_dict["ldm_model_path"], override_dict["ldm_model"]))
        if arg == "vae_ckpt" and "vae_model" in override_dict:
            setattr(parsed_args, arg, os.path.join(
                preferences_dict["vae_model_path"], override_dict["vae_model"]))
        if arg == "lora_models" and "lora_models" in override_dict:
            # Convert each relative path to full path
            if sdxl:
                dir_key = "sdxl_lora_model_path"
            else:
                dir_key = "lora_model_path"
            if override_dict["lora_models"]:
                l = override_dict["lora_models"].split(",")
                l2 = list()
                for e in l:
                    if len(e.strip()) > 0:
                        e = os.path.join(preferences_dict[dir_key], e.strip())
                    else:
                        e = ""
                    l2.append(e)
                l = ",".join(l2)
            else:
                l = ""
            setattr(parsed_args, arg, l)
        if arg == "lora_weights" and "lora_weights" in override_dict:
            setattr(parsed_args, arg, override_dict["lora_weights"])

        # sdxl refiner
        if arg == "refiner_sdxl_ckpt" and "refiner_ldm_model" in override_dict:
            setattr(parsed_args, arg, os.path.join(
                preferences_dict["sdxl_ldm_model_path"], override_dict["refiner_ldm_model"]))

        if arg == "refiner_sdxl_vae_ckpt" and "refiner_vae_model" in override_dict:
            setattr(parsed_args, arg, os.path.join(
                preferences_dict["sdxl_vae_model_path"], override_dict["refiner_vae_model"]))

        if arg == "refiner_sdxl_lora_models" and "refiner_lora_models" in override_dict:
            # Convert each relative path to full path
            if override_dict["refiner_lora_models"]:
                l = override_dict["refiner_lora_models"].split(",")
                l2 = list()
                dir_key = "sdxl_lora_model_path"
                for e in l:
                    if len(e.strip()) > 0:
                        e = os.path.join(preferences_dict[dir_key], e.strip())
                    else:
                        e = ""
                    l2.append(e)
                l = ",".join(l2)
            else:
                l = ""
            setattr(parsed_args, arg, l)

        if arg == "refiner_sdxl_lora_weights" and "refiner_lora_weights" in override_dict:
            setattr(parsed_args, arg, override_dict["refiner_lora_weights"])

        if arg == "refiner_strength" and "refiner_strength" in override_dict:
            setattr(parsed_args, arg, float(override_dict["refiner_strength"]))

        if arg == "hires_fix_upscaler" and "hires_fix_upscaler" in override_dict:
            setattr(parsed_args, arg, str(override_dict["hires_fix_upscaler"]))

        if arg == "hires_fix_scale_factor" and "hires_fix_scale_factor" in override_dict:
            setattr(parsed_args, arg, float(override_dict["hires_fix_scale_factor"]))

        # Not used for now
        # if arg == "low_mem":
        #     if "low_mem" in override_dict:
        #         setattr(parsed_args, arg, True)
        #     else:
        #         setattr(parsed_args, arg, False)

        # if arg == "keep_instance":
        #     if "keep_instance" in override_dict:
        #         setattr(parsed_args, arg, True)
        #     else:
        #         setattr(parsed_args, arg, False)

        if arg == "auto_face_fix":
            if "auto_face_fix" in override_dict:
                setattr(parsed_args, arg, True)
            else:
                setattr(parsed_args, arg, False)


        if arg == "auto_face_fix_strength" and "auto_face_fix_strength" in override_dict:
            setattr(parsed_args, arg, float(override_dict["auto_face_fix_strength"]))

        if arg == "auto_face_fix_prompt" and "auto_face_fix_prompt" in override_dict:
            setattr(parsed_args, arg, str(override_dict["auto_face_fix_prompt"]))

        if arg == "auto_face_fix_face_detection_method" and "auto_face_fix_face_detection_method" in override_dict:
            setattr(parsed_args, arg, str(override_dict["auto_face_fix_face_detection_method"]))

    return parsed_args


def override_kwargs(kwargs: Dict[str, Any],
                       generation_string: str,
                       preferences:Dict=None,
                       generator_model_type:str=None):
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
    skip_keys = ["seed",  # Do not copy seed
                 "time",
                 "generator_model_type",
                 "additional_processing"]

    change_keys = {
        "sd3_ldm_model_path": "checkpoint",
        "ldm_model": "checkpoint",
        'image_height': "height",
        "image_width": "width",
        "sampling_iterations": "steps",
        "cfg": "guidance_scale"
    }

    # Post-conversion keys
    cast_keys = {
        'height': int,
        "width": int
    }

    retval = kwargs.copy()
    try:
        print(generation_string)
        override_dict = json.loads(generation_string)
    except:
        logging.warn("Failed to parse generation_string. Override ignored")
        return retval

    for k, v in override_dict.items():
        if k in change_keys:
            k = change_keys[k]

        if k in cast_keys:  # e.g. height datatype change from "896" to 896
            v = cast_keys[k](v)

        if k not in skip_keys:
            # TODO: Clean this up
            if k == "checkpoint" and generator_model_type == "Pixart Sigma":
                retval[k] = join_directory_and_file_name(
                    preferences["pixart_sigma_ldm_model_path"],
                    v)
            else:
                retval[k] = v

    return retval

def generate_lora_params(preferences: Dict[str, Any],
                         sdxl=False,
                         refiner=False) -> Tuple[str, str]:
    """
    Generate lora_models and lora_weights string parameters from preferences dictionary.
    """
    lora_model_path_key = "lora_model_path"

    # Comma-separated LoRA paths and weight list
    lora_model_params = [
        "lora_model_1",
        "lora_model_2",
        "lora_model_3",
        "lora_model_4",
        "lora_model_5",
    ]
    prefix = ""
    if sdxl:
        prefix = "sdxl_"
        lora_model_path_key = "sdxl_" + lora_model_path_key
    if refiner:
        prefix = "refiner_sdxl_"
        lora_model_path_key = "sdxl_" + lora_model_path_key

    if prefix:
        lora_model_params = [prefix + e for e in lora_model_params]

    lora_weight_params = [
        "lora_weight_1",
        "lora_weight_2",
        "lora_weight_3",
        "lora_weight_4",
        "lora_weight_5",
    ]

    lora_weight_params = [prefix + e for e in lora_weight_params]

    lora_models = []        
    lora_weights = []        
    for e in lora_model_params:
        if preferences[e] is not None and preferences[e] != "None":
            model_path = os.path.join(
                preferences[lora_model_path_key],
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

import os

def strip_directory_from_path_list_str(path_str: str, delimiter: str = ",") -> str:
    """
    Strips directory components from a delimited list of file paths.

    Args:
        path_str (str): The list of full paths delimited by a specified delimiter.
        delimiter (str): The delimiter used to separate the paths in the input string. Default is a comma.

    Returns:
        str: The list of base paths delimited by the specified delimiter.

    Example:
        "/foo/bar.safesensors,baz.safesensors"
        will become
        "bar.safesensors,baz.safesensors"
    """
    def safe_base_name(p):
        if p is None or p == "":
            return p
        else:
            return os.path.basename(p)

    if not path_str:
        return path_str

    paths = path_str.split(delimiter)
    base_paths = [safe_base_name(path) for path in paths]
    return delimiter.join(base_paths)


def get_memory_usage_in_mb():

    # Get the current process ID
    process = psutil.Process(os.getpid())

    # Get the memory usage in bytes
    memory_usage = process.memory_info().rss  # Resident Set Size (RSS)

    # Convert bytes to MB
    memory_usage_in_mb = memory_usage / (1024 * 1024)

    return memory_usage_in_mb




