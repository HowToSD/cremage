"""
Wildcards support

Copyright (c) 2024 Hideyuki Inada.
"""
import os
import sys
import random

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
DATA_ROOT  = os.path.join(PROJECT_ROOT, "data")
sys.path = [MODULE_ROOT] + sys.path

DEPTH_MAX = 50
_wildcards_dir = None
_depth = 0

def _process_file(file_name:str) -> None:
    global _wildcards_dir

    file_path = os.path.realpath(os.path.join(_wildcards_dir, file_name) + ".txt")
    if os.path.exists(file_path) is False:  # return the file name
        return f"__{file_name}__"
    with open(file_path, "r") as f:
        lines = f.read()
    lines = lines.split("\n")

    # Remove comments
    lines = list(filter(lambda e: e.strip().startswith("#") is False, lines))
    # Remove blank lines
    lines = list(filter(lambda e: len(e.strip()) > 0, lines))

    num_choices = len(lines)  # [0, num_choices-1]
    index = random.randint(0, num_choices-1)
    selected_line = lines[index]

    # This can cause a cycle, so check for that.
    # Here is such a case where __food__ was mistakenly entered into drink.txt:
    # food.txt contains: __drink__
    # drink.txt contains:  delicious __food__
    retval = _parse_char(selected_line)  
    return retval


def _parse_char(inputs:str):
    global _wildcards_dir
    global _depth

    if inputs is None:
        return None
    if len(inputs) == 0:
        return ""

    _depth += 1
    if _depth > DEPTH_MAX:
        _depth -= 1
        return inputs

    inputs_len = len(inputs)
    i = -1
    processing_token = False
    file_name = ""  # Contains current file name
    text = ""  # Contains resolved text
    while True:
        i += 1
        if i >= inputs_len:
            break
        c = inputs[i]
        if c == "_":
            if i <= inputs_len - 2 and inputs[i+1] == "_":  # Found
                if processing_token is False:
                    processing_token = True
                    i += 1
                    file_name = ""
                    continue
                else:  # processing
                    selected_word = _process_file(file_name)
                    text += selected_word
                    file_name = ""
                    processing_token = False
                    i += 1
                    continue

        if processing_token:
            file_name += c
        else:
            text += c

    _depth -= 1

    if processing_token: # incomplete
        text += "__" + file_name
    return text


def resolve_wildcards(inputs: str, wildcards_dir:str=None) -> str:
    """
    Resolves wildcards in inputs.

    Args:
        inputs (str): The input string which may or may not contain one or more wildcards.
          Wildcards are marked by two underscores before and after a word (e.g. __word__).
          The word is a base name of the file name (word.txt) under the wildcards directory.
          For example, if the wildcard is __dog__, the name of the file will be dog.txt.
          It supports nested wildcards.  Namely, if dog.txt contains another wildcard
          in the text (e.g. __dog_food__), it will be replaced with the content from 
          dog_food.txt.
    Returns:
        Resolved text.
    """
    global _wildcards_dir
    _wildcards_dir = wildcards_dir

    if wildcards_dir is None:
        raise ValueError("wildcards_dir is not specified")
    if os.path.exists(wildcards_dir) is False:
        raise ValueError(f"{wildcards_dir} does not exist")
    
    retval = _parse_char(inputs)
    return retval
