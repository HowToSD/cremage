"""
Prompt history management
"""
import os
import logging
import json

HISTORY_DIR = os.path.join(os.path.expanduser("~"), ".cremage", "data", "history")
POSITIVE_PROMPT_PATH = os.path.join(HISTORY_DIR, "positive_prompts.json")
NEGATIVE_PROMPT_PATH = os.path.join(HISTORY_DIR, "negative_prompts.json")
POSITIVE_PROMPT_EXPANSION_PATH = os.path.join(HISTORY_DIR, "positive_prompt_expansion.json")
NEGATIVE_PROMPT_EXPANSION_PATH = os.path.join(HISTORY_DIR, "negative_prompt_expansion.json")
POSITIVE_PROMPT_PRE_EXPANSION_PATH = os.path.join(HISTORY_DIR, "positive_prompt_pre_expansion.json")
NEGATIVE_PROMPT_PRE_EXPANSION_PATH = os.path.join(HISTORY_DIR, "negative_prompt_pre_expansion.json")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

positive_prompts_data = None
negative_prompts_data = None
positive_prompts_expansion_data = None
negative_prompts_expansion_data = None
positive_prompts_pre_expansion_data = None
negative_prompts_pre_expansion_data = None

def remove_prompt_entry(prompt_file_name, prompt):
    global positive_prompts_data
    global negative_prompts_data
    global positive_prompts_expansion_data
    global negative_prompts_expansion_data
    global positive_prompts_pre_expansion_data
    global negative_prompts_pre_expansion_data

    if prompt_file_name == POSITIVE_PROMPT_PATH:
        positive_prompts_data["prompts"].remove(prompt)
        write_positive_prompt()

    if prompt_file_name == NEGATIVE_PROMPT_PATH:
        negative_prompts_data["prompts"].remove(prompt)
        write_negative_prompt()

    if prompt_file_name == POSITIVE_PROMPT_EXPANSION_PATH:
        positive_prompts_expansion_data["prompts"].remove(prompt)
        write_positive_prompt_expansion()

    if prompt_file_name == NEGATIVE_PROMPT_EXPANSION_PATH:
        negative_prompts_expansion_data["prompts"].remove(prompt)
        write_negative_prompt_expansion()

    if prompt_file_name == POSITIVE_PROMPT_PRE_EXPANSION_PATH:
        positive_prompts_pre_expansion_data["prompts"].remove(prompt)
        write_positive_prompt_pre_expansion()

    if prompt_file_name == NEGATIVE_PROMPT_PRE_EXPANSION_PATH:
        negative_prompts_pre_expansion_data["prompts"].remove(prompt)
        write_negative_prompt_pre_expansion()



def prompt_data_for_data_path(prompt_file_name):
    """
    Maps prompt history file name to the corresponding data instance.
    """
    if prompt_file_name == POSITIVE_PROMPT_PATH:
        return positive_prompts_data
    elif prompt_file_name == NEGATIVE_PROMPT_PATH:
        return negative_prompts_data
    elif prompt_file_name == POSITIVE_PROMPT_EXPANSION_PATH:
        return positive_prompts_expansion_data
    elif prompt_file_name == NEGATIVE_PROMPT_EXPANSION_PATH:
        return negative_prompts_expansion_data
    elif prompt_file_name == POSITIVE_PROMPT_PRE_EXPANSION_PATH:
        return positive_prompts_pre_expansion_data
    elif prompt_file_name == NEGATIVE_PROMPT_PRE_EXPANSION_PATH:
        return negative_prompts_pre_expansion_data
    else:
        return None

def load_prompt_history():
    global positive_prompts_data
    global negative_prompts_data
    global positive_prompts_expansion_data
    global negative_prompts_expansion_data
    global positive_prompts_pre_expansion_data
    global negative_prompts_pre_expansion_data

    if os.path.exists(HISTORY_DIR) is False:
        os.makedirs(HISTORY_DIR)

    if positive_prompts_data is None:
        if os.path.exists(POSITIVE_PROMPT_PATH):
            with open(POSITIVE_PROMPT_PATH, "r") as f_pos:
                positive_prompts_data = json.load(f_pos)
            if positive_prompts_data is None:
                positive_prompts_data = dict()
                positive_prompts_data["prompts"] = list()
        else:
            positive_prompts_data = dict()
            positive_prompts_data["prompts"] = list()

    if negative_prompts_data is None:
        if os.path.exists(NEGATIVE_PROMPT_PATH):
            with open(NEGATIVE_PROMPT_PATH, "r") as f_neg:
                negative_prompts_data = json.load(f_neg)

            if negative_prompts_data is None:
                negative_prompts_data = dict()
                negative_prompts_data["prompts"] = list()

        else:
            negative_prompts_data = dict()
            negative_prompts_data["prompts"] = list()

    if positive_prompts_expansion_data is None:
        if os.path.exists(POSITIVE_PROMPT_EXPANSION_PATH):
            with open(POSITIVE_PROMPT_EXPANSION_PATH, "r") as f_pos:
                positive_prompts_expansion_data = json.load(f_pos)
            if positive_prompts_expansion_data is None:
                positive_prompts_expansion_data = dict()
                positive_prompts_expansion_data["prompts"] = list()
        else:
            positive_prompts_expansion_data = dict()
            positive_prompts_expansion_data["prompts"] = list()

    if negative_prompts_expansion_data is None:
        if os.path.exists(NEGATIVE_PROMPT_EXPANSION_PATH):
            with open(NEGATIVE_PROMPT_EXPANSION_PATH, "r") as f_neg:
                negative_prompts_expansion_data = json.load(f_neg)

            if negative_prompts_expansion_data is None:
                negative_prompts_expansion_data = dict()
                negative_prompts_expansion_data["prompts"] = list()

        else:
            negative_prompts_expansion_data = dict()
            negative_prompts_expansion_data["prompts"] = list()

    if positive_prompts_pre_expansion_data is None:
        if os.path.exists(POSITIVE_PROMPT_PRE_EXPANSION_PATH):
            with open(POSITIVE_PROMPT_PRE_EXPANSION_PATH, "r") as f_pos:
                positive_prompts_pre_expansion_data = json.load(f_pos)
            if positive_prompts_pre_expansion_data is None:
                positive_prompts_pre_expansion_data = dict()
                positive_prompts_pre_expansion_data["prompts"] = list()
        else:
            positive_prompts_pre_expansion_data = dict()
            positive_prompts_pre_expansion_data["prompts"] = list()

    if negative_prompts_pre_expansion_data is None:
        if os.path.exists(NEGATIVE_PROMPT_PRE_EXPANSION_PATH):
            with open(NEGATIVE_PROMPT_PRE_EXPANSION_PATH, "r") as f_neg:
                negative_prompts_pre_expansion_data = json.load(f_neg)

            if negative_prompts_pre_expansion_data is None:
                negative_prompts_pre_expansion_data = dict()
                negative_prompts_pre_expansion_data["prompts"] = list()

        else:
            negative_prompts_pre_expansion_data = dict()
            negative_prompts_pre_expansion_data["prompts"] = list()


def update_prompt_history(
        positive_prompt:str,
        negative_prompt:str,
        positive_prompt_expansion:str,
        negative_prompt_expansion:str,
        positive_prompt_pre_expansion:str,
        negative_prompt_pre_expansion:str
        ) -> None:
    """
    Updates prompt history files.

    Args:
        positive prompt (str): Positive prompt
        negative prompt (str): Negative prompt
    """
    global positive_prompts_data
    global negative_prompts_data
    global positive_prompts_expansion_data
    global negative_prompts_expansion_data
    global positive_prompts_pre_expansion_data
    global negative_prompts_pre_expansion_data

    pos = positive_prompts_data["prompts"]
    neg = negative_prompts_data["prompts"]
    
    if positive_prompt not in set(pos):  # TODO: Optimize
        pos.append(positive_prompt)
        write_positive_prompt()

    if negative_prompt not in set(neg):  # TODO: Optimize
        neg.append(negative_prompt)
        write_negative_prompt()

    pos = positive_prompts_expansion_data["prompts"]
    neg = negative_prompts_expansion_data["prompts"]
    
    if positive_prompt_expansion not in set(pos):  # TODO: Optimize
        pos.append(positive_prompt_expansion)
        write_positive_prompt_expansion()

    if negative_prompt_expansion not in set(neg):  # TODO: Optimize
        neg.append(negative_prompt_expansion)
        write_negative_prompt_expansion()

    pos = positive_prompts_pre_expansion_data["prompts"]
    neg = negative_prompts_pre_expansion_data["prompts"]
    
    if positive_prompt_pre_expansion not in set(pos):  # TODO: Optimize
        pos.append(positive_prompt_pre_expansion)
        write_positive_prompt_pre_expansion()

    if negative_prompt_pre_expansion not in set(neg):  # TODO: Optimize
        neg.append(negative_prompt_pre_expansion)
        write_negative_prompt_pre_expansion()


def write_positive_prompt():
    global positive_prompts_data
    p = positive_prompts_data["prompts"]
    with open(POSITIVE_PROMPT_PATH, "w") as f:
        positive_prompts_data["prompts"] = p
        json.dump(positive_prompts_data, f)


def write_negative_prompt():
    global negative_prompts_data
    p = negative_prompts_data["prompts"]
    with open(NEGATIVE_PROMPT_PATH, "w") as f:
        negative_prompts_data["prompts"] = p
        json.dump(negative_prompts_data, f)


def write_positive_prompt_expansion():
    global positive_prompts_expansion_data
    p = positive_prompts_expansion_data["prompts"]
    with open(POSITIVE_PROMPT_EXPANSION_PATH, "w") as f:
        positive_prompts_expansion_data["prompts"] = p
        json.dump(positive_prompts_expansion_data, f)


def write_negative_prompt_expansion():
    global negative_prompts_expansion_data
    p = negative_prompts_expansion_data["prompts"]
    with open(NEGATIVE_PROMPT_EXPANSION_PATH, "w") as f:
        negative_prompts_expansion_data["prompts"] = p
        json.dump(negative_prompts_expansion_data, f)


def write_positive_prompt_pre_expansion():
    global positive_prompts_pre_expansion_data
    p = positive_prompts_pre_expansion_data["prompts"]
    with open(POSITIVE_PROMPT_PRE_EXPANSION_PATH, "w") as f:
        positive_prompts_pre_expansion_data["prompts"] = p
        json.dump(positive_prompts_pre_expansion_data, f)


def write_negative_prompt_pre_expansion():
    global negative_prompts_pre_expansion_data
    p = negative_prompts_pre_expansion_data["prompts"]
    with open(NEGATIVE_PROMPT_PRE_EXPANSION_PATH, "w") as f:
        negative_prompts_pre_expansion_data["prompts"] = p
        json.dump(negative_prompts_pre_expansion_data, f)


# Initialize prompt data upon first load
if positive_prompts_data is None or negative_prompts_data is None:
    load_prompt_history()