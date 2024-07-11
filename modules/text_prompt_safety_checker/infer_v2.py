"""
Checks safety of the text prompt.

Copyright (c) 2024 Hideyuki Inada. All rights reserved.
"""
import os
import sys
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

import torch
from transformers import BertTokenizer, BertModel
from safetensors.torch import load_file

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.model_downloader import download_model_if_not_exist

MODEL_ID = 'google-bert/bert-base-uncased'
from text_prompt_safety_checker.bert_model_v2 import BertClassifier

def download_model(model_name=None) -> str:
    """
    Downloads the text safety classifier model from the HowToSD Hugging Face repo.

    Args:
        model_name (str): Model file name

    Returns:
        Full path of the downloaded or cached model.
    """
    # Download the model if not already downloaded
    model_dir = os.path.join(MODELS_ROOT, "text_prompt_safety_checker")
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)
    model_path = download_model_if_not_exist(
        model_dir,
        "HowToSD/text_prompt_safety_checker",
        model_name,
        )
    return model_path


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a pre-trained model from a specified path and prepares it for evaluation on a given device.

    Args:
        model_path (str): The path to the model's state dictionary file.
        device (torch.device): The device on which the model should be loaded (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The model loaded with the state dictionary and moved to the specified device, ready for evaluation.
    """
    model = BertClassifier()
    sd = load_file(model_path)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    return model


def _predict(model: torch.nn.Module, tokenizer: BertTokenizer, text: str, device: torch.device) -> float:
    """
    Predicts the probability of the input text being in a certain class using a pre-trained BERT model.

    Args:
        model (torch.nn.Module): The pre-trained BERT model for classification.
        tokenizer (BertTokenizer): The tokenizer corresponding to the BERT model.
        text (str): The input text string to be classified.
        device (torch.device): The device on which the model and tensors should be loaded (e.g., 'cpu' or 'cuda').

    Returns:
        float: The predicted probability of the input text belonging to the target class.
    """
    encoding = tokenizer(
        text,
        add_special_tokens=True,  # Whether to add special tokens such as [CLS] and [SEP] to the sequence
        max_length=512,
        return_token_type_ids=False,  # Used to distinguish between different sequences. N/A for this classifier
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)  # Token IDs of the text
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        proba = model(input_ids, attention_mask)
    return proba.item()


def predict(text: str)->float:
    """
    Predicts the probability of the input text being in a certain class using a pre-trained BERT model.

    Args:
        text (str): The input text string to be classified.
    Returns:
        float: The predicted probability of the input text belonging to the target class.
    """
    tokenizer = BertTokenizer.from_pretrained(MODEL_ID, local_files_only=local_files_only_value)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = download_model("text_prompt_safety_checker.safetensors")
    model = load_model(model_path, device)
    proba = _predict(model, tokenizer, text, device)
    return proba


def main():
    test_text = "<Put target prompt to test here ...>"
    proba = predict(test_text)
    print(f'Probability of positive: {proba}')


if __name__ == "__main__":
    main()

