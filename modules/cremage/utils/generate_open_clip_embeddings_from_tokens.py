"""
Copyright (c) 2024 Hideyuki Inada.
"""
import os
import sys
import logging
import re
from typing import List, Tuple

import torch

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path

# from transformers import CLIPTokenizer  # HINADA Change
# from clip.modeling_clip import CLIPTextModel as ModCLIPTextModel
import open_clip  # Make sure that this is below the sys.path statement as we are using the custom open_clip
from cremage.utils.token_process_helper import split_token_with_embedding_tags
from cremage.utils.ml_utils import load_embedding

# FIXME. Consolidate definitions
BOS = 49406
EOS = 49407
PAD = 0  # Change from CLIP

def token_to_embedding(model, token_ids):
    # Cremage: nn.Module does not have device attribute
    # so you need to find out by accessing one of the parameters
    model_device = next(model.parameters()).device
    if model_device != token_ids.device:
        token_ids = token_ids.to(model_device)
    retval = model.token_embedding(token_ids)
    return retval

def convert_word_to_tokens(word:str):
    """
    Converts a single word to one or more tokens and return the token length.
    """
    # if model.device != token_ids.device:
    #     token_ids = token_ids.to(model.device)
    tokens = open_clip.tokenize(word)
    tokens = tokens.squeeze()  # Remove batch axis
    assert tokens.shape == (77,)
    index = torch.where(tokens == EOS)[0].item()  # to int index
    tokens = tokens[1:index]
    converted_length = index - 1  # e.g. for [BOS, word, EOS], EOS index = 2, so we want 1
    return tokens, converted_length


def generate_open_clip_embeddings( 
                             model, 
                             embedding_dir: str, 
                             inputs: List[Tuple[str, float]]) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Generates open_clip embeddings from the given inputs.

    Args:
        model: The open_clip instance used for generating embeddings.
        embedding_dir (str): The full path to the directory containing embedding files.
        inputs (List[Tuple[str, float]]): A list of tuples, where each tuple contains a word (str) and its corresponding score (float).

    Returns:
        Tuple of lists:
            List of torch.Tensor: A tensor of shape `[n, 768]`, where `n` is a multiple of 77, representing the CLIP embeddings.
            List of last EOS index
    """    
    EMBEDDING_MARKER = "<embedding:"
    # List of max 75 1280-D embeddings. Not padded and does not have BOS or EOS yet.
    embedding_list = list()
    current_seq_len = 0
    i = 0  # i for embedding list
    embedding_list.append(list())  # Add an empty list
    
    if len(inputs) == 1 and len(inputs[0][0]) == 0:
        # Skip computing embedding for the empty string, just compute BOS and EOS
        logger.debug("Prompt is empty. Skipping embedding computation for the empty string") 

    else:
        for e in inputs:
            word = e[0]
            score = e[1]
            if word.startswith(EMBEDDING_MARKER) and word.endswith(">"):
                if embedding_dir is None:
                    continue   # ignore
                embedding_name = word[len(EMBEDDING_MARKER):-1]
                embedding_path = os.path.join(embedding_dir, embedding_name)
                if os.path.exists(embedding_path) is False:
                    logger.warning(f"Ignoring missing {embedding_path}")
                    continue   # ignore
                embedding = load_embedding(embedding_path)
                if isinstance(embedding, dict):
                    logger.info("Loading sdxl embedding")
                    embedding = embedding["clip_g"]
                embedding = embedding.to(os.environ.get("GPU_DEVICE", "cpu"))
                converted_length = embedding.shape[0]            
            else:
                # Convert to tokens
                tokens, converted_length = convert_word_to_tokens(word)

#                 token_data = tokenizer(
#                     word,
#                     truncation=True,
#                     max_length=77,
#                     return_length=True,
#                     return_overflowing_tokens=False,
#                     return_tensors="pt")
#                 converted_length = token_data["length"][0].item()
#                 tokens = token_data["input_ids"][0][1:converted_length-1]
#                 logger.debug(f"Mapping: {word} => {tokens}")
                embedding = None

            # check if this fits in the current sequence
            new_length = current_seq_len + converted_length
            if new_length > 75:
                i += 1
                embedding_list.append(list())  # Create a blank list for new row
                current_seq_len = converted_length            
            else:
                # Convert to embeddings
                current_seq_len += converted_length 

            if embedding is None:
                # tokens = tokens.to(model.device)
                embedding = token_to_embedding(model, tokens)
            embedding = embedding * score  

            logger.debug(f"Multiplied embedding by {score}")

            embedding_list[i].append(embedding)
               
    # Now we have a list of lists, add BOS, EOS and pad
    bos_tensor = torch.tensor(BOS, dtype=torch.int64)
    eos_tensor = torch.tensor(EOS, dtype=torch.int64)
    pad_tensor = torch.tensor(PAD, dtype=torch.int64)
    
    # Convert to embeddings
    bos_embedding = token_to_embedding(model, bos_tensor).unsqueeze(dim=0)
    eos_embedding = token_to_embedding(model, eos_tensor).unsqueeze(dim=0)
    pad_embedding = token_to_embedding(model, pad_tensor).unsqueeze(dim=0)
    
    assert bos_embedding.shape == (1, 1280)
    assert eos_embedding.shape == (1, 1280)
    assert pad_embedding.shape == (1, 1280)
    
    # Compute length as each list may contain tensors of different shapes
    # e.g. [2, 1280], [1, 1280] if one word translated to multiple tokens
    # so a single row of a sequence may contain something like:
    # [1, 1280], [2, 1280], [3, 1280], [1, 1280]
    seq_len_list = list()
    for e in embedding_list:  # list of list of tensors
        i = 0
        for t in e:
            i += t.shape[0]
        seq_len_list.append(i)

    retval = list()
    eos_index_list = list()
    # Number of 77-token-long sequences
    # e.g. if 77 then 1. If 144, then 2.
    num_sequence_groups = len(embedding_list)
    for i, e in enumerate(embedding_list):  # list of list of tensors
        seq_len = seq_len_list[i]
        pad_len = 75 - seq_len
        assert pad_len >= 0
        pad_embeddings = pad_embedding.repeat(pad_len, 1)  # repeat times along each axis
        assert(pad_embeddings.shape == (pad_len, 1280))
        
        # Convert e from list of torch tensor to a tensor
        if len(e) > 0:
            e_tens = torch.concat(e, dim=0)
            assert e_tens.shape == (seq_len, 1280)

            seq = torch.concat([
                bos_embedding,  # shape = (1, 1280)
                e_tens,         # (seq_len, 1280)
                pad_embeddings, # (pad_len, 1280)
                eos_embedding   # shape = (1, 1280)
                ], dim=0)
        else: # empty prompt
            seq = torch.concat([
                bos_embedding,  # shape = (1, 1280)
                pad_embeddings, # (pad_len, 1280)
                eos_embedding   # shape = (1, 1280)
                ], dim=0)            
        last_eos_index = 1 + seq_len
        assert(seq.shape == (77, 1280))
        retval.append(seq)
        eos_index_list.append(last_eos_index)

    # retval = torch.concat(retval, axis=0)
    # assert retval[0].shape[0] == 77
    # eos_index = (num_sequence_groups - 1)* 77 + last_eos_index
    return retval, eos_index_list

# if __name__ == "__main__":

#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#     model = ModCLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")    
    
#     inputs = [
#         ('Photo', 1.0),
#         ('of', 1.0),
#         ('a', 1.0),        
#          ('dancing', 1.2),
#         ('<embedding:badhandv4.pt>', 1.0),
#         ('man', 1.1)
#     ]
 
#     retval = generate_clip_embeddings(tokenizer, model, "/media/pup/ssd2/recoverable_data/sd_models/embeddings", inputs)
#     for r in retval:
#         print(r.shape)
