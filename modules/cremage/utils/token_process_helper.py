"""
Copyright (c) 2024 Hideyuki Inada.
"""
import os
import sys
import re
import logging
from typing import List, Tuple

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def split_token_with_embedding_tags(s: str)-> Tuple[List[str], List[bool]]:
    """
    Splits a given string into tokens, identifying and tagging embedding references. 
    Each token is checked for an embedding tag, marked by angle brackets (e.g., "<embedding:file.bin>"), 
    and split accordingly. The function returns two lists: one with the split tokens and 
    another with boolean values indicating whether each corresponding token is an embedding reference.

    The method assumes that tokens do not contain whitespace and treats any sequence within "<>"
    as a potential embedding tag, provided it follows immediately after the opening bracket without whitespace.

    Args:
        s (str): The string to be split into tokens.

    Returns:
        Tuple[List[str], List[bool]]: A tuple containing two lists:
            - A list of string tokens split from the input.
            - A list of boolean flags corresponding to each token, indicating whether the token is an embedding tag.

    Examples:
        >>> split_token_with_embedding_tags("abc<embedding:hello.bin>xyz")
        (["abc", "<embedding:hello.bin>", "xyz"], [False, True, False])

    Note:
        - The function is designed to parse and identify embedding tags that are enclosed in "<>".
        - A token is defined as a sequence of characters without any whitespace. Any whitespace should be processed as a token delimiter before calling this function.
    """
    retval_str = list()
    retval_bool = list()

    current_word = ""
    i = 0
    word_len = len(s)
    while True:
        if i >= word_len:
            if len(current_word) > 0:
                retval_str.append(current_word)
                retval_bool.append(False)
            break

        c = s[i]
        if c != "<":
            current_word += c
            i += 1
        else:
            # scan
            rbracket_pos = s[i:].find(">")
            if rbracket_pos > len("embedding:"): # found
                embedding_tag = s[i:i+rbracket_pos+1]
                if len(current_word) > 0:
                    retval_str.append(current_word)
                    retval_bool.append(False)
                retval_str.append(embedding_tag)
                retval_bool.append(True)                
                current_word = ""
                i += len(embedding_tag)
            else:
                current_word += c
                i += 1
            
    return retval_str, retval_bool
