"""
Copyright (c) 2024 Hideyuki Inada.
"""
import os
import sys
import logging
import re
from typing import List, Tuple

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path = [MODULE_ROOT] + sys.path

from transformers import CLIPTokenizer
from clip.modeling_clip import CLIPTextModel as ModCLIPTextModel
from cremage.utils.token_process_helper import split_token_with_embedding_tags
from cremage.utils.generate_clip_embeddings_from_tokens import generate_clip_embeddings
from cremage.utils.generate_open_clip_embeddings_from_tokens import generate_open_clip_embeddings

class Node():
    """
    A class representing a node in a tree structure, designed for storing hierarchical data with a focus on prompt analysis.

    This class encapsulates the basic elements of a node in a tree, including character storage, positional information, hierarchical relationships (children and parent nodes), and scoring attributes for analytical purposes.

    Attributes:
        characters (list of str): Stores individual characters or tokens represented by this node.
        pos (list): A list of positions or indices associated with the characters.
        children (list of Node): A list of child nodes, allowing for the representation of a tree structure.
        parent (Node, optional): The parent node of the current node, facilitating a bidirectional traversal within the tree. Initially set to None.
        score (float): An individual score associated with this node, used for analysis or evaluation purposes. Default is 1.0.
        product_score (float): The cumulative product of scores from this node and all its ancestors, representing a final score for the node. Initial value is set to 1.0, indicating no modification from ancestral scores.
    """
    def __init__(self):
        super().__init__()
        self.characters = []
        self.pos = []
        self.children = []
        self.parent = None
        self.score = 1.0
        self.product_score = 1.0


def _build_tree(text: str) -> Node:
    """
    Builds a tree structure from a given prompt, where each node represents a substring or character of the text. 
    The tree is constructed by recursively creating new node for a substring enclosed in parentheses. 

    Args:
        text (str): The input text from which the tree will be built. The text should contain characters, 
        with parentheses '(' and ')' used to denote the hierarchy and grouping of nodes.

    Returns:
        Node: The root node of the constructed tree. This node acts as the entry point to the entire tree structure 
        created from the input text.
    """
    current = Node()
    root = current

    for i, c in enumerate(text):
        if c == "(":
            child = Node()
            # Set up links
            child.parent = current
            current.children.append(child)
            current = child
        elif c == ")":
            if current.parent is None:  # User error. Prompt is missing "(" at the root level
                pass  # Already the root, so don't switch nodes
            else:
                current = current.parent
        else:
            current.characters.append(c)
            current.pos.append(i)
            
    return root


def _compute_product_score(node: Node, base_score: float) -> None:
    """
    Computes the product score of all nodes.
    A product score is a score of the node that is computed by traversing
    the tree from the root to the current node and multiply the score of
    the node on the path.

    Args:
        node (Node): The input node
        base_score (float): The product score up to the parent node of the current node

    Returns:
        None
    """
    node.product_score = base_score * node.score
    for n in node.children:
        _compute_product_score(n, node.product_score)

def _build_region(prompt: str):
    """
    Compute a tree of regions from a text prompt.

    It first builds a tree of characters where a node contains:
    characters = ['a','b', 'c']

    Then the built tree is parsed and regions will be added to each node:
    For example, for the above character list, the region will be "abc".

    It also add two more attributes to each node:
    start_index (int): This is an index of the first character in the input text.
      This includes parantheses.
    """
    # Build a tree of nodes
    root = _build_tree(prompt)

    # A node on the tree contains a list of characters.
    # Each character has a corresponding position in the original buffer
    # Create a word that are located next to each other.
    # This is called a region.

    # Convert them to list of words (region)
    node = root
    queue = list()
    queue.append(node)

    while True:
        if len(queue) == 0:
            break
        current = queue.pop(0)
 
        # Convert characters to regions
        regions = list()
        start_indices = list()
        prev_index = -1
        start_index = 0  # global character index including "(" and ")" that are not put in a tree
        word = ""
        for i, c in enumerate(current.characters):
            current_index = current.pos[i]
            if current_index != (prev_index + 1):  # Characters are not consecutive
                if len(word) > 0:
                    regions.append(word)
                    logger.debug("compute_prompt_score:word: " + word)
                    start_indices.append(start_index)  # Word position tracker in the global buffer

                word = c
                start_index = current_index
            else:
                word += c
            prev_index = current_index

        logger.debug("compute_prompt_score:word: " + word)
        regions.append(word)

        start_indices.append(start_index)
        current.regions = regions
        current.start_indices = start_indices

        # Check score for the last region
        last_region = current.regions[-1]
 
        # last_region may contain the embedding tag
        # and it may not contain the custom score
        # <embedding:foo.bin>world
        # so extract the embedding.tag
        # First array is the list of strings
        l, _ = split_token_with_embedding_tags(last_region)
        if l != []:  # i.e. if last_region == ['']
            tmp_last_region = l[-1]  
            logger.debug("compute_prompt_score:word:last region:" + last_region)

            # Check if it has ":"
            colon_pos = tmp_last_region.rfind(":")

            # If ":" is found and is not the last character
            if colon_pos >= 0 and colon_pos < len(tmp_last_region)-1:
                # Adjust the colon pos
                # as "<embedding:foo.bin>world:1.2"
                # will be split to <embedding:foo.bin> and world:1.2
                colon_pos = last_region.rfind(":")
                score_candidate = last_region[colon_pos+1:]
                try:
                    score = float(score_candidate)
                    current.score = score
                    current.regions[-1] = last_region[:colon_pos]  # remove score from text
                except:
                    # ignore invalid score
                    if current == root:
                        current.score = 1.0
                    else:
                        current.score = 1.1
            else:  # if a score is missing, assign 1.1
                if current == root:
                    current.score = 1.0
                else:
                    current.score = 1.1
        else:  # current region is empty
            if current == root:
                current.score = 1.0
            else:
                current.score = 1.1

        for n in current.children:
            queue.append(n)

    # Compute product score from the tree
    _compute_product_score(node, 1.0)

    return node


def _convert_region_tree_to_word_list(root):
    """
    Converts a tree of regions to a word list.
    Each element of the word list contains:
    (region, product score of region)

    A region is a list of words that can contain an empty string ''.
    """
    # Generate a list that contains
    # (region, start index of region, product score of region)
    word_list = []   
    node = root
    queue = list()
    queue.append(node)
    while True:
        if len(queue) == 0:
            break
        current = queue.pop(0)
        for i, r in enumerate(current.regions):
            # Add a triplet
            word_list.append((r, current.start_indices[i], current.product_score))
            logger.debug("compute_prompt_score:r:" + r)
        for n in current.children:
            queue.append(n)

    # Sort by the start index
    l = list()
    word_list = sorted(word_list, key=lambda e: e[1])
    for w in word_list:
        l.append((w[0], w[2]))  # Drop start index as we don't need it any more
    
    return l

def _convert_region_score_tuple_list_to_word_list(region_score_tuple_list):
    """
    Converts a tuple of region and score to a word list.

    A region is a string that can contain multiple words.
    """
    l_prev_step = region_score_tuple_list

    # Split each region to words by a white space
    l = list()
    for w in l_prev_step:
        score = w[1]
        region = w[0]
        words = re.split(r'\s+' ,region)  # Handles both carriage returns and white space
        words = list(filter(lambda e: e != '', words))  # Drop empty strings
        l.append((words, score))
    l_prev_step = l

    # Flatten the list to make an element a tuple of a word and the score
    l = list()
    for w in l_prev_step:
        words = w[0]
        score = w[1]
        for w2 in words:
            l.append((w2, score))
    return l

def _compute_prompt_score(prompt: str) -> List[Tuple[str, float]]:
    """
    Computes scores for each token in the input prompt based on specific criteria.

    Args:
        prompt (str): The input text for which the scores are to be computed.

    Returns:
        List[Tuple[str, float]]: A list of tuples, where each tuple contains a token from the input text and its corresponding score as a float. The scoring criteria are defined by the implementation details of the function.

    Example:
        >>> compute_prompt_score("hello, ((world), everyone:1.2).")
        [('hello,', 1.0), ('world', 1.32), (',', 1.2), ('everyone', 1.2), ('.', 1.0)]

    Note:
        The scoring algorithm and the way tokens are identified and extracted should be detailed in the implementation section of this function. This docstring assumes a scoring system that is influenced by the example provided but may need to be adjusted according to the actual implementation.
    """
    if len(prompt.strip()) == 0:
        logger.debug("Prompt is empty")
        return [("", 1.0)]
    
    root = _build_region(prompt)

    # Generate a word list
    word_list = _convert_region_tree_to_word_list(root)
    l_prev_step = _convert_region_score_tuple_list_to_word_list(word_list)

    # Special token processing
    # Take out embedding tag
    # abc<embedding:hello.bin>xyz
    # will be converted to
    # abc, <embedding:hello.bin>, xyz
    l = list()
    for w in l_prev_step:
        words = w[0]
        score = w[1]

        s_list, b_list = split_token_with_embedding_tags(words)
        for s in s_list:
            logger.debug(f"compute_prompt_score: Adding: {s} {score}")
            l.append((s, score))

    return l


def generate_clip_embeddings_from_prompt(tokenizer, model, embedding_dir, prompt:str):
    word_score_pairs = _compute_prompt_score(prompt)
    retval = generate_clip_embeddings(tokenizer,
                             model,
                             embedding_dir,
                             word_score_pairs)
    return retval

def generate_open_clip_embeddings_from_prompt(model, embedding_dir, prompt:str):
    word_score_pairs = _compute_prompt_score(prompt)
    retval, eos_index_list = generate_open_clip_embeddings(
                             model,
                             embedding_dir,
                             word_score_pairs)
    return retval, eos_index_list


if __name__ == "__main__":
    model_id = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = ModCLIPTextModel.from_pretrained(model_id)    

    prompt = "hello, ((<embedding:badhandv4.pt>world), <embedding:foo.bin>everyone:1.2)." \
    """
    The Empire State Building is arguably the most beautiful structure in Manhattan.
    Constructed many years ago, it attracts tens of thousands of visitors daily from
    across the globe. These visitors ascend to the observatory deck, where they can
    enjoy panoramic views of Manhattan from an outdoor balcony.
    """
    prompt = ""
    retval, eos_index_list = generate_clip_embeddings_from_prompt(tokenizer,
                             model,
                             "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
                             prompt)
    for r in retval:
        print(r.shape)