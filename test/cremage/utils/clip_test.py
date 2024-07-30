import os
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True
import sys
import json
import unittest
import tempfile

import torch
import numpy as np
import PIL
from PIL import Image
from transformers import CLIPTokenizer  # HINADA Change

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path

from cremage.utils.generate_clip_embeddings_from_tokens import token_to_embedding
from cremage.utils.generate_clip_embeddings_from_tokens import convert_word_to_tokens
from cremage.utils.generate_clip_embeddings_from_tokens import generate_clip_embeddings
from cremage.utils.prompt_score_parser import generate_clip_embeddings_from_prompt

# from clip.modeling_clip import CLIPTextModel as ModCLIPTextModel
from clip_sdxl.modeling_clip import CLIPTextModel as ModCLIPTextModel  # DEBUG FIXME CREMAGE

class TestClip(unittest.TestCase):
    tokenizer = None
    model = None

    def _load_model(self):
        version="openai/clip-vit-large-patch14"

        if __class__.tokenizer:
            tokenizer = __class__.tokenizer
        else:
            tokenizer = CLIPTokenizer.from_pretrained(version, local_files_only=local_files_only_value)
            __class__.tokenizer = tokenizer
        
        if __class__.model:
            model = __class__.model
        else:
            lora_ranks = []
            lora_weights = []
            model = ModCLIPTextModel.from_pretrained(version,
                                                                lora_ranks=lora_ranks,
                                                                lora_weights=lora_weights,
                                                                local_files_only=local_files_only_value)
            __class__.model == model

        return tokenizer, model

    def test_loading_model(self):
        """
        Note: This method is slow (11 sec on NVIDIA 4090).
        Based on FrozenCLIPEmbedder2 in sgm/modules/encoders/modules.py.
        Check SDXL license in the root of this project.
        """
        tokenizer, model = self._load_model()
        self.assertTrue(tokenizer is not None)
        self.assertTrue(model is not None)

    def test_tokenizer_with_padding(self):
        tokenizer, model = self._load_model()
        text = "hello, world"
        token_data = tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",  # this will cause the converted length to be 77
            return_tensors="pt")
        raw_tokens = token_data["input_ids"]
        converted_length = token_data["length"][0].item()
        self.assertTrue(converted_length == 77)
        self.assertTrue(raw_tokens.shape==(1, 77))  # Note that this is rank 2
        expected = torch.tensor([49406,  3306,   267,  1002, 49407] + [49407] * 72).int()
        self.assertTrue(expected.shape==raw_tokens[0].shape)
        for i, token in enumerate(raw_tokens[0]):
            self.assertTrue(expected[i] == token)

    def test_tokenizer_no_padding(self):
        tokenizer, model = self._load_model()
        text = "hello, world"
        # Note padding="max_length" is not set as we add padding ourselves
        token_data = tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            return_tensors="pt")
        raw_tokens = token_data["input_ids"]
        converted_length = token_data["length"][0].item()
        self.assertTrue(converted_length == 5)
        self.assertTrue(raw_tokens.shape==(1, 5))  # Note that this is rank 2
        expected = torch.tensor([49406,  3306,   267,  1002, 49407]).int()
        self.assertTrue(expected.shape==raw_tokens[0].shape)
        for i, token in enumerate(raw_tokens[0]):
            self.assertTrue(expected[i] == token)

    def test_convert_word_to_tokens_1_word(self):
        tokenizer, model = self._load_model()
        text = "hello"
        tokens, converted_length = convert_word_to_tokens(tokenizer, text)
        self.assertTrue(tokens.shape==(1,))
        
        self.assertTrue(converted_length == 1)
        expected = torch.tensor([3306]).int()

        self.assertTrue(expected.shape==tokens.shape)
        for i, token in enumerate(tokens):
            self.assertTrue(expected[i] == token)

    def test_convert_word_to_tokens_blank(self):
        """
        A blank string "" gets converted to:
        49406, 49407
        """
        tokenizer, model = self._load_model()
        text = ""
        tokens, converted_length = convert_word_to_tokens(tokenizer, text)
        self.assertTrue(tokens.shape==(0,))
        self.assertTrue(converted_length == 0)

    def test_token_to_embedding(self):
        """
        Note embedding size is 768d.
        """
        tokenizer, model = self._load_model()
        text = "hello, world"
        
        # Convert text to tokens
        text = "hello, world"
        token_data = tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",  # this will cause the converted length to be 77
            return_tensors="pt")
        raw_tokens = token_data["input_ids"]
        converted_length = token_data["length"][0].item()
        self.assertTrue(converted_length == 77)
        self.assertTrue(raw_tokens.shape==(1, 77))  # Note that this is rank 2

        # Convert tokens to embeddings
        embeddings = model.text_model.embeddings.token_embedding(raw_tokens)
        self.assertTrue(embeddings.shape == (1, 77, 768))

    def test_token_to_embedding_via_util(self):
        """
        Tests generating embedding using a util function.
        """
        tokenizer, model = self._load_model()
        text = "hello, world"
        
        # Convert text to tokens
        text = "hello, world"
        token_data = tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",  # this will cause the converted length to be 77
            return_tensors="pt")
        raw_tokens = token_data["input_ids"]
        converted_length = token_data["length"][0].item()
        self.assertTrue(converted_length == 77)
        self.assertTrue(raw_tokens.shape==(1, 77))  # Note that this is rank 2

        # Convert tokens to embeddings
        embeddings = token_to_embedding(model, raw_tokens)
        self.assertTrue(embeddings.shape == (1, 77, 768))

    def test_generate_clip_embeddings(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        tokenizer, model = self._load_model()
        text = "hello, world"
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                # ('<embedding:badhandv4.pt>', 1.0),
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_clip_embeddings(
            tokenizer,
            model,
            "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
            inputs)
        self.assertTrue(retval[0].shape == (77, 768))
        self.assertTrue(eos_index_list[0] == 6)


    def test_generate_clip_embeddings_with_embedding_file(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        tokenizer, model = self._load_model()
        text = "hello, world"
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                ('<embedding:badhandv4.pt>', 1.0),  # 6 sequence slots
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_clip_embeddings(
            tokenizer,
            model,
            "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
            inputs)
        self.assertTrue(retval[0].shape == (77, 768))
        self.assertTrue(eos_index_list[0] == 12)  # 6 from prev + 6 for embedding


    def test_generate_clip_embeddings_with_sdxl_embedding_file(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        tokenizer, model = self._load_model()
        text = "hello, world"
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                ('<embedding:ac_neg1.safetensors>', 1.0),  # 52 sequence slots
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_clip_embeddings(
            tokenizer,
            model,
            "/media/pup/ssd2/recoverable_data/sd_models/embeddings_sdxl",
            inputs)
        self.assertTrue(retval[0].shape == (77, 768))
        self.assertTrue(eos_index_list[0] == 58)  # 6 from prev + 52 for embedding


    def test_generate_clip_embeddings_2(self):
        """
        A blank string "" gets converted to:
        49406, 49407, ... 49407
        """
        tokenizer, model = self._load_model()

        prompt = "hello, ((everyone:1.2)." \
        """
        The Empire State Building is arguably the most beautiful structure in Manhattan.
        Constructed many years ago, it attracts tens of thousands of visitors daily from
        across the globe. These visitors ascend to the observatory deck, where they can
        enjoy panoramic views of Manhattan from an outdoor balcony.:0.9).

Paris, the capital city of France, boasts a rich and varied history that spans over two millennia. Founded in the 3rd century BC by a Celtic tribe known as the Parisii, the city's strategic location along the Seine River made it a key trading hub. By 52 BC, the Romans had conquered the area, renaming it Lutetia. The city flourished under Roman rule, with temples, baths, theaters, and an extensive road network.

During the 5th century, Lutetia was renamed Paris, deriving from the original Parisii inhabitants. The city gained prominence as the Merovingian dynasty's power base, particularly under King Clovis I, who made Paris his capital in 508 AD. The Carolingian dynasty, however, preferred Aachen as their capital, causing Paris to lose some of its significance until the late 9th century.

The 12th century marked a period of rapid growth and transformation for Paris. The construction of the Cathedral of Notre-Dame began in 1163, symbolizing the city's spiritual and cultural significance. The University of Paris, founded around 1150, became a leading center of learning in Europe. By the end of the century, Paris had solidified its status as a political, economic, and intellectual hub.

Paris continued to grow throughout the Middle Ages and the Renaissance. The 16th and 17th centuries saw the construction of significant landmarks such as the Louvre Palace and the Tuileries Gardens. During this time, Paris was a center for art, science, and philosophy, attracting figures like René Descartes and Voltaire.

The French Revolution in 1789 brought dramatic changes to Paris. The storming of the Bastille on July 14, 1789, became a symbol of the uprising against the monarchy. The revolution led to the rise of Napoleon Bonaparte, who made Paris the centerpiece of his empire. Under his rule, Paris underwent significant architectural changes, including the construction of the Arc de Triomphe.

The 19th century was marked by further transformation, particularly under Baron Haussmann, who modernized the city with wide boulevards, parks, and improved sanitation. The Eiffel Tower, constructed for the 1889 Exposition Universelle, became an enduring symbol of Paris.

During the 20th century, Paris endured the hardships of both World Wars but emerged as a global center of art, fashion, and culture. The post-war period saw rapid urbanization and the establishment of iconic cultural institutions like the Centre Pompidou.

Today, Paris remains a vibrant metropolis, celebrated for its history, architecture, and cultural influence worldwide. Its historical legacy is reflected in its monuments, museums, and the enduring charm of its neighborhoods.
        """
        retval, eos_index_list = generate_clip_embeddings_from_prompt(
                                tokenizer,
                                model,
                                "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
                                prompt)
        self.assertTrue(len(retval) == 9)
        for r in retval:
            self.assertTrue(r.shape == (77, 768))
        self.assertTrue(eos_index_list[0] == 76)
        self.assertTrue(eos_index_list[-1] == 27)


    def test_generate_clip_embeddings3(self):
        tokenizer, model = self._load_model()

        prompt = """Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter
Cute puppy on the kitchen counter

        
        
        
        
        
        """
        retval, eos_index_list = generate_clip_embeddings_from_prompt(
                                tokenizer,
                                model,
                                "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
                                prompt)
        print(len(retval))
        self.assertTrue(len(retval) == 1)
        for r in retval:
            self.assertTrue(r.shape == (77, 768))
        self.assertTrue(eos_index_list[0] == 73)

    def test_break(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        tokenizer, model = self._load_model()
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                ('BREAK', 1.0),
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_clip_embeddings(
            tokenizer,
            model,
            "/media/pup/ssd2/recoverable_data/sd_models/embeddings_sdxl",
            inputs)
        self.assertTrue(len(retval) == 2)

    def test_break_2(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        tokenizer, model = self._load_model()
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                ('break', 1.0),
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_clip_embeddings(
            tokenizer,
            model,
            "/media/pup/ssd2/recoverable_data/sd_models/embeddings_sdxl",
            inputs)
        self.assertTrue(len(retval) == 1)


if __name__ == '__main__':
    unittest.main()
