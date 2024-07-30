import os
import sys
import json
import unittest
import tempfile

import torch
import numpy as np
import PIL
from PIL import Image


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path

import open_clip
from cremage.utils.generate_open_clip_embeddings_from_tokens import token_to_embedding
from cremage.utils.generate_open_clip_embeddings_from_tokens import convert_word_to_tokens
from cremage.utils.generate_open_clip_embeddings_from_tokens import generate_open_clip_embeddings
from cremage.utils.prompt_score_parser import generate_open_clip_embeddings_from_prompt


class TestOpenClip(unittest.TestCase):
    model = None

    def _load_model(self):
        # sdxl_base settings (see modules/sdxl/configs/inference/sd_xl_base.yaml)
        arch = "ViT-bigG-14"
        version = "laion2b_s39b_b160k"
        freeze = True
        layer = "penultimate"  # API default is "last"
        always_return_pooled = True  # API default is False
        legacy = False  # API default is True
        device=os.environ.get("GPU_DEVICE", "cpu")
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        return model

    def test_loading_model(self):
        """
        Note: This method is slow (11 sec on NVIDIA 4090).
        Based on FrozenCLIPEmbedder2 in sgm/modules/encoders/modules.py.
        Check SDXL license in the root of this project.
        """
        TestOpenClip.model = self._load_model()
        self.assertTrue(TestOpenClip.model is not None)

    def test_tokenizer(self):
        text = "hello, world"
        tokens = open_clip.tokenize(text)
        self.assertTrue(tokens.shape==(1, 77))  # Note that this is rank 2

        expected = torch.tensor([49406,  3306,   267,  1002, 49407] + [0] * 72).int()
        self.assertTrue(expected.shape==tokens[0].shape)
        for i, token in enumerate(tokens[0]):
            self.assertTrue(expected[i] == token)

    def test_convert_word_to_tokens_1_word(self):
        text = "hello"
        tokens, converted_length = convert_word_to_tokens(text)
        self.assertTrue(tokens.shape==(1,))
        
        self.assertTrue(converted_length == 1)
        expected = torch.tensor([3306]).int()

        self.assertTrue(expected.shape==tokens.shape)
        for i, token in enumerate(tokens):
            self.assertTrue(expected[i] == token)

    def test_convert_word_to_tokens_blank(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        text = ""
        tokens, converted_length = convert_word_to_tokens(text)
        self.assertTrue(tokens.shape==(0,))
        self.assertTrue(converted_length == 0)

    def test_tokenizer_truncation(self):
        """
        Check truncation. Note that tokens for "hello, world" is still kept
        so the truncation happens on the tail by default which SDXL uses.
        """
        text = """hello, world
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        filler, filler, filler, filler, filler, filler, filler, filler
        """
        tokens = open_clip.tokenize(text)
        self.assertTrue(tokens.shape==(1, 77))  # Note that this is rank 2
        expected = torch.tensor(
            [49406,  3306,   267,  1002, 32816,   267, 32816, 267, 32816, 267,
         32816,   267, 32816,   267, 32816,   267, 32816,   267, 32816, 32816,
           267, 32816,   267, 32816,   267, 32816,   267, 32816,   267, 32816,
           267, 32816,   267, 32816, 32816,   267, 32816,   267, 32816,   267,
         32816,   267, 32816,   267, 32816,   267, 32816,   267, 32816, 32816,
           267, 32816,   267, 32816,   267, 32816,   267, 32816,   267, 32816,
           267, 32816,   267, 32816, 32816,   267, 32816,   267, 32816,   267,
         32816,   267, 32816,   267, 32816,   267, 49407]
        )
        self.assertTrue(expected.shape==tokens[0].shape)
        for i, token in enumerate(tokens[0]):
            self.assertTrue(expected[i] == token)

    def test_token_to_embedding(self):
        """
        Note embedding size is 1280d and no longer 768 as CLIP in SD 1.5.
        """
        model = TestOpenClip.model
        text = "hello, world"
        
        # Convert text to tokens
        tokens = open_clip.tokenize(text)
        self.assertTrue(tokens.shape==(1, 77))  # Note that this is rank 2

        expected = torch.tensor([49406,  3306,   267,  1002, 49407] + [0] * 72).int()
        self.assertTrue(expected.shape==tokens[0].shape)
        for i, token in enumerate(tokens[0]):
            self.assertTrue(expected[i] == token)

        # Convert tokens to embeddings
        embeddings = model.token_embedding(tokens)
        self.assertTrue(embeddings.shape == (1, 77, 1280))

    def test_token_to_embedding_via_util(self):
        """
        Note embedding size is 1280d and no longer 768 as CLIP in SD 1.5.
        """
        model = TestOpenClip.model
        text = "hello, world"
        
        # Convert text to tokens
        tokens = open_clip.tokenize(text)
        self.assertTrue(tokens.shape==(1, 77))  # Note that this is rank 2

        expected = torch.tensor([49406,  3306,   267,  1002, 49407] + [0] * 72).int()
        self.assertTrue(expected.shape==tokens[0].shape)
        for i, token in enumerate(tokens[0]):
            self.assertTrue(expected[i] == token)

        expected = torch.tensor([49406,  3306,   267,  1002, 49407] + [0] * 72).int()
        self.assertTrue(expected.shape==tokens[0].shape)
        for i, token in enumerate(tokens[0]):
            self.assertTrue(expected[i] == token)

        # Convert tokens to embeddings
        embeddings = token_to_embedding(model, tokens)
        self.assertTrue(embeddings.shape == (1, 77, 1280))

    def test_generate_open_clip_embeddings(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        TestOpenClip.model = self._load_model() if TestOpenClip.model is None else TestOpenClip.model
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                # ('<embedding:badhandv4.pt>', 1.0),
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_open_clip_embeddings(
            TestOpenClip.model, "/media/pup/ssd2/recoverable_data/sd_models/embeddings", inputs)
        self.assertTrue(retval[0].shape == (77, 1280))
        self.assertTrue(eos_index_list[0] == 6)

    def test_generate_clip_embeddings_with_sdxl_embedding_file(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        TestOpenClip.model = self._load_model() if TestOpenClip.model is None else TestOpenClip.model
        model = TestOpenClip.model
        text = "hello, world"
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                ('<embedding:ac_neg1.safetensors>', 1.0),  # 52 sequence slots
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_open_clip_embeddings(
            model,
            "/media/pup/ssd2/recoverable_data/sd_models/embeddings_sdxl",
            inputs)
        self.assertTrue(retval[0].shape == (77, 1280))
        self.assertTrue(eos_index_list[0] == 58)  # 6 from prev + 52 for embedding

    def test_generate_open_clip_embeddings2(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        TestOpenClip.model = self._load_model() if TestOpenClip.model is None else TestOpenClip.model
        model = TestOpenClip.model

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
        retval, eos_index_list = generate_open_clip_embeddings_from_prompt(
                                model,
                                "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
                                prompt)
        self.assertTrue(len(retval) == 9)
        for r in retval:
            self.assertTrue(r.shape == (77, 1280))
        self.assertTrue(eos_index_list[0] == 76)
        self.assertTrue(eos_index_list[-1] == 27)


    def test_generate_open_clip_embeddings3(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        TestOpenClip.model = self._load_model() if TestOpenClip.model is None else TestOpenClip.model
        model = TestOpenClip.model

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
        retval, eos_index_list = generate_open_clip_embeddings_from_prompt(
                                model,
                                "/media/pup/ssd2/recoverable_data/sd_models/embeddings",
                                prompt)
        self.assertTrue(len(retval) == 1)
        for r in retval:
            self.assertTrue(r.shape == (77, 1280))
        self.assertTrue(eos_index_list[0] == 73)

    def test_break(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        TestOpenClip.model = self._load_model() if TestOpenClip.model is None else TestOpenClip.model
        model = TestOpenClip.model
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                ('BREAK', 1.0),
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_open_clip_embeddings(
            model,
            "/media/pup/ssd2/recoverable_data/sd_models/embeddings_sdxl",
            inputs)
        self.assertTrue(len(retval) == 2)

    def test_break_2(self):
        """
        A blank string "" gets converted to:
        49406, 49407, 0, ... 0
        """
        TestOpenClip.model = self._load_model() if TestOpenClip.model is None else TestOpenClip.model
        model = TestOpenClip.model
        inputs = [
                ('Photo', 1.0),
                ('of', 1.0),
                ('a', 1.0),        
                ('dancing', 1.2),
                ('break', 1.0),
                ('man', 1.1)
        ]

        retval, eos_index_list = generate_open_clip_embeddings(
            model,
            "/media/pup/ssd2/recoverable_data/sd_models/embeddings_sdxl",
            inputs)
        self.assertTrue(len(retval) == 1)





if __name__ == '__main__':
    unittest.main()
