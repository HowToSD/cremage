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

from cremage.utils.token_process_helper import split_token_with_embedding_tags

class TestTokenProcessHelper(unittest.TestCase):

    def test_score_1(self):
        s, b = split_token_with_embedding_tags("helloworld")
        self.assertTrue(s[0] == "helloworld")
        self.assertTrue(b[0] is False)
        
    def test_score_2(self):
        s, b = split_token_with_embedding_tags("hello<embedding:foo.bin>world")
        self.assertTrue(s[0] == "hello")
        self.assertTrue(b[0] is False)
        self.assertTrue(s[1] == "<embedding:foo.bin>")
        self.assertTrue(b[1] is True)
        self.assertTrue(s[2] == "world")
        self.assertTrue(b[2] is False)

    def test_score_3(self):
        """
        Tests for missing >.
        """
        s, b = split_token_with_embedding_tags("hello<embedding:foo.binworld")
        self.assertTrue(s[0] == "hello<embedding:foo.binworld")
        self.assertTrue(b[0] is False)

    def test_score_4(self):
        s, b = split_token_with_embedding_tags("hello<embedding:foo.bin>world<embedding:bar.pt>")
        self.assertTrue(s[0] == "hello")
        self.assertTrue(b[0] is False)
        self.assertTrue(s[1] == "<embedding:foo.bin>")
        self.assertTrue(b[1] is True)
        self.assertTrue(s[2] == "world")
        self.assertTrue(b[2] is False)
        self.assertTrue(s[3] == "<embedding:bar.pt>")
        self.assertTrue(b[3] is True)


if __name__ == '__main__':
    unittest.main()
