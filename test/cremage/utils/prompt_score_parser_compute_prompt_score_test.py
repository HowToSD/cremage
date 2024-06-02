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
TEST_ROOT = os.path.join(PROJECT_ROOT, "test")
sys.path = [MODULE_ROOT, TEST_ROOT] + sys.path

from cremage.utils.prompt_score_parser import _compute_prompt_score
# Test package import
from cremage.utils.test_utils import round2digits

class TestComputePromptScore(unittest.TestCase):

    def test_compute_prompt_score_1(self):
        text = "hello"
        output = _compute_prompt_score(text)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))

    def test_compute_prompt_score_2(self):
        text = "(hello)"
        output = _compute_prompt_score(text)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.1))

    def test_compute_prompt_score_3(self):
        """
        Note that even if closing parenthesis is missing, a child node with 1.1
        is still created.
        """
        text = "(hello"
        output = _compute_prompt_score(text)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.1))

    def test_compute_prompt_score_4(self):
        text = "hello(abc)x"
        root = _compute_prompt_score(text)
        output = _compute_prompt_score(text)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))
        self.assertTrue(output[1][0] == "abc")
        self.assertTrue(round2digits(output[1][1]) == round2digits(1.1))
        self.assertTrue(output[2][0] == "x")
        self.assertTrue(round2digits(output[2][1]) == round2digits(1.0))

    def test_compute_prompt_score_5(self):
        text = "hello (abc   ) x "
        root = _compute_prompt_score(text)
        output = _compute_prompt_score(text)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))
        self.assertTrue(output[1][0] == "abc")
        self.assertTrue(round2digits(output[1][1]) == round2digits(1.1))
        self.assertTrue(output[2][0] == "x")
        self.assertTrue(round2digits(output[2][1]) == round2digits(1.0))

    def test_compute_prompt_score_6(self):
        text = "hello(abc)x(def)"
        output = _compute_prompt_score(text)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))
        self.assertTrue(output[1][0] == "abc")
        self.assertTrue(round2digits(output[1][1]) == round2digits(1.1))
        self.assertTrue(output[2][0] == "x")
        self.assertTrue(round2digits(output[2][1]) == round2digits(1.0))
        self.assertTrue(output[3][0] == "def")
        self.assertTrue(round2digits(output[3][1]) == round2digits(1.1))

    def test_compute_prompt_score_7(self):
        text = "hello(abc(def))x"
        output = _compute_prompt_score(text)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))
        self.assertTrue(output[1][0] == "abc")
        self.assertTrue(round2digits(output[1][1]) == round2digits(1.1))
        self.assertTrue(output[2][0] == "def")
        self.assertTrue(round2digits(output[2][1]) == round2digits(1.1 * 1.1))
        self.assertTrue(output[3][0] == "x")
        self.assertTrue(round2digits(output[3][1]) == round2digits(1.0))

    def test_compute_prompt_score_8(self):
        text = "hello(abc(def):1.2)x"
        output = _compute_prompt_score(text)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))
        self.assertTrue(output[1][0] == "abc")
        self.assertTrue(round2digits(output[1][1]) == round2digits(1.2))
        self.assertTrue(output[2][0] == "def")
        self.assertTrue(round2digits(output[2][1]) == round2digits(1.2 * 1.1))
        self.assertTrue(output[3][0] == "x")
        self.assertTrue(round2digits(output[3][1]) == round2digits(1.0))

    def test_compute_prompt_score_9(self):
        text = """Cute puppy on the kitchen counter
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
        output = _compute_prompt_score(text)
        self.assertTrue(len(output) == 72)


if __name__ == '__main__':
    unittest.main()
