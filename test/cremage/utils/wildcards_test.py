import os
import sys
import re
import unittest

import torch
import numpy as np
import cv2
import PIL

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
TEST_DATA_ROOT = os.path.join(PROJECT_ROOT, "test", "data")
WILDCARDS_DATA_DIR = os.path.join(TEST_DATA_ROOT, "wildcards")


from cremage.utils.wildcards import resolve_wildcards


class TestWildcards(unittest.TestCase):

    def test_no_replacement(self):
        inputs = "Photo of a squirrel."
        result = resolve_wildcards(inputs = inputs, wildcards_dir=WILDCARDS_DATA_DIR)
        self.assertEqual(inputs, result)
        
    def test_replacement(self):
        inputs = "Photo of a __pet__, film."
        result = resolve_wildcards(inputs = inputs, wildcards_dir=WILDCARDS_DATA_DIR)
        match = re.search(r"a (.+),", result)
        self.assertTrue(match is not None)
        word = match.group(1)
        self.assertTrue(word in ["dog", "cat", "hamster", "ferret", "capybara"])

    def test_replacement_2(self):
        inputs = "Photo of a __pet__, __color__ film."

        for i in range(10):  # Repeat
            result = resolve_wildcards(inputs = inputs, wildcards_dir=WILDCARDS_DATA_DIR)
            match = re.search(r"a (.+),", result)
            self.assertTrue(match is not None)
            word = match.group(1)
            self.assertTrue(word in ["dog", "cat", "hamster", "ferret", "capybara"])

            match = re.search(r", (.+) film", result)
            self.assertTrue(match is not None)
            word = match.group(1)
            self.assertTrue(word in [
                "red",
                "blue",
                "white",
                "green",
                "orange",
                "yellow",
                "black",
                "purple",
                "cyan"
            ])

    def test_nested(self):
        inputs = "Photo of a __pet2__, film."
        result = resolve_wildcards(inputs = inputs, wildcards_dir=WILDCARDS_DATA_DIR)
        match = re.search(r"a (.+),", result)
        self.assertTrue(match is not None)
        word = match.group(1)
        self.assertTrue(word in ["St. Bernard", "Old English Sheepdog", "Golden Retriever", "cat"])

    def test_cycle(self):
        inputs = "Photo of __cycle1__."
        result = resolve_wildcards(inputs = inputs, wildcards_dir=WILDCARDS_DATA_DIR)
        match = re.search(r"of (.+).", result)
        self.assertTrue(match is not None)
        word = match.group(1)
        self.assertEqual(word, "__cycle1__")

    def test_cycle_2(self):
        inputs = "Photo of __cycle3__."
        result = resolve_wildcards(inputs = inputs, wildcards_dir=WILDCARDS_DATA_DIR)
        match = re.search(r"of (.+).", result)
        self.assertTrue(match is not None)
        word = match.group(1)
        self.assertEqual(word, "__cycle5__")

    def test_missing_file(self):
        inputs = "Photo of __missing__."
        result = resolve_wildcards(inputs = inputs, wildcards_dir=WILDCARDS_DATA_DIR)
        match = re.search(r"of (.+).", result)
        self.assertTrue(match is not None)
        word = match.group(1)
        self.assertEqual(word, "__missing__")

    def test_incomplete(self):
        inputs = "Photo of __incomplete"
        result = resolve_wildcards(inputs = inputs, wildcards_dir=WILDCARDS_DATA_DIR)
        match = re.search(r"of (.+)$", result)
        self.assertTrue(match is not None)
        word = match.group(1)
        self.assertEqual(word, "__incomplete")




if __name__ == '__main__':
    unittest.main()
