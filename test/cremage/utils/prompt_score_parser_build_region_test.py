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

from cremage.utils.prompt_score_parser import _build_region


class TestPromptScoreParserBuildRegion(unittest.TestCase):

    def test_build_region_1(self):
        text = "hello"
        root = _build_region(text)
        self.assertTrue(root.regions == ["hello"])
        self.assertTrue(root.start_indices == [0])
        
    def test_build_region_2(self):
        text = "hello(abc)x"
        root = _build_region(text)
        self.assertTrue(root.regions == ["hello", "x"])
        self.assertTrue(root.start_indices == [0, 10])
        child = root.children[0]
        self.assertTrue(child.regions==["abc"])
        self.assertTrue(child.start_indices==[6])
        
    def test_build_region_3(self):
        text = "(hello)"
        root = _build_region(text)
        self.assertTrue(root.regions==[''])
        self.assertTrue(root.start_indices == [0])
        child = root.children[0]
        self.assertTrue(child.regions==["hello"])
        self.assertTrue(child.start_indices == [1])
        
    def test_build_region_4(self):
        text = "hello(abc)x(def)"
        root = _build_region(text)
        self.assertTrue(root.regions == ["hello", "x"])
        self.assertTrue(root.start_indices==[0, 10])
        child = root.children[0]
        self.assertTrue(child.regions==["abc"])
        self.assertTrue(child.start_indices==[6])
        child = root.children[1]
        self.assertTrue(child.regions==["def"])
        self.assertTrue(child.start_indices==[12])

    def test_build_region_5(self):
        text = "hello(abc(def))x"
        root = _build_region(text)
        self.assertTrue(root.regions==["hello", "x"])
        self.assertTrue(root.start_indices==[0, 15])
        self.assertTrue(root.score==1.0)
        self.assertTrue(root.product_score==1.0)
        
        child = root.children[0]
        self.assertTrue(child.regions==["abc"])
        self.assertTrue(child.start_indices==[6])

        child = child.children[0]
        self.assertTrue(child.regions==["def"])
        self.assertTrue(child.start_indices==[10])
        self.assertTrue(round(child.score, 2) == 1.1)
        self.assertTrue(round(child.product_score, 2) == 1.21)

    def test_build_region_6(self):
        text = "hello(abc(def:1.3))x"
        root = _build_region(text)
        self.assertTrue(root.regions==["hello", "x"])
        self.assertTrue(root.score==1.0)
        self.assertTrue(root.product_score==1.0)
        
        child = root.children[0]
        self.assertTrue(child.regions==["abc"])
        self.assertTrue(round(child.score, 2) == round(1.1, 2))
        self.assertTrue(round(child.product_score, 2) == round(1.1, 2))

        child = child.children[0]
        self.assertTrue(child.regions==["def"])
        self.assertTrue(round(child.score, 2) == 1.3)
        self.assertTrue(round(child.product_score, 2) == round(1.1 * 1.3, 2))

    def test_build_region_7(self):
        text = "hello(abc(def:1.3):1.15)x"
        root = _build_region(text)
        self.assertTrue(root.regions==["hello", "x"])
        self.assertTrue(root.score==1.0)
        self.assertTrue(root.product_score==1.0)
        
        child = root.children[0]
        self.assertTrue(child.regions==["abc", ''])
        self.assertTrue(round(child.score, 2) == round(1.15, 2))
        self.assertTrue(round(child.product_score, 2) == round(1.15, 2))

        child = child.children[0]
        self.assertTrue(child.regions==["def"])
        self.assertTrue(round(child.score, 2) == 1.3)
        self.assertTrue(round(child.product_score, 2) == round(1.15 * 1.3, 2))


if __name__ == '__main__':
    unittest.main()
