import os
import sys
import unittest

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TEST_ROOT = os.path.join(PROJECT_ROOT, "test")
sys.path = [MODULE_ROOT, TEST_ROOT] + sys.path
from cremage.utils.prompt_score_parser import _build_region, _convert_region_tree_to_word_list
# Test package import
from cremage.utils.test_utils import round2digits

class TestPromptScoreParserBuildRegion(unittest.TestCase):

    def test_build_region_1(self):
        text = "hello"
        root = _build_region(text)
        l = _convert_region_tree_to_word_list(root)
        print(l)
        self.assertTrue(l == [("hello", 1.0)])
        
    def test_build_region_2(self):
        text = "hello(abc)x y"
        root = _build_region(text)
        l = _convert_region_tree_to_word_list(root)
        self.assertTrue(l[0][0] == "hello")
        self.assertTrue(round2digits(l[0][1]) == round2digits(1.0))
        self.assertTrue(l[1][0] == "abc")
        self.assertTrue(round2digits(l[1][1]) == round2digits(1.1))
        self.assertTrue(l[2][0] == "x y")
        self.assertTrue(round2digits(l[2][1]) == round2digits(1.0))

    def test_build_region_3(self):
        text = "(hello)"
        root = _build_region(text)
        l = _convert_region_tree_to_word_list(root)
        self.assertTrue(l[0][0] == "")  # empty string
        self.assertTrue(round2digits(l[0][1]) == round2digits(1.0))
        self.assertTrue(l[1][0] == "hello")
        self.assertTrue(round2digits(l[1][1]) == round2digits(1.1))


if __name__ == '__main__':
    unittest.main()
