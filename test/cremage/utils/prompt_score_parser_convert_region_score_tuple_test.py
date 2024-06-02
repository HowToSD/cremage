import os
import sys
import unittest

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TEST_ROOT = os.path.join(PROJECT_ROOT, "test")
sys.path = [MODULE_ROOT, TEST_ROOT] + sys.path
from cremage.utils.prompt_score_parser import _convert_region_score_tuple_list_to_word_list
# Test package import
from cremage.utils.test_utils import round2digits

class TestPromptScoreConvertRegionScoreTuple(unittest.TestCase):

    def test_1(self):
        source = [
            ("hello", 1.0)
        ]
        output = _convert_region_score_tuple_list_to_word_list(source)
        self.assertTrue(source[0][0] == output[0][0])
        self.assertTrue(round2digits(source[0][1]) == round2digits(output[0][1]))

    def test_1(self):
        source = [
            ("hello world", 1.0)
        ]
        output = _convert_region_score_tuple_list_to_word_list(source)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))
        self.assertTrue(output[1][0] == "world")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))

    def test_2(self):
        source = [
            ("""hello 
             world""", 1.0)
        ]
        output = _convert_region_score_tuple_list_to_word_list(source)
        self.assertTrue(output[0][0] == "hello")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))
        self.assertTrue(output[1][0] == "world")
        self.assertTrue(round2digits(output[0][1]) == round2digits(1.0))

if __name__ == '__main__':
    unittest.main()
