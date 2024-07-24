import os
import sys
import unittest

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.misc_utils import strip_directory_from_path_list_str


class TestMiscUtil(unittest.TestCase):

    def test_strip_directory_from_path_list_str(self):
        data = [
            # Input, expected
            (None, None),
            ("", ""),
            (",,,,", ",,,,"),
            ("a,,,,", "a,,,,"),
            (",a,,,", ",a,,,"),
            ("bar.safesensors,baz.safesensors,,", "bar.safesensors,baz.safesensors,,"),
            ("bar.safesensors,baz.safesensors", "bar.safesensors,baz.safesensors"),
            ("/foo/bar.safesensors,baz.safesensors", "bar.safesensors,baz.safesensors")
        ]

        for e in data:
            expected = e[1]
            actual = strip_directory_from_path_list_str(e[0])

        self.assertTrue(expected == actual)

if __name__ == '__main__':
    unittest.main()
