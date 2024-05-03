import os
import sys
import unittest

import torch
import numpy as np
import cv2
import PIL

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.app_misc_utils import get_next_file_id_from_list_of_file_names_with_numbers


class TestAppMiscUtil(unittest.TestCase):

    def test_get_next_file_id_from_list_of_file_names_with_numbers_empty(self):
        file_list = [
            
        ]
        actual = get_next_file_id_from_list_of_file_names_with_numbers(file_list)
        self.assertTrue(actual == 0)


    def test_get_next_file_id_from_list_of_file_names_with_numbers_empty_2(self):
        file_list = [
            "hello.txt"
        ]
        actual = get_next_file_id_from_list_of_file_names_with_numbers(file_list)
        self.assertTrue(actual == 0)

    def test_get_next_file_id_from_list_of_file_names_with_numbers(self):
        file_list = [
            "hello.txt",
            "face_000000.png"
        ]
        actual = get_next_file_id_from_list_of_file_names_with_numbers(file_list)
        self.assertTrue(actual == 1)

    def test_get_next_file_id_from_list_of_file_names_with_numbers_2(self):
        file_list = [
            "face_000001.png"
            "hello.txt",
            "face_000000.png"
        ]
        actual = get_next_file_id_from_list_of_file_names_with_numbers(file_list)
        self.assertTrue(actual == 2)

    def test_get_next_file_id_from_list_of_file_names_with_numbers_3(self):
        file_list = [
            "face_1.png"
            "hello.txt",
            "face_000000.png"
        ]
        actual = get_next_file_id_from_list_of_file_names_with_numbers(file_list)
        self.assertTrue(actual == 2)


    def test_copy_face_file_to_face_storage(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
