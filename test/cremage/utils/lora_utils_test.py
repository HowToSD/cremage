import os
import sys
import unittest

import numpy as np
import PIL

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.lora_utils import LORA_WEIGHTS
from cremage.utils.lora_utils import is_valid_lora_weight_name
from cremage.utils.lora_utils import _parse_sd_weight_for_lora_weight
from cremage.utils.lora_utils import sd_weight_to_lora_weight
from cremage.utils.lora_utils import _generate_lora_weight_name_from_parsed_sd_weight
from cremage.utils.sd15_weight_list_with_lora import SD15_WEIGHT_LIST


def extract_lora_weight_names_from_sd_weight_list():
    sd_weight_names = SD15_WEIGHT_LIST.split("\n")
    sd_weight_names = [n.strip() for n in sd_weight_names]
    sd_weight_names = filter(lambda l: len(l) > 0 and (l.endswith("alpha") or l.find("lora") >=0), sd_weight_names)
    return sd_weight_names

class TestLoraUtil(unittest.TestCase):


    def test_is_valid_lora_weight_name_1(self):
        """
        Check if the weight name is a valid LoRA weight name in the LoRA model.
        This is done by checking against the complete list of parameters for the LoRA model.
        Note that this is not to check the LoRA weight in the SD model.
        """
        name = "foo"
        actual = is_valid_lora_weight_name(name)
        expected = False
        self.assertTrue(actual == expected)

    def test_is_valid_lora_weight_name_all(self):
        for name in LORA_WEIGHTS:
            actual = is_valid_lora_weight_name(name)
            expected = True
            self.assertTrue(actual == expected)
        
    def test_conversion(self):
        """
        Checks mapping of LoRA weights in the SD model to the Lora weights in the
        LoRA model.
        Check to see if all LoRA weights in the LoRA model were mapped at the end.
        """
        lora_weights_dict = {
            k:0 for k in LORA_WEIGHTS
        }

        for w in extract_lora_weight_names_from_sd_weight_list():
            weight_dict = _parse_sd_weight_for_lora_weight(w)
            lora_weight = _generate_lora_weight_name_from_parsed_sd_weight(weight_dict)
            lora_weights_dict[lora_weight] += 1

        # Check to see if all key has value 1 as the test case is for loading 1 LoRA
        for k, v in lora_weights_dict.items():
            self.assertTrue(v == 1)
            
    def test_conversion_2(self):
        """Same test as conversion 1 but is using a wrapper method."""
        sd_weight_names = extract_lora_weight_names_from_sd_weight_list()

        lora_weights_dict = {
            k:0 for k in LORA_WEIGHTS
        }

        for w in sd_weight_names:
            lora_weight = sd_weight_to_lora_weight(w)
            lora_weights_dict[lora_weight] += 1

        for k, v in lora_weights_dict.items():
            self.assertTrue(v == 1)

if __name__ == '__main__':
    unittest.main()
