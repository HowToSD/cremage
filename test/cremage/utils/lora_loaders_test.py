import os
import sys
import unittest

import numpy as np
import PIL

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
TEST_DATA_ROOT = os.path.join(PROJECT_ROOT, "test", "data")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.lora_loader import load_loras
from cremage.utils.lora_loader import map_sdxl_lora_weight_name_to_mode_weight_name

# FIXME
test_file_name = "/media/pup/ssd2/recoverable_data/sd_models/Lora_sdxl/howtosd_alpha_model_v2y.safetensors"

class TestLoraLoader(unittest.TestCase):


    # def test_1(self):
    #     name = test_file_name
    #     loras, ranks, weights = load_loras(None, None, model_type="SDXL")
    #     self.assertTrue(len(loras) == 0)
    #     self.assertTrue(len(ranks) == 0)
    #     self.assertTrue(len(weights) == 0)

    # def test_2(self):
    #     name = test_file_name
    #     loras, ranks, weights = load_loras("", "", model_type="SDXL")
    #     self.assertTrue(len(loras) == 0)
    #     self.assertTrue(len(ranks) == 0)
    #     self.assertTrue(len(weights) == 0)

    # def test_3(self):
    #     name = test_file_name
    #     loras, ranks, weights = load_loras(f"{name}", f"1.0", model_type="SDXL")
    #     self.assertTrue(len(loras) == 1)
    #     self.assertTrue(len(ranks) == 1 )
    #     self.assertTrue(len(weights) == 1)
 
     def test_4(self):
        model_sd_text = os.path.join(TEST_DATA_ROOT, "lora_weight_in_model_sdxl.txt")
        lora_sd_text = os.path.join(TEST_DATA_ROOT, "lora_weight_in_lora_sdxl.txt")
        
        with open(model_sd_text, "r") as f:
            model_sd_keys = f.read()
            model_sd_keys = model_sd_keys.split("\n")

        with open(lora_sd_text, "r") as f:
            lora_sd_keys = f.read()
            lora_sd_keys = lora_sd_keys.split("\n")

        for sd_lora_key in lora_sd_keys:
            if sd_lora_key == "":
                continue
            model_sd_key = map_sdxl_lora_weight_name_to_mode_weight_name(sd_lora_key, 0)
            if model_sd_key not in model_sd_keys:
                print(f"Not found {model_sd_key}")

if __name__ == '__main__':
    unittest.main()
