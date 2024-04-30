import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..") 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
LDM_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs/ldm/configs")
sys.path = [MODULE_ROOT] + sys.path

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from cremage.utils.ml_utils import load_model, face_id_model_weight_to_sd_15_model_weight

class TestIPAdapterModelLoading(unittest.TestCase):
    def test_no_lora_no_control_net(self):
        current_config = OmegaConf.load(f"{LDM_CONFIG_DIR}/stable-diffusion/v1-inference.yaml")
        current_config = current_config.model

        lora_ranks = []
        lora_weights = []
        lora_ranks = [128] + lora_ranks
        lora_weights = [1.0] + lora_weights
        current_config["params"]["cond_stage_config"]["params"]["lora_ranks"] = []
        current_config["params"]["cond_stage_config"]["params"]["lora_weights"] = []
        current_config["params"]["unet_config"]["params"]["ipa_scale"] = 1.0
        current_config["params"]["unet_config"]["params"]["ipa_num_tokens"] = 4
        current_config["params"]["unet_config"]["params"]["lora_ranks"] = lora_ranks
        current_config["params"]["unet_config"]["params"]["lora_weights"] = lora_weights
        ldm_model = instantiate_from_config(current_config).to("cuda")
        face_model = load_model("/media/pup/ssd2/recoverable_data/sd_models/ControlNet/ip-adapter-faceid-plusv2_sd15.bin")
        ldm_model_sd = ldm_model.state_dict()
        # for k, v in ldm_model_sd.items():
        #     if k.find("ipa") >= 0:
        #         print(k)

        missing_weight = 0
        mismatch = 0
        for k, v in face_model["ip_adapter"].items():
            sdw = face_id_model_weight_to_sd_15_model_weight(k)
            if sdw not in ldm_model_sd:
                missing_weight += 1
                print(f"Missing: {sdw}")
            if ldm_model_sd[sdw].shape != v.shape:
                print(f"weight mismatch for {k}: {v.shape} vs {ldm_model_sd[sdw].shape}")
                mismatch += 1

            # Now actually load weight
            # Split the parameter name to separate the submodule path and the parameter name
            path_parts = sdw.split('.')
            submodule = ldm_model
            for part in path_parts[:-1]:  # Navigate to the submodule
                submodule = getattr(submodule, part)
            
            # Now, set the parameter on the submodule
            setattr(submodule, path_parts[-1], torch.nn.Parameter(v))

        self.assertTrue(missing_weight == 0)
        self.assertTrue(mismatch == 0)


if __name__ == '__main__':
    unittest.main()
