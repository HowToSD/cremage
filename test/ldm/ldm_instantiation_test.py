import os
import sys
import unittest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..") 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
LDM_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs/ldm/configs")
sys.path = [MODULE_ROOT] + sys.path
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

class LDMTest(unittest.TestCase):
    def test_instantiate_from_config(self):
        current_config = OmegaConf.load(f"{LDM_CONFIG_DIR}/stable-diffusion/v1-inference.yaml")
        current_config = current_config.model
        current_config["params"]["cond_stage_config"]["params"]["lora_ranks"] = []
        current_config["params"]["cond_stage_config"]["params"]["lora_weights"] = []
        ldm_model = instantiate_from_config(current_config).to("cuda")
        sd = ldm_model.state_dict()
        
        # Spot check weight for each model
        self.assertTrue("alphas_cumprod" in sd)
        self.assertTrue("model.diffusion_model.input_blocks.1.0.in_layers.0.weight" in sd)
        self.assertTrue("first_stage_model.encoder.mid.block_2.conv2.bias" in sd)
        self.assertTrue("cond_stage_model.transformer.text_model.final_layer_norm.bias" in sd)


if __name__ == '__main__':
    unittest.main()
