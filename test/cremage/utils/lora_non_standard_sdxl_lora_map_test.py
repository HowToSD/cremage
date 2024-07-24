import os
import sys
import unittest

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.lora_loader import map_non_standard_lora_key_to_standard_lora_key


class TestNonStandardSdxlLora(unittest.TestCase):

    def test_map(self):
        data = [
            ("lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha",
             "lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha"),  # same
            ("lora_unet_down_blocks_1_attentions_0_proj_in.alpha",
             "lora_unet_input_blocks_4_1_proj_in.alpha"),
            ("lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.alpha",
             "lora_unet_input_blocks_5_1_transformer_blocks_0_attn1_to_k.alpha"),
            ("lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.alpha",
             "lora_unet_input_blocks_7_1_transformer_blocks_0_attn1_to_k.alpha"),
            ("lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.alpha",
             "lora_unet_input_blocks_8_1_transformer_blocks_0_attn1_to_k.alpha"),
            ("lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.alpha",
             "lora_unet_input_blocks_8_1_transformer_blocks_0_attn1_to_k.alpha"),
            ("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.alpha",
             "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.alpha"),
            ("lora_unet_mid_block_attentions_0_proj_in.alpha",
             "lora_unet_middle_block_1_proj_in.alpha"),
            ("lora_unet_up_blocks_0_attentions_0_proj_in.alpha",
             "lora_unet_output_blocks_0_1_proj_in.alpha"),
            ("lora_unet_up_blocks_0_attentions_1_proj_in.alpha",
             "lora_unet_output_blocks_1_1_proj_in.alpha"),
            ("lora_unet_up_blocks_0_attentions_2_proj_in.alpha",
             "lora_unet_output_blocks_2_1_proj_in.alpha"),
            ("lora_unet_up_blocks_1_attentions_0_proj_in.alpha",
             "lora_unet_output_blocks_3_1_proj_in.alpha"),
            ("lora_unet_up_blocks_1_attentions_1_proj_in.alpha",
             "lora_unet_output_blocks_4_1_proj_in.alpha"),
            ("lora_unet_up_blocks_1_attentions_2_proj_in.alpha",
             "lora_unet_output_blocks_5_1_proj_in.alpha"),
        ]

        for e in data:
            actual = map_non_standard_lora_key_to_standard_lora_key(e[0])
            self.assertTrue(e[1] == actual, f"Actual:{actual}, expected:{e[1]}")


if __name__ == '__main__':
    unittest.main()
