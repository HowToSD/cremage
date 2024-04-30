import os
import sys
import unittest

import numpy as np
import torch
from diffusers.pipelines.controlnet import multicontrolnet
from PIL import Image

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..") 
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path = [MODULE_ROOT] + sys.path


from ip_adapter.attention_processor import CNAttnProcessor2_0
from ip_adapter.utils import is_torch2_available
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus, reshape_image_embeddings
from ip_adapter.ip_adapter_faceid import generate_face_embedding_from_image
from ip_adapter.resampler import reshape_tensor
from ip_adapter.resampler import PerceiverAttention

class TestIPAdapter(unittest.TestCase):
    def test_is_torch2_available(self):
      
        self.assertTrue(is_torch2_available())

    def test_image_embedding_generator(self):
        """
        image_encoder_path = "https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder",
        For FaceID,it's different per https://huggingface.co/h94/IP-Adapter-FaceID
        """
        ipa = IPAdapterFaceIDPlus(
            None, # pipe
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            ip_ckpt="/media/pup/ssd2/recoverable_data/sd_models/ControlNet/ip-adapter-faceid-plusv2_sd15.bin", 
            device="cuda", 
            lora_rank=128,
            num_tokens=4,
            torch_dtype=torch.float16)
        self.assertTrue(ipa is not None)

        self.assertTrue(ipa.image_encoder.config.hidden_size == 1280)  # vector size at each seq pos

        # Check processor to make sure that it resizes
        image = torch.rand((1, 768, 512, 3)).half()  # h, w
        clip_image = ipa.clip_image_processor(images=image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(ipa.device, dtype=ipa.torch_dtype)
        self.assertTrue(clip_image.shape == (1, 3, 224, 224))

        # Encode to 1024d embedding
        image_embedding = ipa.image_encoder(clip_image)
        self.assertTrue(image_embedding.image_embeds.shape == (1, 1024))  # torch.float16

        # Encode to 1280d embedding instead of 1024d. This is used in the Plus model
        image_embedding = ipa.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        print(image_embedding.shape)
        self.assertTrue(image_embedding.shape == (1, 257, 1280))

    def test_reshape_tensor(self):
        x = torch.rand((2, 77, 8 * 64))  # B, Seq, num_heads * head_size
        x = reshape_tensor(x, 8)  # heads
        self.assertTrue(x.shape == (2, 8, 77, 64))

    def test_perceiver_attention(self):
        # Prepare inputs
        x = torch.rand((2, 77, 8 * 64))  # B, Seq, num_heads * head_size
        latents = torch.rand((2, 77, 8 * 64))  # B, Seq, num_heads * head_size

        # Compute attention between x and latents
        attn = PerceiverAttention(dim=8 * 64, dim_head=64, heads=8)
        outputs = attn(x, latents)
        self.assertTrue(outputs.shape == (2, 77, 8 * 64))  # same shape as input

    def test_image_embedding_mixer_model(self):
        """
        Tests a model that mixes two face embeddings.
        """
        # Prepare inputs
        face_id_embeddings = torch.rand((1, 512)).to("cuda").half()  # Cremage CHECKME

        # Cremage note: CLIP embedding used here is NOT the regular embedding.
        # This is grabbed from the intermediate transformer outputs [-2].
        # Regular Vision embedding is 1024d.
        clip_embeddings = torch.rand((1, 257, 1280)).to("cuda").half()
        
        ipa = IPAdapterFaceIDPlus(
            None, # pipe
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            ip_ckpt="/media/pup/ssd2/recoverable_data/sd_models/ControlNet/ip-adapter-faceid-plusv2_sd15.bin", 
            device="cuda", 
            lora_rank=128,
            num_tokens=4,
            torch_dtype=torch.float16)
        self.assertTrue(ipa is not None)

        mixer_model = ipa.image_proj_model
        embedding = mixer_model(face_id_embeddings, clip_embeddings)
        self.assertTrue(embedding.shape == (1, 4, 768))

    def test_get_image_embeds(self):
        """
        Tests a model that mixes two face embeddings.
        """
        # Prepare inputs
        face_id_embeddings = torch.rand((1, 512)).to("cuda").half()  # Cremage CHECKME
        pil_face_image = Image.fromarray(np.random.rand(512, 768))
        self.assertTrue(pil_face_image.size == (768, 512))  # width, height
        # Instantiate IP Adapter
        ipa = IPAdapterFaceIDPlus(
            None, # pipe
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            ip_ckpt="/media/pup/ssd2/recoverable_data/sd_models/ControlNet/ip-adapter-faceid-plusv2_sd15.bin", 
            device="cuda", 
            lora_rank=128,
            num_tokens=4,
            torch_dtype=torch.float16)
        image_prompt_embeds, uncond_image_prompt_embeds = \
            ipa.get_image_embeds(face_id_embeddings, pil_face_image, s_scale=1.0, shortcut=False)
        self.assertTrue(image_prompt_embeds.shape == (1, 4, 768))

        num_samples = 3
        image_prompt_embeds, uncond_image_prompt_embeds = \
            reshape_image_embeddings(num_samples, image_prompt_embeds, uncond_image_prompt_embeds)
        self.assertTrue(image_prompt_embeds.shape == (3, 4, 768))

    def test_get_image_embeds_wrapper(self):
        """
        Tests a model that mixes two face embeddings.
        """
        num_samples = 3
        input_file = os.path.join(os.path.dirname(__file__), "..", "resources", "images", "photo_input_example.jpg")
        self.assertTrue(os.path.exists(input_file))
        image_prompt_embeds, uncond_image_prompt_embeds = generate_face_embedding_from_image(
                input_file,
                face_model_path="/media/pup/ssd2/recoverable_data/sd_models/ControlNet/ip-adapter-faceid-plusv2_sd15.bin",
                batch_size=num_samples)
        self.assertTrue(image_prompt_embeds.shape == (3, 4, 768))


if __name__ == '__main__':
    unittest.main()
