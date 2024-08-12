import os
import logging
from typing import List
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from .utils import is_torch2_available, get_generator

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

USE_DAFAULT_ATTN = False # should be True for visualization_attnmap
if is_torch2_available() and (not USE_DAFAULT_ATTN):
    from .attention_processor_faceid import (
        LoRAAttnProcessor2_0 as LoRAAttnProcessor,
    )
    from .attention_processor_faceid import (
        LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor,
    )
else:
    from .attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from .resampler import PerceiverAttention, FeedForward

def reshape_image_embeddings(num_samples, image_prompt_embeds, uncond_image_prompt_embeds):
    """
    Reshape image_prompt_embeddings and unconditional image prompt embeddings
    based on the number of samples.

    This was taken from the IP adapter face IP plus class' generate method and moved here by Cremage.
    """
    bs_embed, seq_len, _ = image_prompt_embeds.shape
    image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
    image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
    return image_prompt_embeds, uncond_image_prompt_embeds


class FacePerceiverResampler(torch.nn.Module):
    """
    A Transformer model to mix two face embeddings.
    Uses 4 Transformer blocks.

    Cremage note: Docstring was added by Cremage.
    """
    def __init__(
        self,
        *,  # Cremage: the rest will be keyword arguments
        dim=768,  # Cremage CHECK: This should be hidden dimension at each seq pos. Double-check.
        depth=4,  # Num transformer blocks
        dim_head=64,
        heads=16,  # 64 * 16 = 1024
        embedding_dim=1280,  # CLIP Vision hidden embedding size
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()

        self.proj_in = torch.nn.Linear(embedding_dim, dim)  # in: 1280
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)  # out: 768
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):  # 4 transformer blocks
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        # Attention
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),

                        # FF
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        """
        
        Input projection
        4 Transformer blocks
        Output projection
        
        Cremage note: Docstring was added by Cremage.
        """
        x = self.proj_in(x)  # Projection to transformer blocks
        for attn, ff in self.layers:  # 4 transformer blocks
            latents = attn(x, latents) + latents  # Add skip
            latents = ff(latents) + latents  # Add skip
        latents = self.proj_out(latents)  # Projection out from transformer blocks
        return self.norm_out(latents)  # Normalize it


class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class ProjPlusModel(torch.nn.Module):
    """
    Two face embeddings mixer model.
    Takes both InsightFace's face embedding and CLIP's face embedding and
    produces an embedding.  Consider this as embeddings mixer.

    Note: Docstring added by Cremage
    """
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),  # 512 to 1024
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),  # 1024 -> 768*4
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,  # 1280
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        
    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        """
        Generates an image embedding from Insight Face and CLIP face embeddings.

        Args:
            id_embeds: InsightFace's face embedding. Shape should be (1, 512) (Cremage CHECK)
            clip_embeds: CLIP's face embedding. Shape should be (1, 257, 1280)
        Returns:
            Face embedding: Shape should be (1, 4, 768)
        Note: Docstring added by Cremage
        """
        # Preprocess InsightFace's face embedding
        #  1.projection, 2.normalization
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)  # -1, 4, 768
        x = self.norm(x)

        # Now process both InsightFace face embedding and CLIP embedding
        # in transformer
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:  # Add skip connection if shortcut is on
            out = x + scale * out
        return out


class IPAdapterFaceID:
    def __init__(self, sd_pipe, ip_ckpt, device, lora_rank=128, num_tokens=4, torch_dtype=torch.float16):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.lora_rank = lora_rank
        self.num_tokens = num_tokens
        self.torch_dtype = torch_dtype

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.torch_dtype)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype)
            else:
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank, num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.torch_dtype)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, faceid_embeds):
        
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        image_prompt_embeds = self.image_proj_model(faceid_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

       
        num_prompts = faceid_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterFaceIDPlus:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, lora_rank=128, num_tokens=4, torch_dtype=torch.float16):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.lora_rank = lora_rank
        self.num_tokens = num_tokens
        self.torch_dtype = torch_dtype

        # self.pipe = sd_pipe.to(self.device)
        if sd_pipe:  # Cremage change
            self.pipe = sd_pipe.to(self.device)
        else:
            self.pipe = None

        if sd_pipe:
            self.set_ip_adapter()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_path, local_files_only=local_files_only_value).to(os.environ.get("GPU_DEVICE", "cpu"))
        self.clip_image_processor = CLIPImageProcessor()

        # image proj model to mix InsightFace's face embedding and CLIP face embedding
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        """
        Initialize the face embedding mixing model.

        Returns:
            image projection model (nn.Module)
        Note: Docstring added by Cremage
        """
        DUMMY_VALUE = 768  # Cremage change
        cross_attention_dim = self.pipe.unet.config.cross_attention_dim if self.pipe else DUMMY_VALUE

        # Cremage: Load two face image embeddings mixer model
        image_proj_model = ProjPlusModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
            num_tokens=self.num_tokens,
        ).to(os.environ.get("GPU_DEVICE", "cpu"))
        return image_proj_model

    def set_ip_adapter(self):
        """
        Replace UNet's attention module with a custom attention module for IP Adapter
        

        Steps
        Iterate all the UNet's weights that start with attn1.processor
        Instantiate LoRAAttnProcessor or LoRAIPAttnProcessor for each and set in a dict

        Note: Docstring was added by Cremage.
        """
        unet = self.pipe.unet
        attn_procs = {}  # Cremage: New UNet module dict
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype)
            else:
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank, num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.torch_dtype)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self):
        # Cremage: Load IP Adapter model's state dict
        #   There are two keys we care about:
        #   * image_proj: To mix two image embeddings
        #   * ip_adapter: To be used in cross-attention in UNet
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            # Cremage: If IP adapter model is in .safetensors format
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            # Cremage: If IP adapter model is not in .safetensors format
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        
        self.image_proj_model.load_state_dict(state_dict["image_proj"])

        # Cremage: UNet attention weights
        if self.pipe:
            ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, faceid_embeds, face_image, s_scale, shortcut):
        """
        Args:
            faceid_embeds (torch.tensor): A face ID embedding generated by InsightFace.
            face_image (Image): A face image in PIL format

        Note: Docstring added by Cremage
        """
        # Check to see if the face image in the PIL format
        if isinstance(face_image, Image.Image):
            pil_image = [face_image]  # Put it in a list

        # Preprocess the face PIL image including resizing to 224x224
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        
        # Move to CUDA
        clip_image = clip_image.to(os.environ.get("GPU_DEVICE", "cpu"))
        
        # Generate [1, 257, 1280] embedding
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]

        # Generate the embedding for a black image
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        
        # Move InsightFace's face embedding to CUDA
        faceid_embeds = faceid_embeds.to(os.environ.get("GPU_DEVICE", "cpu"))

        # Process both FaceID and CLIP embedding in the projection model
        image_prompt_embeds = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)
        
        # Do the same for black images
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=shortcut, scale=s_scale)
        
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        face_image=None,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        s_scale=1.0,
        shortcut=False,
        **kwargs,
    ):
        self.set_scale(scale)

       
        num_prompts = faceid_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # Combined image embeddings
        # First is the mixture of InsightFace and CLIP VISION
        # Second are black images processed the same way
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds, face_image, s_scale, shortcut)

        # Cremage change
        # bs_embed, seq_len, _ = image_prompt_embeds.shape
        # image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        # image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        # uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        # uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        image_prompt_embeds, uncond_image_prompt_embeds = \
            reshape_image_embeddings(num_samples, image_prompt_embeds, uncond_image_prompt_embeds)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            # Cremage: Combine text embeddings with image embeddings
            #   image_prompt_embeds shape is [bs, 4, 768]
            #   Typically, text embedding shape is [bs, 77, 768]
            #   This is effectively appending appending the 4 seq embeddings
            #   to the end of text embedding.
            #   Therefore, you need to refactor transformer blocks
            #   so that it can handle additional 4 seq-long embeddings.
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterFaceIDXL(IPAdapterFaceID):
    """SDXL"""

    def generate(
        self,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = faceid_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterFaceIDPlusXL(IPAdapterFaceIDPlus):
    """SDXL"""

    def generate(
        self,
        face_image=None,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        s_scale=1.0,
        shortcut=True,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = faceid_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds, face_image, s_scale, shortcut)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images

        return images

def generate_face_embedding_from_image(face_input_img_path: str,
                                       face_model_path: str,
                                       batch_size:str=1):
    """
    Generate face embeddings from the input image using the face model.

    Face analysis code was adapted from [1].

    Args:
    face_input_img_path (str): The file path of the input face image.
    face_model_path (str): The file path of the face model.
    
    Returns
        Tuple of face embedding and unconditional face embedding.

    References
    [1] https://huggingface.co/h94/IP-Adapter-FaceID
    """
    ipa = IPAdapterFaceIDPlus(
            None, # pipe
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            ip_ckpt=face_model_path,
            device=os.environ.get("GPU_DEVICE", "cpu"),
            lora_rank=128,
            num_tokens=4,
            torch_dtype=torch.float16)

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(face_input_img_path)
    faces = app.get(image)
    
    # Cremage note if there is not enough margin aruond
    #   the face, detection can fail, so retry after adding some margin
    if len(faces) == 0:
        logger.info("Retrying detection with additional margin around the face")
        # Add a 200px border (margin) to all sides
        top, bottom, left, right = 200, 200, 200, 200
        image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        faces = app.get(image)
        if len(faces) == 0:
            logger.warn("No face was detected")
            return None, None

    face_id_embeddings = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    pil_face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face

    image_prompt_embeds, uncond_image_prompt_embeds = \
        ipa.get_image_embeds(face_id_embeddings, pil_face_image, s_scale=1.0,
                             shortcut=True  # True means that this is v2 embedding mixing model.
                             )

    num_samples = batch_size
    image_prompt_embeds, uncond_image_prompt_embeds = \
        reshape_image_embeddings(num_samples, image_prompt_embeds, uncond_image_prompt_embeds)
    return image_prompt_embeds, uncond_image_prompt_embeds


