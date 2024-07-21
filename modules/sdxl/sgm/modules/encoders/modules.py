import os
if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

import sys
import logging
import math
import time
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import kornia
import numpy as np

import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from transformers import (ByT5Tokenizer, CLIPTextModel, CLIPTokenizer,
                          T5EncoderModel, T5Tokenizer)

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..")
SDXL_MODULE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path = [SDXL_MODULE_ROOT, MODULE_ROOT] + sys.path

import open_clip  # Cremage: this is in the code tree. Don't do pip install open_clip_torch
from sgm.modules.autoencoding.regularizers import DiagonalGaussianRegularizer
from sgm.modules.diffusionmodules.model import Encoder
from sgm.modules.diffusionmodules.openaimodel import Timestep
from sgm.modules.diffusionmodules.util import (extract_into_tensor,
                                              make_beta_schedule)
from sgm.modules.distributions.distributions import DiagonalGaussianDistribution
from sgm.util import (append_dims, autocast, count_params, default,
                     disabled_train, expand_dims_like, instantiate_from_config)
from .abstract_emb_model import AbstractEmbModel

# from clip.modeling_clip import CLIPTextModel as ModCLIPTextModel
from clip_sdxl.modeling_clip import CLIPTextModel as ModCLIPTextModel  # DEBUG FIXME CREMAGE

from cremage.utils.prompt_score_parser import generate_clip_embeddings_from_prompt
from cremage.utils.prompt_score_parser import generate_open_clip_embeddings_from_prompt

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# class AbstractEmbModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._is_trainable = None
#         self._ucg_rate = None
#         self._input_key = None

#     @property
#     def is_trainable(self) -> bool:
#         return self._is_trainable

#     @property
#     def ucg_rate(self) -> Union[float, torch.Tensor]:
#         return self._ucg_rate

#     @property
#     def input_key(self) -> str:
#         return self._input_key

#     @is_trainable.setter
#     def is_trainable(self, value: bool):
#         self._is_trainable = value

#     @ucg_rate.setter
#     def ucg_rate(self, value: Union[float, torch.Tensor]):
#         self._ucg_rate = value

#     @input_key.setter
#     def input_key(self, value: str):
#         self._input_key = value

#     @is_trainable.deleter
#     def is_trainable(self):
#         del self._is_trainable

#     @ucg_rate.deleter
#     def ucg_rate(self):
#         del self._ucg_rate

#     @input_key.deleter
#     def input_key(self):
#         del self._input_key


class GeneralConditioner(nn.Module):
    """
    Cremage note: Generic embedding generator wrapper.
    This is used to generate two text embeddings.
    """
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig],
                 lora_ranks=None,
                 lora_weights=None):
        super().__init__()
        embedders = []
        self.lora_ranks = lora_ranks
        self.lora_weights = lora_weights

        logger.debug("DEBUG Cremage: GeneralConditioner init")

        for n, embconfig in enumerate(emb_models):

            if embconfig.target.endswith("FrozenCLIPEmbedder") or \
                embconfig.target.endswith("FrozenOpenCLIPEmbedder2"):
                    embconfig["params"]["lora_ranks"] = lora_ranks
                    embconfig["params"]["lora_weights"] = lora_weights

            t_start = time.perf_counter()
            embedder = instantiate_from_config(embconfig)
            t_end = time.perf_counter()
            logger.debug(f"{embconfig.target} instantiation took {t_end-t_start} seconds")

            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            logger.debug(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders) # 5 embedders including FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None,
        embedding_dir=None  # Cremage change
    ) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)

                    if isinstance(embedder, FrozenCLIPEmbedder) or \
                       isinstance(embedder, FrozenOpenCLIPEmbedder2):
                        # Cremage: Call CLIP.
                        # this returns (3, 77, 1280) & (3, 1, 1280).
                        # Latter is for pool.
                        emb_out = embedder(batch[embedder.input_key],
                                           embedding_dir=embedding_dir)
                    else:
                        emb_out = embedder(batch[embedder.input_key])  # Cremage: Call CLIP. TODO: Add clip skip For open_clip, this returns (3, 77, 1280) & (3, 1, 1280). Latter is for pool.
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]  # e.g. for CLIP, [3, 77, 768] for bs=3
            for emb in emb_out:  # For open_clip, add non-pool embedding, then pool
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)

                # Cremage note
                # for open_clip, embedder outputs (for bs=3):
                #   non-pool: (3, 77, 1280)
                #   pool:(3, 1280)
                # non-pool is concatenated in output["crosattn"] with CLIP
                # CLIP embedding is (3, 77, 768) so after concat, it will become
                #  (3, 77, 2048)
                # pool gets stored in output["vector"]
                if out_key in output: # out_key is crossattn
                    output[out_key] = torch.cat(
                        (output[out_key], emb), self.KEY2CATDIM[out_key]
                    )
                else:  # first addition (e.g. CLIP)
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(
        self,
        batch_c: Dict,
        batch_uc: Optional[Dict] = None,
        force_uc_zero_embeddings: Optional[List[str]] = None,
        force_cond_zero_embeddings: Optional[List[str]] = None,
        embedding_dir:Optional[str] = None
    ):
        if force_uc_zero_embeddings is None:  # ['txt']
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c, force_cond_zero_embeddings,
                 embedding_dir=embedding_dir) # Cremage added embedding_dir
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings,
                 embedding_dir=embedding_dir)
        # uc shape: [1, 77, 2048]
        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class InceptionV3(nn.Module):
    """Wrapper around the https://github.com/mseitzer/pytorch-fid inception
    port with an additional squeeze at the end"""

    def __init__(self, normalize_input=False, **kwargs):
        super().__init__()
        from pytorch_fid import inception

        kwargs["resize_input"] = True
        self.model = inception.InceptionV3(normalize_input=normalize_input, **kwargs)

    def forward(self, inp):
        outp = self.model(inp)

        if len(outp) == 1:
            return outp[0].squeeze()

        return outp


class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x

    def forward(self, x):
        return x


class ClassEmbedder(AbstractEmbModel):
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device=os.environ.get("GPU_DEVICE", "cpu")):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ClassEmbedderForMultiCond(ClassEmbedder):
    def forward(self, batch, key=None, disable_dropout=False):
        out = batch
        key = default(key, self.key)
        islist = isinstance(batch[key], list)
        if islist:
            batch[key] = batch[key][0]
        c_out = super().forward(batch, key, disable_dropout)
        out[key] = [c_out] if islist else c_out
        return out


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/t5-v1_1-xxl", device=os.environ.get("GPU_DEVICE", "cpu"), max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenByT5Embedder(AbstractEmbModel):
    """
    Uses the ByT5 transformer encoder for text. Is character-aware.
    """

    def __init__(
        self, version="google/byt5-base", device=os.environ.get("GPU_DEVICE", "cpu"), max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = ByT5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

# marker

CLIP_BOS = 49406
CLIP_EOS = 49407
MAX_CLIP_SEQ_LEN_WITHOUT_BOS_EOS = 75
MAX_CLIP_SEQ_LEN = 77

class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from Hugging Face)
    
    Method calling sequence
    * init
    * freeze
    * encode
    * forward
    """

    LAYERS = ["last", "pooled", "hidden"]  # Cremage note: New in SDXL

    def __init__(self,
                 version="openai/clip-vit-large-patch14",
                 device=os.environ.get("GPU_DEVICE", "cpu"),
                 max_length=MAX_CLIP_SEQ_LEN,
                 freeze=True,  # Cremage note: New in SDXL
                 layer="last", # New in SDXL
                 layer_idx=None, # New in SDXL
                 always_return_pooled=False,  # New in SDXL
                 lora_ranks=None,
                 lora_weights=None):
        super().__init__()
        assert layer in self.LAYERS
        logger.debug("Initializing ModCLIPTextModel")
        logger.info(f"CLIP is initialized with lora_ranks: {lora_ranks}, lora_weights: {lora_weights}")
        logger.debug(f"CLIPTokenizer and ModCLIPTextModel connection to internet disabled : {local_files_only_value}")
        self.tokenizer = CLIPTokenizer.from_pretrained(version, local_files_only=local_files_only_value)
        self.transformer = ModCLIPTextModel.from_pretrained(version,
                                                            lora_ranks=lora_ranks,
                                                            lora_weights=lora_weights,
                                                            local_files_only=local_files_only_value)
        self.device = device
        self.max_length = max_length
        # New for SDXL start
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12
        # New for SDXL end

    def freeze(self):
        """
        Set requires_grad attribute to False for all model parameters.
        """
        logger.debug(" FrozenCLIPEmbedder freeze")        
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text: List[str], embedding_dir=None, clip_skip=1) -> torch.Tensor:
        """
        Processes a batch of text inputs and generates a CLIP embedding.

        If the number of tokens in an input text exceeds 77 (including BOS and EOS tokens), the input is divided into sequences of 75 tokens plus BOS and EOS tokens. Each of these sequences is then used to generate a CLIP embedding. These embeddings are concatenated along axis 1 to produce the final output.

        Args:
            text (List[str]): A batch of text inputs. All elements should be identical strings. It's assumed that the same prompt is used across the batch.
            embedding_dir (str): A directory where embedding files are located
            clip_skip (int): A clip skip value to specify the transformer block whose output to be used. See note below for more information.
        Returns:
            torch.Tensor: The output is a PyTorch tensor with the shape [batch_size, embedding_sequence_len, 768], where `embedding_sequence_len` is a multiple of 77, depending on how the input text is divided.

        Example:
            >>> embedder = FrozenCLIPEmbedder()
            >>> output = embedder.forward(["cute puppy ... (followed by long sequence of words)", <same as the first sequence>])
            Assuming the inputs require division into four sequences each, the output shape would be [2, 308, 768], corresponding to the batch size of 2, with the sequence length adjusted to fit the concatenated embeddings.

        Note:
            - The input list must contain identical strings.
            - This method is designed to handle input sequences longer than the maximum length allowed by the underlying CLIP model by dividing the input into smaller sequences, each processed separately. The division strategy ensures that each segment, except possibly the last, has exactly 75 tokens plus BOS and EOS, maximizing the utilization of the model's capacity.
            - If the division of tokens results in a segment shorter than 75 tokens, it's padded with EOS tokens to maintain consistency.
 
            - Clip skip allows the user to grab the intermediate hidden layer output of a
              CLIP Text Transformer model. There are twelve transformer blocks
              in CLIP Text Transformer model. By default, it grabs the output of the last one, but
              if you specify a number greater than 1, it grabs the intermediate output.

              Here is the mapping:
              clip skip, transformer output block (0-based index)
              1, 12 (last or top block)
              2, 11 (penultimate)
              ...
              12, 1 (first block)
              12, 0 (input embedding)                     
        """
        logger.debug(f"FrozenCLIPEmbedder forward. Clip_skip: {clip_skip}")
        
        # Get the batch size
        # Assumption is that the same prompt is used across batches       
        batch_size = len(text)
        single_text = text[0]
        logger.debug(f"In CLIP: single_text: {single_text}")
        # unprocessed embeding : List[embedding] where embeding is [77, 768]
        unprocessed_embeddings, eos_index_list = generate_clip_embeddings_from_prompt(
            self.tokenizer, 
            self.transformer, 
            embedding_dir, 
            single_text)
  
        embedding_list = list()
        pooler_list = list()
        for e in unprocessed_embeddings:
            outputs=self.transformer(input_embeddings=e, output_hidden_states=True)          
            assert len(outputs.hidden_states) == 13  # 13 for input embedding + 12 transformers
            transformer_block_index = 13 - clip_skip
            
            if self.layer == "last": # Code path for SD 1.5
                # z = outputs.last_hidden_state
                z = outputs.hidden_states[transformer_block_index]
                # Cremage, the behavior is the same as specifying layer_idx.
                # This is to use clip skip the same way as SD 1.5.
            elif self.layer == "pooled":  # New
                z = outputs.pooler_output[:, None, :]
            else: # New (self.layer is set to "hidden")
                z = outputs.hidden_states[self.layer_idx]  # Cremage: Code path layer_idx=11

            pooler_list.append(outputs.pooler_output)  # shape [1, 768]
            embedding_list.append(z)

        output = torch.concat(embedding_list, axis=1)  # [1, 385, 768]
        output = output.repeat(batch_size, 1, 1)

        if self.return_pooled:  # False
            pooler_output = torch.concat(pooler_list, axis=1)
            pooler_output = pooler_output.repeat(batch_size, 1, 1)
            return output, pooler_output
        return output

    def encode(self, text, embedding_dir=None, clip_skip=1):
        logger.debug(" FrozenCLIPEmbedder encode")
        return self(text, embedding_dir=embedding_dir, clip_skip=clip_skip)


class FrozenCLIPEmbedderOriginal(AbstractEmbModel):
    """
    Cremage note: Original SDXL method. Renamed to _Original.
    Uses the CLIP transformer encoder for text (from huggingface)
    """

    LAYERS = ["last", "pooled", "hidden"]  # Cremage note: New in SDXL

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device=os.environ.get("GPU_DEVICE", "cpu"),
        max_length=77,
        freeze=True,  # Cremage note: New in SDXL
        layer="last", # New in SDXL
        layer_idx=None, # New in SDXL
        always_return_pooled=False,  # New in SDXL
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS

        print("Cremage debug: FrozenCLIPEmbedder init")

        self.tokenizer = CLIPTokenizer.from_pretrained(version, local_files_only=local_files_only_value)
        self.transformer = CLIPTextModel.from_pretrained(version, local_files_only=local_files_only_value)

        print("Cremage debug: FrozenCLIPEmbedder init 2")

        self.device = device
        self.max_length = max_length

        # New for SDXL start
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12
        # New for SDXL end

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )  # Cremage note: output_hidden_states=self.layer == "hidden" is new for SDXL

        # New code for SDXL start
        if self.layer == "last": # Code path for SD 1.5
            z = outputs.last_hidden_state
        elif self.layer == "pooled":  # New
            z = outputs.pooler_output[:, None, :]
        else: # New
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:  # New
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text

    Cremage: Added to support prompt scoring and textual inversion embedding.
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device=os.environ.get("GPU_DEVICE", "cpu"),
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
        lora_ranks=None,
        lora_weights=None
    ):
        super().__init__()
        assert layer in self.LAYERS

        t_start = time.perf_counter()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
            lora_ranks=lora_ranks,
            lora_weights=lora_weights,
            # precision="fp16",  # Cremage TODO. Enable after checking the behavior on Mac.
            disable_loading_state_dict=True  # Do not load state dict, load later
        )
        t_end = time.perf_counter()
        logger.debug(f"open_clip.create_model_and_transforms took {t_end-t_start} seconds")

        # del model.visual
        
        
        self.model = model

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward_original(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]
        return z[self.layer]

    @autocast
    def forward(self, text, embedding_dir=None, clip_skip=1):
        # tokens = open_clip.tokenize(text)
        # z = self.encode_with_transformer(tokens.to(self.device))
        batch_size = len(text)
        # text contains ["puppy", "puppy", "puppy"] for batch size 3
        # so extract just "puppy" to pass a single string
        single_text = text[0]
        unprocessed_embeddings, eos_index_list = generate_open_clip_embeddings_from_prompt(
            self.model,
            embedding_dir,
            single_text #  prompt
            )
        # z = self.encode_embeddings_with_transformer(embeddings.to(self.device))

        # marker start

        embedding_list = list()
        pooler_list = list()
        # Process 77 embeddings at a time
        # eos_index_list[i] contains the EOS position for
        # the single sequence of 77.
        for i, e in enumerate(unprocessed_embeddings):
            z = self.encode_embeddings_with_transformer(e.to(self.device),
                                                        eos_index_list[i])
            embedding_list.append(z)

        # Now the list contains processed encoded embeddings

        if not self.return_pooled and self.legacy:
            # return z
            output = torch.concat(embedding_list, axis=1)  # [1, 385, 768]
            output = output.repeat(batch_size, 1, 1)
            return output

        if self.return_pooled:  # code path for SDXL 1.0
            assert not self.legacy
            output = torch.concat([e[self.layer] for e in embedding_list], axis=1)
            output = output.repeat(batch_size, 1, 1)  # e.g. (3, 77, 1280) for bs=3

            # Cremage note on pooling
            # SDXL expects pooling to be in [bs, hidden_dim] e.g. [3, 1280]
            # For each example, pooling is taken from the seq index where EOS is first found.
            # Since Cremage supports a prompt whose token length exceeds 77,
            # it can generate encoding that has more than 77 in the sequence length.
            # e.g. [3, 154, 1280].
            # However, each 77 tokens are processed one at a time, so
            # pooling is only valid fro the group of 77 at a time.
            # For example, if you want to encode 154 tokens, it goes through
            # a chunk of 77 tokens at a time.
            # The second feed of 77 tokens has no idea of the first 77 tokens
            # so the pool embedding at the EOS pos of the second feed
            # does not represent the first feed at all.
            # To overcome this, you need to take a look at the pool
            # for each iteration, and compute the mean.
            # So we have this list, [[1, 1280], [1, 1280] for a long prompt (77*2).
            # Stack to convert to a tensor first.
            tens = torch.concat([e["pooled"] for e in embedding_list], dim=0)
            assert tens.dim() == 2
            # This is now [2, 1280], so take the mean
            pooled = torch.mean(tens, axis=0) # Compute mean and drop 0th axis
            pooled = torch.unsqueeze(pooled, dim=0) # This is now (1, 1280)
            # Repeat for the batch.
            pooled = pooled.repeat(batch_size, 1)  # e.g. (3, 1280) for bs=3

            # return z[self.layer], z["pooled"]
            return output, pooled  # code path for SDXL 1.0

        output = torch.concat([e[self.layer] for e in embedding_list], axis=1)
        output = output.repeat(batch_size, 1, 1)

        # return z[self.layer]
        return output


    def encode_embeddings_with_transformer(self, embeddings,
                                           eos_index=0):
        """
        Added by Cremage
        """
        x = embeddings  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        if x.dim() == 2:
            x = torch.unsqueeze(x, 0)  # Add batch axis
            assert x.dim() == 3
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool_with_eos_index(o, eos_index)
            x["pooled"] = pooled
            return x


    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.model.text_projection
        )
        return x

    def pool_with_eos_index(self, x, eos_index):
        """
        Added by Cremage
        """
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), eos_index]
            @ self.model.text_projection
        )
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2Original(AbstractEmbModel):  # Original
    """
    Uses the OpenCLIP transformer encoder for text

    Cremage: FIXME to add support for embedding
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device=os.environ.get("GPU_DEVICE", "cpu"),
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model] (3, 77, 1280)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask) # x is a dict
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)  # (3, 77, 1280)
            pooled = self.pool(o, text)  # (3, 1280)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.model.text_projection
        )
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device=os.environ.get("GPU_DEVICE", "cpu"),
        max_length=77,
        freeze=True,
        layer="last",
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device=os.environ.get("GPU_DEVICE", "cpu"),
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        init_device=None,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device(default(init_device, "cpu")),
            pretrained=version,
        )
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer(
            "mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                )[:, None]
                * z
            )
            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.ucg_rate)
                            * torch.ones(tokens.shape[0], device=tokens.device)
                        ),
                        tokens,
                    )
                    * tokens
                )
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img):
        # if self.max_crops > 0:
        #    img = self.preprocess_by_cropping(img)
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                print(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        if self.output_tokens:
            return x, tokens
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEmbModel):
    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        device=os.environ.get("GPU_DEVICE", "cpu"),
        clip_max_length=77,
        t5_max_length=77,
    ):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(
            clip_version, device, max_length=clip_max_length
        )
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method="bilinear",
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
        wrap_video=False,
        kernel_size=1,
        remap_output=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None or remap_output
        if self.remap_output:
            print(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )
        self.wrap_video = wrap_video

    def forward(self, x):
        if self.wrap_video and x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, "b c t h w -> b t c h w")
            x = rearrange(x, "b t c h w -> (b t) c h w")

        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.wrap_video:
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T, c=C)
            x = rearrange(x, "b t c h w -> b c t h w")
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class LowScaleEncoder(nn.Module):
    def __init__(
        self,
        model_config,
        linear_start,
        linear_end,
        timesteps=1000,
        max_noise_level=250,
        output_size=64,
        scale_factor=1.0,
    ):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.model = instantiate_from_config(model_config)
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps, linear_start=linear_start, linear_end=linear_end
        )
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def forward(self, x):
        z = self.model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        z = z * self.scale_factor
        noise_level = torch.randint(
            0, self.max_noise_level, (x.shape[0],), device=x.device
        ).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = torch.nn.functional.interpolate(z, size=self.out_size, mode="nearest")
        return z, noise_level

    def decode(self, z):
        z = z / self.scale_factor
        return self.model.decode(z)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb


class GaussianEncoder(Encoder, AbstractEmbModel):
    def __init__(
        self, weight: float = 1.0, flatten_output: bool = True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.posterior = DiagonalGaussianRegularizer()
        self.weight = weight
        self.flatten_output = flatten_output

    def forward(self, x) -> Tuple[Dict, torch.Tensor]:
        z = super().forward(x)
        z, log = self.posterior(z)
        log["loss"] = log["kl_loss"]
        log["weight"] = self.weight
        if self.flatten_output:
            z = rearrange(z, "b c h w -> b (h w ) c")
        return log, z


class VideoPredictionEmbedderWithEncoder(AbstractEmbModel):
    def __init__(
        self,
        n_cond_frames: int,
        n_copies: int,
        encoder_config: dict,
        sigma_sampler_config: Optional[dict] = None,
        sigma_cond_config: Optional[dict] = None,
        is_ae: bool = False,
        scale_factor: float = 1.0,
        disable_encoder_autocast: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.encoder = instantiate_from_config(encoder_config)
        self.sigma_sampler = (
            instantiate_from_config(sigma_sampler_config)
            if sigma_sampler_config is not None
            else None
        )
        self.sigma_cond = (
            instantiate_from_config(sigma_cond_config)
            if sigma_cond_config is not None
            else None
        )
        self.is_ae = is_ae
        self.scale_factor = scale_factor
        self.disable_encoder_autocast = disable_encoder_autocast
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def forward(
        self, vid: torch.Tensor
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, dict],
        Tuple[Tuple[torch.Tensor, torch.Tensor], dict],
    ]:
        if self.sigma_sampler is not None:
            b = vid.shape[0] // self.n_cond_frames
            sigmas = self.sigma_sampler(b).to(vid.device)
            if self.sigma_cond is not None:
                sigma_cond = self.sigma_cond(sigmas)
                sigma_cond = repeat(sigma_cond, "b d -> (b t) d", t=self.n_copies)
            sigmas = repeat(sigmas, "b -> (b t)", t=self.n_cond_frames)
            noise = torch.randn_like(vid)
            vid = vid + noise * append_dims(sigmas, vid.ndim)

        with torch.autocast("cuda", enabled=not self.disable_encoder_autocast):
            n_samples = (
                self.en_and_decode_n_samples_a_time
                if self.en_and_decode_n_samples_a_time is not None
                else vid.shape[0]
            )
            n_rounds = math.ceil(vid.shape[0] / n_samples)
            all_out = []
            for n in range(n_rounds):
                if self.is_ae:
                    out = self.encoder.encode(vid[n * n_samples : (n + 1) * n_samples])
                else:
                    out = self.encoder(vid[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)

        vid = torch.cat(all_out, dim=0)
        vid *= self.scale_factor

        vid = rearrange(vid, "(b t) c h w -> b () (t c) h w", t=self.n_cond_frames)
        vid = repeat(vid, "b 1 c h w -> (b t) c h w", t=self.n_copies)

        return_val = (vid, sigma_cond) if self.sigma_cond is not None else vid

        return return_val


class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):
    def __init__(
        self,
        open_clip_embedding_config: Dict,
        n_cond_frames: int,
        n_copies: int,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.open_clip = instantiate_from_config(open_clip_embedding_config)

    def forward(self, vid):
        vid = self.open_clip(vid)
        vid = rearrange(vid, "(b t) d -> b t d", t=self.n_cond_frames)
        vid = repeat(vid, "b t d -> (b s) t d", s=self.n_copies)

        return vid
