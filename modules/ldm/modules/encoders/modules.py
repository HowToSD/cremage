"""

# Major changes from the original code
* Support of a long promt exceeding 77 tokens using the method proposed by A1111.
"""
import os
import sys
from typing import List
import logging

import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
# from transformers import CLIPTokenizer, CLIPTextModel
from transformers import CLIPTokenizer  # HINADA Change

import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from clip.modeling_clip import CLIPTextModel as ModCLIPTextModel

MODULE_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path = [MODULE_ROOT] + sys.path
from cremage.utils.prompt_score_parser import generate_clip_embeddings_from_prompt

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

if os.environ.get("ENABLE_HF_INTERNET_CONNECTION") == "1":
    local_files_only_value=False
else:
    local_files_only_value=True

CLIP_BOS = 49406
CLIP_EOS = 49407
MAX_CLIP_SEQ_LEN_WITHOUT_BOS_EOS = 75
MAX_CLIP_SEQ_LEN = 77

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device=os.environ.get("GPU_DEVICE", "cpu")):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device=os.environ.get("GPU_DEVICE", "cpu"), vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device=os.environ.get("GPU_DEVICE", "cpu"),use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)
    
    Method calling sequence
    * init
    * freeze
    * encode
    * forward
    """
    def __init__(self,
                 version="openai/clip-vit-large-patch14",
                 device=os.environ.get("GPU_DEVICE", "cpu"),
                 max_length=MAX_CLIP_SEQ_LEN,
                 lora_ranks=None,
                 lora_weights=None):
        super().__init__()
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
        self.freeze()

    def freeze(self):
        """
        Set requires_grad attribute to False for all model parameters.
        """
        logger.debug(" FrozenCLIPEmbedder freeze")        
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward_old(self, text):

        logger.debug(" FrozenCLIPEmbedder forward")        
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,  # Set to 77 for CLIP
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def forward_v2(self, text: List[str], clip_skip=1) -> torch.Tensor:
        """
        Processes a batch of text inputs and generates a CLIP embedding.

        If the number of tokens in an input text exceeds 77 (including BOS and EOS tokens), the input is divided into sequences of 75 tokens plus BOS and EOS tokens. Each of these sequences is then used to generate a CLIP embedding. These embeddings are concatenated along axis 1 to produce the final output.

        Args:
            text (List[str]): A batch of text inputs. All elements should be identical strings. It's assumed that the same prompt is used across the batch.
            clip_skip: int: A clip skip value to specify the transformer block whose output to be used. See note below for more information.
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
        batch_encoding = self.tokenizer(
            single_text,
            truncation=True,
            max_length=self.max_length,  # Set to 77 for CLIP
            return_length=True,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt")
        num_truncated_tokens = batch_encoding['num_truncated_tokens'].to("cpu").item()
        # num_truncated tokens can be negative if no overflowing      
        num_truncated_tokens = max(num_truncated_tokens, 0)
        num_seq_left_over = num_truncated_tokens // MAX_CLIP_SEQ_LEN_WITHOUT_BOS_EOS
        last_seq_len = num_truncated_tokens % MAX_CLIP_SEQ_LEN_WITHOUT_BOS_EOS
        overflowing_tokens = batch_encoding["overflowing_tokens"].to("cpu").tolist()[0]

        # Build token list
        tokens_list = list()
        first_seq = batch_encoding["input_ids"]
        tokens_list.append(first_seq.cpu().tolist()[0])
        for i in range(num_seq_left_over):
            s = overflowing_tokens[i*MAX_CLIP_SEQ_LEN_WITHOUT_BOS_EOS:(i+1)*MAX_CLIP_SEQ_LEN_WITHOUT_BOS_EOS]
            seq = [CLIP_BOS] + s + [CLIP_EOS]
            tokens_list.append(seq)

        # If the the number of tokens in overflowing portion is not the multiple of 75,
        # create a sequence padded with EOS   
        if last_seq_len > 0:
            s = overflowing_tokens[num_seq_left_over*MAX_CLIP_SEQ_LEN_WITHOUT_BOS_EOS:]
            if len(s) != last_seq_len:
                raise ValueError(f"Length mismatch: actual remaining len: {len(s)} while expected len: {last_seq_len} ")
            seq = [CLIP_BOS] + s + [CLIP_EOS]  + [CLIP_EOS] * (MAX_CLIP_SEQ_LEN_WITHOUT_BOS_EOS - last_seq_len)
            assert len(seq) == MAX_CLIP_SEQ_LEN
            tokens_list.append(seq)

        # For each element on token list, generate CLIP embedding
        embedding_list = list()
        for s in tokens_list:
            tokens = torch.tensor([s]).to(self.device)
            outputs = self.transformer(input_ids=tokens, output_hidden_states=True)
            
            assert len(outputs.hidden_states) == 13  # 13 for input embedding + 12 transformers
            # z = outputs.last_hidden_state
            transformer_block_index = 13 - clip_skip
            z = outputs.hidden_states[transformer_block_index]
            embedding_list.append(z)

        output = torch.concat(embedding_list, axis=1)
        output = output.repeat(batch_size, 1, 1)
        return output

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
        unprocessed_embeddings = generate_clip_embeddings_from_prompt(
            self.tokenizer, 
            self.transformer, 
            embedding_dir, 
            single_text)
  
        embedding_list = list()
        for e in unprocessed_embeddings:
            outputs=self.transformer(input_embeddings=e, output_hidden_states=True)          
            assert len(outputs.hidden_states) == 13  # 13 for input embedding + 12 transformers
            transformer_block_index = 13 - clip_skip
            z = outputs.hidden_states[transformer_block_index]
            embedding_list.append(z)

        output = torch.concat(embedding_list, axis=1)
        output = output.repeat(batch_size, 1, 1)
        return output

    def encode(self, text, embedding_dir=None, clip_skip=1):
        logger.debug(" FrozenCLIPEmbedder encode")
        return self(text, embedding_dir=embedding_dir, clip_skip=clip_skip)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device=os.environ.get("GPU_DEVICE", "cpu"), max_length=MAX_CLIP_SEQ_LEN, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device=os.environ.get("GPU_DEVICE", "cpu"),
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
