"""
Implemenentation of MultiHead Attention using xformers.
Note that this only works on NVIDIA GPU.  If you want to run this code,
update the xformer memory attention line with other attentions.

Test code path: test/unblur_face/mha_test.py

Copyright 2024 Hideyuki Inada.  All rights reserved.
"""
import torch
import torch.nn as nn
from einops import rearrange
from xformers.ops import memory_efficient_attention

class MultiHeadSelfAttention(nn.Module):
    """
    Multihead attention implementation using xformers.

    Args:
        embed_size (int): The size of the embedding vector.
        heads (int): The number of attention heads.

    Attributes:
        embed_size (int): The size of the embedding vector.
        heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        values (nn.Linear): Linear layer to project input to value vectors.
        keys (nn.Linear): Linear layer to project input to key vectors.
        queries (nn.Linear): Linear layer to project input to query vectors.
        out (nn.Linear): Linear layer to project concatenated heads to output.

    Methods:
        forward(q, k, v):
            Args:
                q (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_size).
                k (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_size).
                v (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_size).

            Returns:
                torch.Tensor: Output tensor after applying multihead attention.
    """
    def __init__(self, embed_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.out = nn.Linear(embed_size, embed_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multihead self-attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_size).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_size).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_size).

        Returns:
            torch.Tensor: Output tensor after applying multihead attention.
        """
        q = self.queries(q)
        k = self.keys(k)
        v = self.values(v)

        # Split embedding into heads
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.heads)

        # Apply xformers' memory efficient attention
        # TODO: Support Non-CUDA devices
        out = memory_efficient_attention(q, k, v)
        
        # Combine heads back to embedding dimension
        out = rearrange(out, 'b s h d -> b s (h d)')
        
        out = self.out(out)
        return out
