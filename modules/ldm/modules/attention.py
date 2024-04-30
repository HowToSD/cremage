import os
import sys
import logging

from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import List, Optional, Any
from ldm.modules.diffusionmodules.util import checkpoint

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
    logger.info("xformers detected. Using xformers for attention.")
except:
    XFORMERS_IS_AVAILABLE = False
    logger.info("xformers not detected. xformers is not used for attention.")

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

# feedforward
class GEGLU_with_lora(nn.Module):
    def __init__(self, dim_in, dim_out,
                 lora_ranks:List[int]=None, lora_weights:List[float]=None):
        super().__init__()
        self.lora_ranks = lora_ranks
        if self.lora_ranks is None:
            self.lora_ranks = []
        self.lora_weights = lora_weights
        if self.lora_weights is None:
            self.lora_weights = [1.0] * len(self.lora_ranks)

        self.proj = nn.Linear(dim_in, dim_out * 2)

        self.proj_lora_downs = torch.nn.ModuleList()
        self.proj_lora_ups = torch.nn.ModuleList()
        self.proj_lora_alphas = torch.nn.ParameterList()

        for rank in self.lora_ranks:
            self.proj_lora_downs.append(zero_init_module(nn.Linear(dim_in, rank, bias=False)))
            self.proj_lora_ups.append(zero_init_module(nn.Linear(rank, dim_out * 2, bias=False)))
            self.proj_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

    def forward(self, x):
        out = self.proj(x)
        for i in range(len(self.lora_ranks)):
            d = self.proj_lora_downs[i](x) 
            d = self.proj_lora_ups[i](d) 
            out += d * self.lora_weights[i] * (self.proj_lora_alphas[i] / self.lora_ranks[i])

        x, gate = out.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForwardOriginal(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.,
                 lora_ranks:List[int]=None, lora_weights:List[float]=None):
        super().__init__()
        self.lora_ranks = lora_ranks
        if self.lora_ranks is None:
            self.lora_ranks = []
        self.lora_weights = lora_weights
        if self.lora_weights is None:
            self.lora_weights = [1.0] * len(self.lora_ranks)
        
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        # Note: GEGLU is set during the inference execution.
        # If GEGLU is not set, LoRA for this Linear layer will be ignored.
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU_with_lora(dim, inner_dim,
                                          lora_ranks=lora_ranks,
                                          lora_weights=lora_weights)

        self.net = nn.ModuleList([
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        ])

        self.net_2_lora_downs = torch.nn.ModuleList()
        self.net_2_lora_ups = torch.nn.ModuleList()
        self.net_2_lora_alphas = torch.nn.ParameterList()

        for rank in self.lora_ranks:
            self.net_2_lora_downs.append(zero_init_module(nn.Linear(inner_dim, rank, bias=False)))
            self.net_2_lora_ups.append(zero_init_module(nn.Linear(rank, dim_out, bias=False)))
            self.net_2_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

    def forward(self, x):
        x = self.net[0](x) # GELU
        x = self.net[1](x)  # Dropout
        x_original = self.net[2](x)  # Linear

        d_sum = 0
        for i in range(len(self.lora_ranks)):
            d = self.net_2_lora_downs[i](x) 
            d = self.net_2_lora_ups[i](d) 
            d_sum += d * self.lora_weights[i] * (self.net_2_lora_alphas[i] / self.lora_ranks[i])
        x = x_original + d_sum
        return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
        # FIXME. This is from the original SD code, but this may be buggy. Change to p.data._zero()?
        # as detach() is supposed to return a new tensor???
    return module

def zero_init_module(module):
    """
    Updated version of zero_module. Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.data.zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 lora_ranks:List[int]=None, lora_weights:List[float]=None,
                 ipa_scale=1.0,
                 ipa_num_tokens=0):
        """
        Adopted from Doggetx's code.

        Args:
            lora_ranks (List[int]): List of LoRA ranks to be used for the size of the LoRA
                matrices.  If Linear is to map from A to B, then two matrices are defined
                AxR and RxB.
                We can have a variable number of LoRA matrice pairs.
        """
        super().__init__()
        self.lora_ranks = lora_ranks
        if self.lora_ranks is None:
            self.lora_ranks = []
        self.lora_weights = lora_weights
        if self.lora_weights is None:
            self.lora_weights = [1.0] * len(self.lora_ranks)

        self.ipa_scale = ipa_scale
        self.ipa_num_tokens = ipa_num_tokens

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        # LoRAs
        self.q_lora_downs = torch.nn.ModuleList()
        self.q_lora_ups = torch.nn.ModuleList()
        self.q_lora_alphas = torch.nn.ParameterList()
        self.k_lora_downs = torch.nn.ModuleList()
        self.k_lora_ups = torch.nn.ModuleList()
        self.k_lora_alphas = torch.nn.ParameterList()
        self.v_lora_downs = torch.nn.ModuleList()
        self.v_lora_ups = torch.nn.ModuleList()
        self.v_lora_alphas = torch.nn.ParameterList()
        self.out_lora_downs = torch.nn.ModuleList()
        self.out_lora_ups = torch.nn.ModuleList()
        self.out_lora_alphas = torch.nn.ParameterList()

        for rank in self.lora_ranks:
            self.q_lora_downs.append(zero_init_module(nn.Linear(query_dim, rank, bias=False)))
            self.q_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.q_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.k_lora_downs.append(zero_init_module(nn.Linear(context_dim, rank, bias=False)))
            self.k_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.k_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.v_lora_downs.append(zero_init_module(nn.Linear(context_dim, rank, bias=False)))
            self.v_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.v_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.out_lora_downs.append(zero_init_module(nn.Linear(inner_dim, rank, bias=False)))
            self.out_lora_ups.append(zero_init_module(nn.Linear(rank, query_dim, bias=False)))
            self.out_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

        # IP-Adapter FaceID support
        if self.ipa_num_tokens > 0:
            self.to_k_ipa = torch.nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ipa = torch.nn.Linear(context_dim, inner_dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        # Add LoRA for q
        for i in range(len(self.lora_ranks)):
            d = self.q_lora_downs[i](x) 
            d = self.q_lora_ups[i](d) 
            q_in += d * self.lora_weights[i] * (self.q_lora_alphas[i] / self.lora_ranks[i])
        if len(self.lora_ranks) > 0:
            del d

        context = default(context, x)

        if self.ipa_num_tokens > 0:
            end_pos = context.shape[1] - self.ipa_num_tokens
            org_context = context[: ,:end_pos, :]
            ipa_context = context[: ,end_pos:, :]
            context = org_context

        k_in = self.to_k(context)
        # Add LoRA for k
        for i in range(len(self.lora_ranks)):
            d = self.k_lora_downs[i](context) 
            d = self.k_lora_ups[i](d) 
            k_in += d * self.lora_weights[i] * (self.k_lora_alphas[i] / self.lora_ranks[i])
        if len(self.lora_ranks) > 0:
            del d

        v_in = self.to_v(context)
        # Add LoRA for v
        for i in range(len(self.lora_ranks)):
            d = self.v_lora_downs[i](context) 
            d = self.v_lora_ups[i](d) 
            v_in += d * self.lora_weights[i] * (self.v_lora_alphas[i] / self.lora_ranks[i])
        if len(self.lora_ranks) > 0:
            del d
        del context, x

        q_save = q_in

        # attention core part start
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        use_optimized_cross_attention = True
        if use_optimized_cross_attention:

            r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

            stats = torch.cuda.memory_stats(q.device)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

            gb = 1024 ** 3
            tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
            modifier = 3 if q.element_size() == 2 else 2.5
            mem_required = tensor_size * modifier
            steps = 1


            if mem_required > mem_free_total:
                steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))
                # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
                #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

            if steps > 64:
                max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
                raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                                f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale

                s2 = s1.softmax(dim=-1, dtype=q.dtype)
                del s1

                r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
                del s2

            del q, k, v

            out = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
            del r1

        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # attention core part end

        # ipa attention start
        if self.ipa_num_tokens > 0:
            ipa_key = self.to_k_ipa(ipa_context)
            ipa_value = self.to_v_ipa(ipa_context)
            k_in = ipa_key
            v_in = ipa_value
            q_in = q_save

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
            del q_in, k_in, v_in

            use_optimized_cross_attention = True
            if use_optimized_cross_attention:

                r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

                stats = torch.cuda.memory_stats(q.device)
                mem_active = stats['active_bytes.all.current']
                mem_reserved = stats['reserved_bytes.all.current']
                mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
                mem_free_torch = mem_reserved - mem_active
                mem_free_total = mem_free_cuda + mem_free_torch

                gb = 1024 ** 3
                tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
                modifier = 3 if q.element_size() == 2 else 2.5
                mem_required = tensor_size * modifier
                steps = 1

                if mem_required > mem_free_total:
                    steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))
                    # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
                    #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

                if steps > 64:
                    max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
                    raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                                    f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

                slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
                for i in range(0, q.shape[1], slice_size):
                    end = i + slice_size
                    s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale

                    s2 = s1.softmax(dim=-1, dtype=q.dtype)
                    del s1

                    r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
                    del s2

                del q, k, v

                out_ipa = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
                del r1

            else:  # Original version of attention
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

                if exists(mask):
                    mask = rearrange(mask, 'b ... -> b (...)')
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = repeat(mask, 'b j -> (b h) () j', h=h)
                    sim.masked_fill_(~mask, max_neg_value)

                # attention, what we cannot get enough of
                attn = sim.softmax(dim=-1)

                out_ipa = einsum('b i j, b j d -> b i d', attn, v)
                out_ipa = rearrange(out_ipa, '(b h) n d -> b n (h d)', h=h)

            out = out + self.ipa_scale * out_ipa
            if torch.cuda.is_available() is False:
                out = out.half()
            del out_ipa
        # ipa attention end

        out_original = self.to_out(out)
        # Add LoRA for out
        d_sum = 0
        for i in range(len(self.lora_ranks)):
            d = self.out_lora_downs[i](out) 
            d = self.out_lora_ups[i](d) 
            d_sum += d * self.lora_weights[i] * (self.out_lora_alphas[i] / self.lora_ranks[i])
        out = out_original + d_sum
        if len(self.lora_ranks) > 0:
            del d
        del out_original, d_sum
        return out

# original
class CrossAttentionOriginal(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 lora_ranks:List[int]=None, lora_weights:List[float]=None,
                 ipa_scale=1.0,
                 ipa_num_tokens=0):
        """
        
        Args:
            lora_ranks (List[int]): List of LoRA ranks to be used for the size of the LoRA
                matrices.  If Linear is to map from A to B, then two matrices are defined
                AxR and RxB.
                We can have a variable number of LoRA matrice pairs.
        """
        super().__init__()
        self.lora_ranks = lora_ranks
        if self.lora_ranks is None:
            self.lora_ranks = []
        self.lora_weights = lora_weights
        if self.lora_weights is None:
            self.lora_weights = [1.0] * len(self.lora_ranks)
        self.ipa_scale = ipa_scale
        self.ipa_num_tokens = ipa_num_tokens

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        # LoRAs
        self.q_lora_downs = torch.nn.ModuleList()
        self.q_lora_ups = torch.nn.ModuleList()
        self.q_lora_alphas = torch.nn.ParameterList()
        self.k_lora_downs = torch.nn.ModuleList()
        self.k_lora_ups = torch.nn.ModuleList()
        self.k_lora_alphas = torch.nn.ParameterList()
        self.v_lora_downs = torch.nn.ModuleList()
        self.v_lora_ups = torch.nn.ModuleList()
        self.v_lora_alphas = torch.nn.ParameterList()
        self.out_lora_downs = torch.nn.ModuleList()
        self.out_lora_ups = torch.nn.ModuleList()
        self.out_lora_alphas = torch.nn.ParameterList()

        for rank in self.lora_ranks:
            self.q_lora_downs.append(zero_init_module(nn.Linear(query_dim, rank, bias=False)))
            self.q_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.q_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.k_lora_downs.append(zero_init_module(nn.Linear(context_dim, rank, bias=False)))
            self.k_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.k_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.v_lora_downs.append(zero_init_module(nn.Linear(context_dim, rank, bias=False)))
            self.v_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.v_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.out_lora_downs.append(zero_init_module(nn.Linear(inner_dim, rank, bias=False)))
            self.out_lora_ups.append(zero_init_module(nn.Linear(rank, query_dim, bias=False)))
            self.out_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

        # IP-Adapter FaceID support
        if self.ipa_num_tokens > 0:
            self.to_k_ipa = torch.nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ipa = torch.nn.Linear(context_dim, inner_dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        # Add LoRA for q
        for i in range(len(self.lora_ranks)):
            d = self.q_lora_downs[i](x) 
            d = self.q_lora_ups[i](d) 
            q += d * self.lora_weights[i] * (self.q_lora_alphas[i] / self.lora_ranks[i])

        context = default(context, x)

        if self.ipa_num_tokens > 0:
            end_pos = context.shape[1] - self.ipa_num_tokens
            org_context = context[: ,:end_pos, :]
            ipa_context = context[: ,end_pos:, :]
            context = org_context

        k = self.to_k(context)
        # Add LoRA for k
        for i in range(len(self.lora_ranks)):
            d = self.k_lora_downs[i](context) 
            d = self.k_lora_ups[i](d) 
            k += d * self.lora_weights[i] * (self.k_lora_alphas[i] / self.lora_ranks[i])

        v = self.to_v(context)
        # Add LoRA for v
        for i in range(len(self.lora_ranks)):
            d = self.v_lora_downs[i](context) 
            d = self.v_lora_ups[i](d) 
            v += d * self.lora_weights[i] * (self.v_lora_alphas[i] / self.lora_ranks[i])

        q_save = q
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if self.ipa_num_tokens > 0:
            ipa_key = self.to_k_ipa(ipa_context)
            ipa_value = self.to_v_ipa(ipa_context)
            k = ipa_key
            v = ipa_value
            q = q_save

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out_ipa = einsum('b i j, b j d -> b i d', attn, v)
            out_ipa = rearrange(out_ipa, '(b h) n d -> b n (h d)', h=h)
            out = out + self.ipa_scale * out_ipa
            if torch.cuda.is_available() is False:
                out = out.half()

        out_original = self.to_out(out)
        # Add LoRA for out
        d_sum = 0
        for i in range(len(self.lora_ranks)):
            d = self.out_lora_downs[i](out) 
            d = self.out_lora_ups[i](d) 
            d_sum += d * self.lora_weights[i] * (self.out_lora_alphas[i] / self.lora_ranks[i])
        out = out_original + d_sum
        return out


class MemoryEfficientCrossAttention(nn.Module):
    """
    Attention using xformers.
    This is method is used by default when xformers package is found.    

    https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0,
                 lora_ranks:List[int]=None, lora_weights:List[float]=None,
                 ipa_scale=1.0,
                 ipa_num_tokens=0):
        super().__init__()
        self.lora_ranks = lora_ranks
        if self.lora_ranks is None:
            self.lora_ranks = []
        self.lora_weights = lora_weights
        if self.lora_weights is None:
            self.lora_weights = [1.0] * len(self.lora_ranks)        

        self.ipa_scale = ipa_scale
        self.ipa_num_tokens = ipa_num_tokens

        # print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
        #      f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

        # LoRAs
        self.q_lora_downs = torch.nn.ModuleList()
        self.q_lora_ups = torch.nn.ModuleList()
        self.q_lora_alphas = torch.nn.ParameterList()
        self.k_lora_downs = torch.nn.ModuleList()
        self.k_lora_ups = torch.nn.ModuleList()
        self.k_lora_alphas = torch.nn.ParameterList()
        self.v_lora_downs = torch.nn.ModuleList()
        self.v_lora_ups = torch.nn.ModuleList()
        self.v_lora_alphas = torch.nn.ParameterList()
        self.out_lora_downs = torch.nn.ModuleList()
        self.out_lora_ups = torch.nn.ModuleList()
        self.out_lora_alphas = torch.nn.ParameterList()

        for rank in self.lora_ranks:
            self.q_lora_downs.append(zero_init_module(nn.Linear(query_dim, rank, bias=False)))
            self.q_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.q_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.k_lora_downs.append(zero_init_module(nn.Linear(context_dim, rank, bias=False)))
            self.k_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.k_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.v_lora_downs.append(zero_init_module(nn.Linear(context_dim, rank, bias=False)))
            self.v_lora_ups.append(zero_init_module(nn.Linear(rank, inner_dim, bias=False)))
            self.v_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

            self.out_lora_downs.append(zero_init_module(nn.Linear(inner_dim, rank, bias=False)))
            self.out_lora_ups.append(zero_init_module(nn.Linear(rank, query_dim, bias=False)))
            self.out_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

        # IP-Adapter FaceID support
        if self.ipa_num_tokens > 0:
            self.to_k_ipa = torch.nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ipa = torch.nn.Linear(context_dim, inner_dim, bias=False)

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        # Add LoRA for q
        for i in range(len(self.lora_ranks)):
            d = self.q_lora_downs[i](x) 
            d = self.q_lora_ups[i](d) 
            q += d * self.lora_weights[i] * (self.q_lora_alphas[i] / self.lora_ranks[i])

        context = default(context, x)

        if self.ipa_num_tokens > 0:
            end_pos = context.shape[1] - self.ipa_num_tokens
            org_context = context[: ,:end_pos, :]
            ipa_context = context[: ,end_pos:, :]
            context = org_context

        k = self.to_k(context)
        # Add LoRA for k
        for i in range(len(self.lora_ranks)):
            d = self.k_lora_downs[i](context) 
            d = self.k_lora_ups[i](d) 
            k += d * self.lora_weights[i] * (self.k_lora_alphas[i] / self.lora_ranks[i])

        v = self.to_v(context)
        # Add LoRA for v
        for i in range(len(self.lora_ranks)):
            d = self.v_lora_downs[i](context) 
            d = self.v_lora_ups[i](d) 
            v += d * self.lora_weights[i] * (self.v_lora_alphas[i] / self.lora_ranks[i])

        b, _, _ = q.shape
        q_save = q
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )

        if self.ipa_num_tokens > 0:
            ipa_key = self.to_k_ipa(ipa_context)
            ipa_value = self.to_v_ipa(ipa_context)
            k = ipa_key
            v = ipa_value
            q = q_save
            q, k, v = map(
                    lambda t: t.unsqueeze(3)
                    .reshape(b, t.shape[1], self.heads, self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b * self.heads, t.shape[1], self.dim_head)
                    .contiguous(),
                    (q, k, v),
                )

            # actually compute the attention, what we cannot get enough of
            out_ipa = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

            if exists(mask):
                raise NotImplementedError
            out_ipa = (
                out_ipa.unsqueeze(0)
                .reshape(b, self.heads, out_ipa.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out_ipa.shape[1], self.heads * self.dim_head)
            )

            out = out + self.ipa_scale * out_ipa
            if torch.cuda.is_available() is False:
                out = out.half()

        out_original = self.to_out(out)
        # Add LoRA for out
        d_sum = 0
        for i in range(len(self.lora_ranks)):
            d = self.out_lora_downs[i](out) 
            d = self.out_lora_ups[i](d) 
            d_sum += d * self.lora_weights[i] * (self.out_lora_alphas[i] / self.lora_ranks[i])
        out = out_original + d_sum
        return out


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention + doggetx's optimization
        "softmax-xformers": MemoryEfficientCrossAttention,
        "softmax-original": CrossAttentionOriginal  # original (used for Mac & non-CUDA)
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
                 gated_ff=True, checkpoint=True,
                 disable_self_attn=False, # CONTROLNET change
                 lora_ranks:List[int]=None, lora_weights:List[float]=None,
                 ipa_scale=1.0,
                 ipa_num_tokens=0):
        super().__init__()
        attn_mode = "softmax-original"
        if XFORMERS_IS_AVAILABLE:
            attn_mode = "softmax-xformers"
        elif os.environ.get("GPU_DEVICE", "cpu").startswith("cuda"):
            attn_mode = "softmax"  # doggetx which relies on cuda
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn        
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None,
                                    lora_ranks=lora_ranks,
                                    lora_weights=lora_weights)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff,
                              lora_ranks=lora_ranks,
                              lora_weights=lora_weights)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head,
                                    dropout=dropout,
                                    lora_ranks=lora_ranks,
                                    lora_weights=lora_weights,   # is self-attn if context is none
                                    ipa_scale=ipa_scale,
                                    ipa_num_tokens=ipa_num_tokens)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,
                 lora_ranks:List[int]=None, lora_weights:List[float]=None,
                 ipa_scale=1.0,
                 ipa_num_tokens=0):
        """
        Cremage note:
        use_linear was added in ControlNet version of Unet model, but it is not
        used in this code as that would change the shape of the weight for LoRA
        for proj in which makes it impossible to load the weight for the layer from
        a LoRA model.

        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        """
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.lora_ranks = lora_ranks
        if self.lora_ranks is None:
            self.lora_ranks = []
        self.lora_weights = lora_weights
        if self.lora_weights is None:
            self.lora_weights = [1.0] * len(self.lora_ranks)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.proj_in_lora_downs = torch.nn.ModuleList()
        self.proj_in_lora_ups = torch.nn.ModuleList()
        self.proj_in_lora_alphas = torch.nn.ParameterList()
                
        for rank in self.lora_ranks:
            self.proj_in_lora_downs.append(
                zero_init_module(nn.Conv2d(
                    in_channels,
                    rank,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)))
            self.proj_in_lora_ups.append(
                zero_init_module(nn.Conv2d(
                    rank,
                    inner_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)))
            self.proj_in_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head,
                                   dropout=dropout,
                                   context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn,
                                   checkpoint=use_checkpoint,
                                   lora_ranks=self.lora_ranks,
                                   lora_weights=self.lora_weights,
                                   ipa_scale=ipa_scale,
                                   ipa_num_tokens=ipa_num_tokens)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

        self.proj_out_lora_downs = torch.nn.ModuleList()
        self.proj_out_lora_ups = torch.nn.ModuleList()
        self.proj_out_lora_alphas = torch.nn.ParameterList()
                
        for rank in self.lora_ranks:
            self.proj_out_lora_downs.append(
                zero_init_module(nn.Conv2d(
                    inner_dim,
                    rank,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)))
            self.proj_out_lora_ups.append(
                zero_init_module(nn.Conv2d(
                    rank,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False)))
            self.proj_out_lora_alphas.append(torch.nn.Parameter(torch.tensor(float(rank))))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x_original = self.proj_in(x)

        d_sum = 0
        for i in range(len(self.lora_ranks)):
            d = self.proj_in_lora_downs[i](x)
            d = self.proj_in_lora_ups[i](d)
            d_sum += d * self.lora_weights[i] * (self.proj_in_lora_alphas[i] / self.lora_ranks[i])
        x = x_original + d_sum

        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x_original = self.proj_out(x)

        d_sum = 0
        for i in range(len(self.lora_ranks)):
            d = self.proj_out_lora_downs[i](x)
            d = self.proj_out_lora_ups[i](d)
            d_sum += d * self.lora_weights[i]* (self.proj_out_lora_alphas[i] / self.lora_ranks[i])
        x = x_original + d_sum

        return x + x_in