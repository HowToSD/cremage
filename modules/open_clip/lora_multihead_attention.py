"""
MultiheadAttention to support LoRA.
Added by Cremage.
"""
# from .utils import zero_init_module
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def multi_head_attention_forward(
    query: F.Tensor,
    key: F.Tensor,
    value: F.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[F.Tensor],
    in_proj_bias: Optional[F.Tensor],
    bias_k: Optional[F.Tensor],
    bias_v: Optional[F.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: F.Tensor,
    out_proj_bias: Optional[F.Tensor],
    training: bool = True,
    key_padding_mask: Optional[F.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[F.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[F.Tensor] = None,
    k_proj_weight: Optional[F.Tensor] = None,
    v_proj_weight: Optional[F.Tensor] = None,
    static_k: Optional[F.Tensor] = None,
    static_v: Optional[F.Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    lora_ranks=None,  # Cremage added
    lora_weights=None,  # Cremage added
    out_lora_downs = None,  # Cremage added
    out_lora_ups = None,  # Cremage added
    out_lora_alphas = None  # Cremage added
) -> Tuple[F.Tensor, Optional[F.Tensor]]:
    r"""Forward method for MultiHeadAttention.

    See :class:`torch.nn.MultiheadAttention` for details.

    Cremage note: This is adopted from:
    https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py.
    See the 3rd party license file at the root of the project.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if F.has_torch_function(tens_ops):
        return F.handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights
        )

    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q * F.math.sqrt(1.0 / float(E))

        assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)

        # Cremage LoRA start
        # Original code
        # attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

        attn_output_original = F.linear(attn_output, out_proj_weight, out_proj_bias)

        d_sum = 0
        for i in range(len(lora_ranks)):
            d = out_lora_downs[i](attn_output) 
            d = out_lora_ups[i](d) 
            d_sum += d * lora_weights[i] * (out_lora_alphas[i] / lora_ranks[i])
        attn_output = attn_output_original + d_sum
        # LoRA end

        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


def zero_init_module(module):
    """
    Updated version of zero_module. Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.data.zero_()
    return module

class LoRAMultiheadAttention(nn.Module):
    """
    Note that if the weight name was : foo.attn.bar, it will be changed to:
    foo.attn.multihead_attn.bar so code that deals with load_state_dict needs
    to be updated.
    """
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 dropout=0.0, 
                 bias=True, 
                 add_bias_kv=False, 
                 add_zero_attn=False, 
                 kdim=None, 
                 vdim=None, 
                 lora_ranks: List[int] = None, 
                 lora_weights: List[float] = None):
        super(LoRAMultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout, 
            bias, 
            add_bias_kv, 
            add_zero_attn, 
            kdim, 
            vdim)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads  # Should be integer division
        self.lora_ranks = lora_ranks if lora_ranks is not None else []
        self.lora_weights = lora_weights if lora_weights is not None else [1.0] * len(self.lora_ranks)

        self.q_lora_downs = nn.ModuleList()
        self.q_lora_ups = nn.ModuleList()
        self.q_lora_alphas = nn.ParameterList()

        self.k_lora_downs = nn.ModuleList()
        self.k_lora_ups = nn.ModuleList()
        self.k_lora_alphas = nn.ParameterList()

        self.v_lora_downs = nn.ModuleList()
        self.v_lora_ups = nn.ModuleList()
        self.v_lora_alphas = nn.ParameterList()

        self.out_lora_downs = nn.ModuleList()
        self.out_lora_ups = nn.ModuleList()
        self.out_lora_alphas = nn.ParameterList()

        for rank in self.lora_ranks:
            self.q_lora_downs.append(zero_init_module(nn.Linear(embed_dim, rank, bias=False)))
            self.q_lora_ups.append(zero_init_module(nn.Linear(rank, embed_dim, bias=False)))
            self.q_lora_alphas.append(nn.Parameter(torch.tensor(float(rank))))

            self.k_lora_downs.append(zero_init_module(nn.Linear(embed_dim, rank, bias=False)))
            self.k_lora_ups.append(zero_init_module(nn.Linear(rank, embed_dim, bias=False)))
            self.k_lora_alphas.append(nn.Parameter(torch.tensor(float(rank))))

            self.v_lora_downs.append(zero_init_module(nn.Linear(embed_dim, rank, bias=False)))
            self.v_lora_ups.append(zero_init_module(nn.Linear(rank, embed_dim, bias=False)))
            self.v_lora_alphas.append(nn.Parameter(torch.tensor(float(rank))))

            self.out_lora_downs.append(zero_init_module(nn.Linear(embed_dim, rank, bias=False)))
            self.out_lora_ups.append(zero_init_module(nn.Linear(rank, embed_dim, bias=False)))
            self.out_lora_alphas.append(nn.Parameter(torch.tensor(float(rank))))

    def forward(self, 
                query, 
                key, 
                value, 
                key_padding_mask=None, 
                need_weights=True, 
                attn_mask=None):
        # Ensure the inputs have the correct dimensions
        assert query.size(-1) == self.embed_dim
        assert key.size(-1) == self.embed_dim
        assert value.size(-1) == self.embed_dim

        # Directly call the internal multi_head_attention_forward method
        Q, K, V = query, key, value

        # Apply LoRA to Q, K, and V
        for i in range(len(self.lora_ranks)):
            Q = Q + self.q_lora_ups[i](self.q_lora_downs[i](query)) * self.lora_weights[i] * (self.q_lora_alphas[i] / self.lora_ranks[i])
            K = K + self.k_lora_ups[i](self.k_lora_downs[i](key)) * self.lora_weights[i] * (self.k_lora_alphas[i] / self.lora_ranks[i])
            V = V + self.v_lora_ups[i](self.v_lora_downs[i](value)) * self.lora_weights[i] * (self.v_lora_alphas[i] / self.lora_ranks[i])

        # Compute attention
        # attn_output, attn_output_weights = F.multi_head_attention_forward(
        attn_output, attn_output_weights = multi_head_attention_forward(
            Q, K, V, self.embed_dim, self.num_heads,
            self.multihead_attn.in_proj_weight, self.multihead_attn.in_proj_bias,
            self.multihead_attn.bias_k, self.multihead_attn.bias_v,
            self.multihead_attn.add_zero_attn, self.multihead_attn.dropout,
            self.multihead_attn.out_proj.weight, self.multihead_attn.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=False,
            lora_ranks=self.lora_ranks,
            lora_weights=self.lora_weights,
            out_lora_downs = self.out_lora_downs,
            out_lora_ups = self.out_lora_ups,
            out_lora_alphas = self.out_lora_alphas
        )
        
        return attn_output, attn_output_weights


if __name__ == "__main__":
    batch_size = 8
    seq_len = 77
    embed_dim = 2048
    num_heads = 8

    x = torch.randn(seq_len, batch_size, embed_dim)
    multihead_attn = LoRAMultiheadAttention(embed_dim, num_heads)
    attn_output, attn_output_weights = multihead_attn(x, x, x)

    print(attn_output.shape)  # Should be torch.Size([77, 8, 2048])
    print(attn_output_weights.shape)  # Should be torch.Size([8, 77, 77])