from __future__ import annotations

import math
from jaxtyping import Float
import torch
from torch import Tensor
from einops import einsum, rearrange

from .rope import RotaryPositionEmbedding


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    amax = torch.amax(in_features, dim=dim, keepdim=True)
    return torch.exp(in_features - amax) / torch.sum(
        torch.exp(in_features - amax), dim=dim, keepdim=True
    )


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... seq_len d_k"],
    K: Float[Tensor, " ... seq_len d_k"],
    V: Float[Tensor, " ... seq_len d_v"],
    mask: Float[Tensor, " ... seq_len keys"] | None = None,
) -> Float[Tensor, " ... seq_len d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k: int = Q.shape[-1]
    x = einsum(Q, K, "... seq_len1 d_k, ... seq_len2 d_k -> ... seq_len1 seq_len2")
    x = x / math.sqrt(d_k)
    if mask is not None:
        bias = torch.where(mask == 1, torch.tensor(1.0), torch.tensor(float("-inf")))
    else:
        bias = torch.zeros(x.shape)
    out = softmax(x + bias, -1)
    out = einsum(out, V, "... seq_len1 seq_len2, ... seq_len2 d_v -> ... seq_len1 d_v")
    return out


class MultiheadSelfAttention(torch.nn.Module):
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionEmbedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.q_weights = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))
        self.k_weights = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))
        self.v_weights = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))
        self.o_weights = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))

    def forward(
        self,
        in_features: Float[Tensor, " ... seq_len d_in"],
        token_positions: Float[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        in_dtype = in_features.dtype
        in_features = in_features.to(torch.float32)
        q = rearrange(self.q_weights, "(h d_k) d_model -> h d_k d_model", h=self.num_heads)
        k = rearrange(self.k_weights, "(h d_k) d_model -> h d_k d_model", h=self.num_heads)
        v = rearrange(self.v_weights, "(h d_k) d_model -> h d_k d_model", h=self.num_heads)
        qx = einsum(q, in_features, "h d_k d_model, ... seq_len d_model -> ... h seq_len d_k")
        kx = einsum(k, in_features, "h d_k d_model, ... seq_len d_model -> ... h seq_len d_k")
        vx = einsum(v, in_features, "h d_k d_model, ... seq_len d_model -> ... h seq_len d_k")

        if self.rope is not None:
            if token_positions is None:
                seq_len = in_features.shape[-2]
                token_positions = rearrange(
                    torch.arange(seq_len, device=in_features.device), "x -> 1 x"
                )
            token_positions = token_positions.squeeze()
            qx = self.rope(qx, token_positions)
            kx = self.rope(kx, token_positions)

        mask = torch.tril(torch.ones(vx.shape[:-1] + (vx.shape[-2],)))
        attention = scaled_dot_product_attention(Q=qx, K=kx, V=vx, mask=mask)
        attention = rearrange(attention, "... h seq_len d_k -> ... seq_len (h d_k)")
        out = einsum(
            self.o_weights, attention, "d_model hd_k, ... seq_len hd_k -> ... seq_len d_model"
        )
        return out.to(in_dtype)
