from __future__ import annotations

from collections.abc import Iterable
from jaxtyping import Float, Int
import torch
from torch import Tensor
from einops import einsum, rearrange

from .attention import MultiheadSelfAttention, softmax
from .rope import RotaryPositionEmbedding
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .embedding import Embedding
from .linear import Linear


class TransformerBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model=d_model)
        if theta is not None:
            self.rope = RotaryPositionEmbedding(
                theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len
            )
        else:
            self.rope = None
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=self.rope,
        )
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"]
    ) -> Float[Tensor, "... seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        out: Float[Tensor, "... seq_len d_model"] = x + self.attn(self.ln1(x))
        out: Float[Tensor, "... seq_len d_model"] = out + self.ffn(self.ln2(out))
        return out.to(in_dtype)


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = torch.nn.Sequential(
            *[
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: Float[Tensor, "... seq_len vocab_size"]):
        out = self.token_embeddings(x)
        out = self.layers(out)
        out = self.ln_final(out)
        out = self.lm_head(out)
        return out
