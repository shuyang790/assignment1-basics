from __future__ import annotations

import os
import time
import math
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
import numpy.typing as npt
import torch
from torch import Tensor
from einops import einsum, rearrange


class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.log_theta = math.log(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = rearrange(torch.arange(max_seq_len), "seq_len -> seq_len 1")
        ks = rearrange(torch.arange(0, d_k, 2), "half_d -> 1 half_d")
        angles = positions / torch.exp(ks / d_k * self.log_theta)
        sin = torch.sin(angles)
        if device is not None:
            sin = sin.to(device)
        cos = torch.cos(angles)
        if device is not None:
            cos = cos.to(device)

        assert sin.shape == (max_seq_len, d_k // 2)
        assert cos.shape == (max_seq_len, d_k // 2)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def verify_shape(self, x_shape: tuple, sin_cos_shape: tuple) -> None:
        assert x_shape[-1] == 2
        assert sin_cos_shape[-1] == 1

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Float[Tensor, "... seq_len"],
        verbose: bool = False,
    ) -> Float[Tensor, "... seq_len d_k"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x = rearrange(x, "... seq_len (half_d_k p) -> ... seq_len half_d_k p", p=2)
        pattern = (
            "... seq_len half_d_k -> ... " + " ".join(["1"] * (x.ndim - 3)) + " seq_len half_d_k 1"
        )
        sin = rearrange(self.sin[token_positions], pattern)
        cos = rearrange(self.cos[token_positions], pattern)
        # sin = rearrange(self.sin[token_positions], "seq_len half_d_k -> ... seq_len half_d_k 1")
        # cos = rearrange(self.cos[token_positions], "seq_len half_d_k -> ... seq_len half_d_k 1")
        self.verify_shape(x.shape, sin.shape)
        self.verify_shape(x.shape, cos.shape)
        if verbose:
            print(f"TEST: x {x.shape}")
            print(f"TEST: x slice {x[..., 0:1].shape}")
            print(f"TEST: sin {sin.shape}")
            print(f"TEST: cos {cos.shape}")
        concated = torch.concat(
            [x[..., 0:1] * cos - x[..., 1:2] * sin, x[..., 0:1] * sin + x[..., 1:2] * cos],
            dim=-1,
        )
        if verbose:
            print(f"TEST: concated {concated.shape}")
        result = rearrange(
            concated,
            "... seq_len half_d_k p -> ... seq_len (half_d_k p)",
            p=2,
        )
        return result.to(in_dtype)
