from __future__ import annotations

import math
import torch
from torch import Tensor
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights
        std = math.sqrt(2 / (in_features + out_features))
        tensor = torch.normal(mean=0, std=std, size=(out_features, in_features))
        torch.nn.init.trunc_normal_(tensor, mean=0, std=std, a=-3 * std, b=3 * std)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        self.weights = torch.nn.Parameter(tensor)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weights, "... in, out in -> ... out")
