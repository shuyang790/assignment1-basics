from __future__ import annotations

from jaxtyping import Float
import torch
from torch import Tensor
from einops import einsum


def apply_device_and_dtype(
    tensor: Tensor, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> Tensor:
    """
    Move a tensor to the specified device and dtype.
    """
    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(d_model * 8 / 3) // 64 * 64
        else:
            self.d_ff = d_ff
        # Initialize weights
        self.w1_weights = torch.nn.Parameter(
            apply_device_and_dtype(
                torch.randn(self.d_ff, self.d_model), device=device, dtype=dtype
            )
        )
        self.w2_weights = torch.nn.Parameter(
            apply_device_and_dtype(
                torch.randn(self.d_model, self.d_ff), device=device, dtype=dtype
            )
        )
        self.w3_weights = torch.nn.Parameter(
            apply_device_and_dtype(
                torch.randn(self.d_ff, self.d_model), device=device, dtype=dtype
            )
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        w1_out = einsum(x, self.w1_weights, "... d_model, d_ff d_model -> ... d_ff")
        silu = w1_out * torch.sigmoid(w1_out)
        w3_out = einsum(x, self.w3_weights, "... d_model, d_ff d_model -> ... d_ff")
        gate_out = einsum(silu, w3_out, "... d_ff, ... d_ff -> ... d_ff")
        result = einsum(
            self.w2_weights, gate_out, "d_model d_ff, ... d_ff -> ... d_model"
        )
        return result.to(in_dtype)
