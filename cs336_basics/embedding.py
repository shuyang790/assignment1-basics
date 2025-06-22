from __future__ import annotations

import torch
from torch import Tensor


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Initialize embedding weights
        tensor = torch.randn(num_embeddings, embedding_dim)
        torch.nn.init.trunc_normal_(tensor, mean=0, std=1, a=-3, b=3)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        self.weights = torch.nn.Parameter(tensor)

    def forward(self, token_ids: torch.Tensor) -> Tensor:
        return self.weights[token_ids]
