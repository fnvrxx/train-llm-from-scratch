import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunedMLP(nn.Module):
    """Dense MLP with persistent magnitude-pruning masks."""

    def __init__(self, dense_mlp: nn.Module, sparsity: float) -> None:
        super().__init__()
        if not 0 <= sparsity < 1:
            raise ValueError("sparsity must be in [0, 1)")
        self.hidden = dense_mlp.hidden
        self.proj = dense_mlp.proj
        self.register_buffer("hidden_mask", self._mask(self.hidden.weight, sparsity))
        self.register_buffer("proj_mask", self._mask(self.proj.weight, sparsity))

    @staticmethod
    def _mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        count = int(weight.numel() * sparsity)
        mask = torch.ones_like(weight)
        if count:
            indices = weight.detach().abs().flatten().argsort()[:count]
            mask.view(-1)[indices] = 0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(F.linear(x, self.hidden.weight * self.hidden_mask, self.hidden.bias))
        return F.linear(hidden, self.proj.weight * self.proj_mask, self.proj.bias)


def convert_model(model: nn.Module, sparsity: float) -> nn.Module:
    for block in model.attn_blocks:
        block.mlp = PrunedMLP(block.mlp, sparsity)
    return model
