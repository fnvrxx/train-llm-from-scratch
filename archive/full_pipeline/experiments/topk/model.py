import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKMLP(nn.Module):
    """Compute all hidden scores, then project only the k strongest per token."""

    def __init__(self, dense_mlp: nn.Module, active_fraction: float) -> None:
        super().__init__()
        if not 0 < active_fraction <= 1:
            raise ValueError("active_fraction must be in (0, 1]")
        self.hidden = dense_mlp.hidden
        self.proj = dense_mlp.proj
        self.k = max(1, round(self.hidden.out_features * active_fraction))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(x))
        if self.k == hidden.shape[-1]:
            return self.proj(hidden)
        values, indices = hidden.topk(self.k, dim=-1)
        columns = F.embedding(indices, self.proj.weight.T.contiguous())
        return (values.unsqueeze(-1) * columns).sum(dim=-2) + self.proj.bias


def convert_model(model: nn.Module, active_fraction: float) -> nn.Module:
    for block in model.attn_blocks:
        block.mlp = TopKMLP(block.mlp, active_fraction)
    return model
