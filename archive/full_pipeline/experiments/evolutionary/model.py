"""Masked MLP weights whose active connections can change during training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer import Transformer


class EvolutionaryMLP(nn.Module):
    def __init__(self, dense_mlp: nn.Module, density: float) -> None:
        super().__init__()
        if not 0 < density < 1:
            raise ValueError("density must be between 0 and 1")
        self.hidden = dense_mlp.hidden
        self.proj = dense_mlp.proj
        self.register_buffer("hidden_mask", self._random_mask(self.hidden.weight, density))
        self.register_buffer("proj_mask", self._random_mask(self.proj.weight, density))
        self.enforce_masks()

    @staticmethod
    def _random_mask(weight: torch.Tensor, density: float) -> torch.Tensor:
        count = max(1, min(weight.numel() - 1, round(weight.numel() * density)))
        mask = torch.zeros(weight.numel(), dtype=torch.bool, device=weight.device)
        mask[torch.randperm(weight.numel(), device=weight.device)[:count]] = True
        return mask.reshape_as(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(F.linear(x, self.hidden.weight * self.hidden_mask, self.hidden.bias))
        return F.linear(hidden, self.proj.weight * self.proj_mask, self.proj.bias)

    @torch.no_grad()
    def enforce_masks(self) -> None:
        self.hidden.weight.mul_(self.hidden_mask)
        self.proj.weight.mul_(self.proj_mask)

    def masked_layers(self):
        yield self.hidden.weight, self.hidden_mask
        yield self.proj.weight, self.proj_mask


def convert_model(model: nn.Module, density: float) -> nn.Module:
    for block in model.attn_blocks:
        block.mlp = EvolutionaryMLP(block.mlp, density)
    return model


def evolutionary_mlps(model: nn.Module):
    for block in model.attn_blocks:
        yield block.mlp


def load_experiment_checkpoint(path: str, device: str = "cpu") -> Transformer:
    """Rebuild this variant, including its saved masks, for generation or evaluation."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("experiment") != "evolutionary":
        raise ValueError("not an evolutionary experiment checkpoint")
    dims = checkpoint["model_config"]
    model = Transformer(dims["n_head"], dims["n_embed"], dims["context_length"],
                        dims["vocab_size"], dims["n_blocks"])
    model = convert_model(model, checkpoint["density"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
