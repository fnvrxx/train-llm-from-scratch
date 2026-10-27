"""Empat cara mengurangi koneksi atau komputasi pada MLP Transformer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from small_llm.model import DenseMLP, StoryTransformer

VARIANTS = ("dense", "pruned", "topk", "moe", "set")


class PrunedMLP(nn.Module):
    def __init__(self, source: DenseMLP, sparsity: float) -> None:
        super().__init__()
        if not 0 <= sparsity < 1:
            raise ValueError("sparsity harus berada di [0, 1)")
        self.hidden, self.proj = source.hidden, source.proj
        self.register_buffer("hidden_mask", self._mask(self.hidden.weight, sparsity))
        self.register_buffer("proj_mask", self._mask(self.proj.weight, sparsity))

    @staticmethod
    def _mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        mask = torch.ones_like(weight, dtype=torch.bool)
        count = int(weight.numel() * sparsity)
        if count:
            weakest = weight.detach().abs().flatten().argsort()[:count]
            mask.view(-1)[weakest] = False
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(F.linear(x, self.hidden.weight * self.hidden_mask, self.hidden.bias))
        return F.linear(hidden, self.proj.weight * self.proj_mask, self.proj.bias)


class TopKMLP(nn.Module):
    def __init__(self, source: DenseMLP, active_fraction: float) -> None:
        super().__init__()
        if not 0 < active_fraction <= 1:
            raise ValueError("active_fraction harus berada di (0, 1]")
        self.hidden, self.proj = source.hidden, source.proj
        self.k = max(1, round(self.hidden.out_features * active_fraction))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = F.relu(self.hidden(x))
        if self.k == activations.shape[-1]:
            return self.proj(activations)
        values, indices = activations.topk(self.k, dim=-1)
        columns = F.embedding(indices, self.proj.weight.T.contiguous())
        return (values.unsqueeze(-1) * columns).sum(dim=-2) + self.proj.bias


class Expert(nn.Module):
    def __init__(self, n_embed: int, width: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_embed, width)
        self.proj = nn.Linear(width, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.hidden(x)))


class MoEMLP(nn.Module):
    def __init__(self, source: DenseMLP, n_experts: int, top_k: int) -> None:
        super().__init__()
        if n_experts < 2 or not 1 <= top_k <= n_experts:
            raise ValueError("MoE memerlukan 2+ expert dan 1 <= top_k <= n_experts")
        width = source.hidden.out_features
        if width % n_experts:
            raise ValueError("Lebar MLP harus habis dibagi jumlah expert")
        n_embed = source.hidden.in_features
        self.gate = nn.Linear(n_embed, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(n_embed, width // n_experts) for _ in range(n_experts))
        self.top_k = top_k
        self.aux_loss: torch.Tensor | None = None
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                start, end = i * (width // n_experts), (i + 1) * (width // n_experts)
                expert.hidden.weight.copy_(source.hidden.weight[start:end])
                expert.hidden.bias.copy_(source.hidden.bias[start:end])
                expert.proj.weight.copy_(source.proj.weight[:, start:end])
                expert.proj.bias.copy_(source.proj.bias / n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        gate_logits = self.gate(flat)
        scores, chosen = gate_logits.topk(self.top_k, dim=-1)
        weights = scores.softmax(dim=-1)
        importance = gate_logits.softmax(dim=-1).mean(dim=0)
        load = F.one_hot(chosen, len(self.experts)).float().mean(dim=(0, 1))
        self.aux_loss = len(self.experts) * (importance * load).sum()
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(chosen == expert_id)
            if token_idx.numel():
                contribution = expert(flat[token_idx]) * (
                    self.top_k * weights[token_idx, slot_idx, None]
                )
                output = output.index_add(0, token_idx, contribution)
        return output.reshape_as(x)


class SETMLP(nn.Module):
    def __init__(self, source: DenseMLP, density: float) -> None:
        super().__init__()
        if not 0 < density < 1:
            raise ValueError("density harus berada di (0, 1)")
        self.hidden, self.proj = source.hidden, source.proj
        for name, weight in (("hidden_mask", self.hidden.weight), ("proj_mask", self.proj.weight)):
            count = max(1, min(weight.numel() - 1, round(weight.numel() * density)))
            mask = torch.zeros(weight.numel(), dtype=torch.bool, device=weight.device)
            mask[torch.randperm(weight.numel(), device=weight.device)[:count]] = True
            self.register_buffer(name, mask.reshape_as(weight))
        self.enforce_masks()

    def masked_layers(self):
        yield self.hidden.weight, self.hidden_mask
        yield self.proj.weight, self.proj_mask

    @torch.no_grad()
    def enforce_masks(self) -> None:
        for weight, mask in self.masked_layers():
            weight.mul_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(F.linear(x, self.hidden.weight * self.hidden_mask, self.hidden.bias))
        return F.linear(hidden, self.proj.weight * self.proj_mask, self.proj.bias)


def convert_model(model: StoryTransformer, variant: str, options: dict) -> StoryTransformer:
    if variant not in VARIANTS:
        raise ValueError(f"Varian tidak dikenal: {variant}")
    for block in model.blocks:
        source = block.mlp
        if variant == "pruned":
            block.mlp = PrunedMLP(source, options["sparsity"])
        elif variant == "topk":
            block.mlp = TopKMLP(source, options["active_fraction"])
        elif variant == "moe":
            block.mlp = MoEMLP(source, options["n_experts"], options["top_k"])
        elif variant == "set":
            block.mlp = SETMLP(source, options["density"])
    return model


def moe_aux_loss(model: StoryTransformer) -> torch.Tensor:
    losses = [block.mlp.aux_loss for block in model.blocks if isinstance(block.mlp, MoEMLP)]
    return torch.stack(losses).sum() if losses else torch.zeros((), device=next(model.parameters()).device)


@torch.no_grad()
def rewire_set(model: StoryTransformer, optimizer: torch.optim.Optimizer, fraction: float) -> int:
    if not 0 <= fraction <= 1:
        raise ValueError("rewire_fraction harus berada di [0, 1]")
    changed = 0
    for block in model.blocks:
        if not isinstance(block.mlp, SETMLP):
            continue
        for weight, mask in block.mlp.masked_layers():
            flat_weight, flat_mask = weight.view(-1), mask.view(-1)
            active = flat_mask.nonzero(as_tuple=True)[0]
            dormant = (~flat_mask).nonzero(as_tuple=True)[0]
            count = min(round(active.numel() * fraction), dormant.numel())
            if count == 0:
                continue
            removed = active[flat_weight[active].abs().topk(count, largest=False).indices]
            grown = dormant[torch.randperm(dormant.numel(), device=dormant.device)[:count]]
            flat_mask[removed], flat_mask[grown] = False, True
            flat_weight[removed], flat_weight[grown] = 0, 0
            for value in optimizer.state.get(weight, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == weight.shape:
                    value.view(-1)[removed] = 0
                    value.view(-1)[grown] = 0
            changed += count
        block.mlp.enforce_masks()
    return changed


def active_connections(model: StoryTransformer) -> tuple[int, int]:
    active = total = 0
    for block in model.blocks:
        mlp = block.mlp
        if isinstance(mlp, (PrunedMLP, SETMLP)):
            for mask in (mlp.hidden_mask, mlp.proj_mask):
                active += int(mask.sum().item())
                total += mask.numel()
    return active, total
