"""Magnitude-based pruning, random regrowth, and optimizer-state cleanup."""

import torch

from experiments.evolutionary.model import evolutionary_mlps


@torch.no_grad()
def _clear_optimizer_state(optimizer: torch.optim.Optimizer, weight: torch.Tensor,
                           positions: torch.Tensor) -> None:
    for value in optimizer.state.get(weight, {}).values():
        if isinstance(value, torch.Tensor) and value.shape == weight.shape:
            value.reshape(-1)[positions] = 0


@torch.no_grad()
def rewire_weight(weight: torch.Tensor, mask: torch.Tensor, optimizer: torch.optim.Optimizer,
                  fraction: float) -> int:
    if not 0 <= fraction <= 1:
        raise ValueError("rewire fraction must be in [0, 1]")
    if fraction == 0:
        return 0
    flat_weight = weight.reshape(-1)
    flat_mask = mask.reshape(-1)
    active = flat_mask.nonzero(as_tuple=True)[0]
    dormant = (~flat_mask).nonzero(as_tuple=True)[0]
    count = min(max(1, round(active.numel() * fraction)), dormant.numel())
    if count == 0:
        return 0

    # Grow from the previously dormant set, so each event really changes topology.
    weakest = flat_weight[active].abs().topk(count, largest=False).indices
    removed = active[weakest]
    grown = dormant[torch.randperm(dormant.numel(), device=dormant.device)[:count]]
    flat_mask[removed] = False
    flat_mask[grown] = True
    flat_weight[removed] = 0
    # New connections begin at zero and acquire values on subsequent gradient steps.
    flat_weight[grown] = 0
    changed = torch.cat((removed, grown))
    _clear_optimizer_state(optimizer, weight, changed)
    return count


@torch.no_grad()
def rewire_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 fraction: float) -> int:
    changed = 0
    for mlp in evolutionary_mlps(model):
        for weight, mask in mlp.masked_layers():
            changed += rewire_weight(weight, mask, optimizer, fraction)
    return changed


@torch.no_grad()
def enforce_model_masks(model: torch.nn.Module) -> None:
    for mlp in evolutionary_mlps(model):
        mlp.enforce_masks()


@torch.no_grad()
def active_connections(model: torch.nn.Module) -> tuple[int, int]:
    active = total = 0
    for mlp in evolutionary_mlps(model):
        for _, mask in mlp.masked_layers():
            active += int(mask.sum().item())
            total += mask.numel()
    return active, total
