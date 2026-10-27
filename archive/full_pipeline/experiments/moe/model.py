import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, n_embed: int, width: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_embed, width)
        self.proj = nn.Linear(width, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.hidden(x)))


class MoEMLP(nn.Module):
    """Route each token to top-k experts with normalized gate weights."""

    def __init__(self, dense_mlp: nn.Module, num_experts: int, top_k: int) -> None:
        super().__init__()
        if num_experts < 2 or not 1 <= top_k <= num_experts:
            raise ValueError("require num_experts >= 2 and 1 <= top_k <= num_experts")
        width = dense_mlp.hidden.out_features
        if width % num_experts:
            raise ValueError("dense hidden width must be divisible by num_experts")
        n_embed = dense_mlp.hidden.in_features
        expert_width = width // num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(n_embed, num_experts, bias=False)
        self.experts = nn.ModuleList(Expert(n_embed, expert_width) for _ in range(num_experts))
        self.aux_loss: torch.Tensor | None = None

        # Partition the dense FFN into expert slices as a warm start. Router weights
        # are new; selecting fewer than all experts changes the original function.
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                start, end = i * expert_width, (i + 1) * expert_width
                expert.hidden.weight.copy_(dense_mlp.hidden.weight[start:end])
                expert.hidden.bias.copy_(dense_mlp.hidden.bias[start:end])
                expert.proj.weight.copy_(dense_mlp.proj.weight[:, start:end])
                expert.proj.bias.copy_(dense_mlp.proj.bias / num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        gate_logits = self.gate(flat)
        scores, chosen = gate_logits.topk(self.top_k, dim=-1)
        weights = scores.softmax(dim=-1)
        importance = gate_logits.softmax(dim=-1).mean(dim=0)
        load = F.one_hot(chosen, self.num_experts).float().mean(dim=(0, 1))
        self.aux_loss = self.num_experts * (importance * load).sum()
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(chosen == expert_id)
            if token_idx.numel():
                contribution = expert(flat[token_idx]) * (self.top_k * weights[token_idx, slot_idx, None])
                output = output.index_add(0, token_idx, contribution)
        return output.reshape_as(x)


def convert_model(model: nn.Module, num_experts: int, top_k: int) -> nn.Module:
    for block in model.attn_blocks:
        block.mlp = MoEMLP(block.mlp, num_experts, top_k)
    return model
