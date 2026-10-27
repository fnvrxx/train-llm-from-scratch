"""Transformer decoder kecil yang memperlihatkan tiap operasi utama."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from small_llm.tokenizer import END_OF_STORY, VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    context_length: int = 256
    n_embed: int = 192
    n_head: int = 4
    n_blocks: int = 4
    vocab_size: int = VOCAB_SIZE

    def __post_init__(self) -> None:
        if min(self.context_length, self.n_embed, self.n_head, self.n_blocks) < 1:
            raise ValueError("Dimensi model harus positif")
        if self.n_embed % self.n_head:
            raise ValueError("n_embed harus habis dibagi n_head")
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"Tokenizer byte memerlukan vocab_size={VOCAB_SIZE}")


class DenseMLP(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_embed, 4 * n_embed)
        self.proj = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.hidden(x)))


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embed // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embed, 3 * cfg.n_embed, bias=False)
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q = q.reshape(batch, length, self.n_head, self.head_size).transpose(1, 2)
        k = k.reshape(batch, length, self.n_head, self.head_size).transpose(1, 2)
        v = v.reshape(batch, length, self.n_head, self.head_size).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        joined = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.proj(joined)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_attention = nn.LayerNorm(cfg.n_embed)
        self.attention = CausalAttention(cfg)
        self.ln_mlp = nn.LayerNorm(cfg.n_embed)
        self.mlp: nn.Module = DenseMLP(cfg.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_attention(x))
        return x + self.mlp(self.ln_mlp(x))


class StoryTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.n_embed)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.final_norm = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length = tokens.shape
        if length > self.cfg.context_length:
            raise ValueError("Urutan melebihi context_length")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, tokens: torch.Tensor, max_new_tokens: int, *, temperature: float = 1.0
    ) -> torch.Tensor:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature harus positif dan finite")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens tidak boleh negatif")
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("Prompt harus berisi sedikitnya satu token")
        self.eval()
        batch_size = tokens.shape[0]
        finished = [False] * batch_size
        pending = [0] * batch_size
        lower = [0x80] * batch_size
        upper = [0xBF] * batch_size
        for step in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.cfg.context_length :])
            allowed = torch.zeros((batch_size, self.cfg.vocab_size), dtype=torch.bool,
                                  device=tokens.device)
            remaining = max_new_tokens - step
            for row in range(batch_size):
                if finished[row]:
                    allowed[row, END_OF_STORY] = True
                elif pending[row]:
                    allowed[row, lower[row]:upper[row] + 1] = True
                else:
                    allowed[row, 9:11] = True
                    allowed[row, 32:127] = True
                    allowed[row, END_OF_STORY] = True
                    if remaining >= 2:
                        allowed[row, 0xC2:0xE0] = True
                    if remaining >= 3:
                        allowed[row, 0xE0:0xF0] = True
                    if remaining >= 4:
                        allowed[row, 0xF0:0xF5] = True
            probs = F.softmax(logits[:, -1, :].masked_fill(~allowed, float("-inf"))
                              / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat((tokens, next_token), dim=1)
            for row, token in enumerate(next_token.squeeze(1).tolist()):
                if token == END_OF_STORY:
                    finished[row] = True
                elif pending[row]:
                    pending[row] -= 1
                    lower[row], upper[row] = 0x80, 0xBF
                elif token >= 0xC2:
                    if token < 0xE0:
                        pending[row] = 1
                    elif token < 0xF0:
                        pending[row] = 2
                    else:
                        pending[row] = 3
                    lower[row], upper[row] = {
                        0xC2: (0xA0, 0xBF),
                        0xE0: (0xA0, 0xBF),
                        0xED: (0x80, 0x9F),
                        0xF0: (0x90, 0xBF),
                        0xF4: (0x80, 0x8F),
                    }.get(token, (0x80, 0xBF))
            if all(finished):
                break
        return tokens
