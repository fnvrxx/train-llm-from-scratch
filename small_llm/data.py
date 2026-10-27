"""Baca cerita dan pisahkan train/validasi pada tingkat berkas."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from small_llm.tokenizer import encode


@dataclass(frozen=True)
class StoryCorpus:
    train: torch.Tensor
    valid: torch.Tensor
    train_files: tuple[str, ...]
    valid_files: tuple[str, ...]


def load_stories(folder: str | Path, *, seed: int, valid_fraction: float = 0.2) -> StoryCorpus:
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction harus berada di (0, 1)")
    root = Path(folder)
    files = sorted(root.glob("*.txt"))
    if len(files) < 3:
        raise ValueError(f"Perlu sedikitnya tiga berkas .txt di {root}")
    random.Random(seed).shuffle(files)
    n_valid = max(1, min(len(files) - 1, round(len(files) * valid_fraction)))
    valid_files = files[:n_valid]
    train_files = files[n_valid:]

    def combine(selected: list[Path]) -> torch.Tensor:
        ids: list[int] = []
        for path in selected:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError(f"Cerita kosong: {path}")
            ids.extend(encode(text, end_of_story=True))
        return torch.tensor(ids, dtype=torch.long)

    return StoryCorpus(
        train=combine(train_files),
        valid=combine(valid_files),
        train_files=tuple(path.name for path in train_files),
        valid_files=tuple(path.name for path in valid_files),
    )


def sample_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= context_length:
        raise ValueError(
            f"Korpus berisi {tokens.numel()} token; perlu lebih dari context_length={context_length}"
        )
    starts = torch.randint(tokens.numel() - context_length, (batch_size,), generator=generator)
    x = torch.stack([tokens[i : i + context_length] for i in starts])
    y = torch.stack([tokens[i + 1 : i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)

