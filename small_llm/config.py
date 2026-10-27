"""Konfigurasi eksperimen yang sama untuk setiap varian."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from small_llm.model import ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data/stories_demo"
    out_dir: str = "artifacts"
    device: str = "auto"
    seed: int = 42
    valid_fraction: float = 0.2
    batch_size: int = 2
    grad_accum: int = 4
    steps: int = 300
    eval_every: int = 50
    eval_batches: int = 8
    log_every: int = 10
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        counts = (self.batch_size, self.grad_accum, self.steps, self.eval_every,
                  self.eval_batches, self.log_every)
        if min(counts) < 1:
            raise ValueError("Batch, langkah, dan interval harus positif")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate dan grad_clip harus positif")
        if self.weight_decay < 0:
            raise ValueError("weight_decay tidak boleh negatif")


@dataclass(frozen=True)
class SparseConfig:
    sparsity: float = 0.5
    active_fraction: float = 0.25
    n_experts: int = 4
    top_k: int = 2
    density: float = 0.5
    rewire_fraction: float = 0.1
    rewire_every: int = 50
    aux_coef: float = 0.01

    def __post_init__(self) -> None:
        if not 0 <= self.sparsity < 1 or not 0 < self.density < 1:
            raise ValueError("sparsity atau density tidak valid")
        if not 0 < self.active_fraction <= 1 or not 0 <= self.rewire_fraction <= 1:
            raise ValueError("active_fraction atau rewire_fraction tidak valid")
        if self.rewire_every < 1 or self.n_experts < 2 or not 1 <= self.top_k <= self.n_experts:
            raise ValueError("Pengaturan SET atau MoE tidak valid")
        if self.aux_coef < 0:
            raise ValueError("aux_coef tidak boleh negatif")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        model=ModelConfig(**payload.get("model", {})),
        train=TrainConfig(**payload.get("train", {})),
        sparse=SparseConfig(**payload.get("sparse", {})),
    )


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS tidak tersedia; pilih cpu atau periksa instalasi PyTorch")
    if requested not in ("cpu", "mps"):
        raise ValueError("device harus auto, cpu, atau mps")
    return requested

