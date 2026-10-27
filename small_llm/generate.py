"""Muat checkpoint dan lanjutkan awal cerita yang diberikan pengguna."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from small_llm.config import resolve_device
from small_llm.model import ModelConfig, StoryTransformer
from small_llm.sparse import convert_model
from small_llm.tokenizer import decode, encode


def load_model(path: str | Path, device: str) -> StoryTransformer:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = StoryTransformer(ModelConfig(**payload["model_config"]))
    model = convert_model(model, payload["variant"], payload["sparse_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lanjutkan cerita dari checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    prefix = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    result = model.generate(prefix, args.max_new_tokens, temperature=args.temperature)
    print(decode(result.squeeze(0).tolist()))


if __name__ == "__main__":
    main()
