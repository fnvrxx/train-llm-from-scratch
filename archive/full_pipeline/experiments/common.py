"""Small synthetic comparison runner shared by the isolated MLP experiments."""

import argparse
import copy
import time
from pathlib import Path

import torch

from src.models.transformer import Transformer


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _measure(model: Transformer, tokens: torch.Tensor, iterations: int, device: str) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(2):
            model(tokens)
        _sync(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(tokens)
        _sync(device)
    return (time.perf_counter() - start) * 1000 / iterations


def run(name: str, add_options, convert) -> None:
    parser = argparse.ArgumentParser(description=f"Compare the {name} MLP with the dense baseline")
    parser.add_argument("--n-embed", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=12)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--benchmark-iters", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dense-checkpoint", type=Path)
    parser.add_argument("--load-variant", type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--aux-coef", type=float, default=0.01 if name == "moe" else 0.0)
    add_options(parser)
    args = parser.parse_args()
    if args.n_embed <= 0 or args.n_head <= 0 or args.n_embed % args.n_head:
        parser.error("n-embed must be positive and divisible by n-head")
    if args.n_blocks < 1 or args.batch_size < 1 or args.steps < 1 or args.benchmark_iters < 1:
        parser.error("n-blocks, batch-size, steps and benchmark-iters must be positive")
    if not 1 <= args.seq_length <= args.context_length or args.vocab_size < 2:
        parser.error("require 1 <= seq-length <= context-length and vocab-size >= 2")
    if args.dense_checkpoint and args.load_variant:
        parser.error("choose one of --dense-checkpoint or --load-variant")

    torch.manual_seed(args.seed)
    dense = Transformer(args.n_head, args.n_embed, args.context_length, args.vocab_size, args.n_blocks)
    if args.dense_checkpoint:
        payload = torch.load(args.dense_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        dense.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    baseline = copy.deepcopy(dense).to(args.device)
    variant = convert(dense, args).to(args.device)
    if args.load_variant:
        payload = torch.load(args.load_variant, map_location="cpu", weights_only=False)
        expected = {key: value for key, value in vars(args).items() if key not in
                    {"steps", "benchmark_iters", "device", "seed", "dense_checkpoint", "load_variant", "save"}}
        saved = {key: payload["config"][key] for key in expected}
        if saved != expected or payload["experiment"] != name:
            raise ValueError("Variant checkpoint architecture/options do not match this run")
        variant.load_state_dict(payload["model_state_dict"], strict=True)

    tokens = torch.randint(args.vocab_size, (args.batch_size, args.seq_length), device=args.device)
    targets = torch.randint(args.vocab_size, tokens.shape, device=args.device)
    base_ms = _measure(baseline, tokens, args.benchmark_iters, args.device)
    variant_ms = _measure(variant, tokens, args.benchmark_iters, args.device)
    base_params = sum(p.numel() for p in baseline.parameters())
    variant_params = sum(p.numel() for p in variant.parameters())
    print(f"{name}: params dense={base_params:,}, variant={variant_params:,}")
    print(f"inference ms/batch dense={base_ms:.3f}, variant={variant_ms:.3f} "
          f"(synthetic input, {args.device})")

    variant.train()
    optimizer = torch.optim.AdamW(variant.parameters(), lr=1e-3)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = variant(tokens, targets)
        if name == "moe":
            loss = loss + args.aux_coef * sum(block.mlp.aux_loss for block in variant.attn_blocks)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1} shape={tuple(logits.shape)} loss={loss.item():.4f}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"experiment": name, "config": vars(args), "model_state_dict": variant.state_dict()}, args.save)
        print(f"saved {args.save}")
