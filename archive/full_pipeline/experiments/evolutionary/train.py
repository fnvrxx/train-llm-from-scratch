"""Train or fine-tune a SET Transformer on the repository's tokenized HDF5 data."""

import argparse
import contextlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from data_loader.data_loader import get_batch_iterator
from experiments.evolutionary.model import convert_model
from experiments.evolutionary.rewire import active_connections, enforce_model_masks, rewire_model
from src.models.transformer import Transformer


MODEL_FIELDS = ("n_head", "n_embed", "context_length", "vocab_size", "n_blocks")


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse Evolutionary Training for Transformer MLPs")
    parser.add_argument("--model-config", type=Path, default=Path("configs/base.json"))
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path)
    parser.add_argument("--dense-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--rewire-fraction", type=float, default=0.3)
    parser.add_argument("--rewire-every", type=int, default=100)
    parser.add_argument("--freeze-after", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--amp-bf16", action="store_true")
    args = parser.parse_args()
    if args.dense_checkpoint and args.resume:
        parser.error("choose either --dense-checkpoint or --resume")
    if args.dense_checkpoint and args.output.resolve() == args.dense_checkpoint.resolve():
        parser.error("--output must not overwrite --dense-checkpoint")
    if not 0 < args.density < 1 or not 0 <= args.rewire_fraction <= 1:
        parser.error("density must be in (0, 1) and rewire-fraction in [0, 1]")
    if min(args.batch_size, args.grad_accum, args.steps, args.rewire_every,
           args.eval_every, args.eval_batches, args.save_every) < 1:
        parser.error("batch-size, grad-accum, steps, and intervals must be positive")
    if args.freeze_after < 0 or args.lr <= 0 or args.grad_clip <= 0:
        parser.error("freeze-after must be non-negative; lr and grad-clip must be positive")
    if args.amp_bf16 and not args.device.startswith("cuda"):
        parser.error("--amp-bf16 requires a CUDA device")
    return args


def model_dimensions(path: Path) -> dict:
    with path.open() as handle:
        cfg = json.load(handle)
    dims = {key: int(cfg[key]) for key in MODEL_FIELDS}
    if dims["n_embed"] % dims["n_head"]:
        raise ValueError("n_embed must be divisible by n_head")
    return dims


def check_data(path: Path, batch_size: int, context_length: int) -> None:
    with h5py.File(path, "r") as handle:
        count = len(handle["tokens"])
    if (count - 1) // context_length < batch_size:
        raise ValueError(f"{path} has fewer than {batch_size} full training sequences")


def autocast(args):
    if args.amp_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, batches, args) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        inputs, targets = next(batches)
        with autocast(args):
            _, loss = model(inputs, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: Path, model, optimizer, step: int, args, dims: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "evolutionary",
        "step": step,
        "model_config": dims,
        "density": args.density,
        "rewire_fraction": args.rewire_fraction,
        "rewire_every": args.rewire_every,
        "freeze_after": args.freeze_after,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dims = model_dimensions(args.model_config)
    check_data(args.train_path, args.batch_size, dims["context_length"])
    if args.val_path:
        check_data(args.val_path, args.batch_size, dims["context_length"])

    model = Transformer(dims["n_head"], dims["n_embed"], dims["context_length"],
                        dims["vocab_size"], dims["n_blocks"])
    if args.dense_checkpoint:
        checkpoint = torch.load(args.dense_checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    model = convert_model(model, args.density).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        settings = ("density", "rewire_fraction", "rewire_every", "freeze_after")
        if checkpoint.get("experiment") != "evolutionary" or checkpoint["model_config"] != dims:
            raise ValueError("Resume checkpoint has a different experiment or model architecture")
        if any(checkpoint[key] != getattr(args, key) for key in settings):
            raise ValueError("Resume checkpoint has different sparsity or rewiring settings")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])

    train_batches = get_batch_iterator(str(args.train_path), args.batch_size,
                                       dims["context_length"], device=args.device)
    val_batches = (get_batch_iterator(str(args.val_path), args.batch_size,
                                     dims["context_length"], device=args.device)
                   if args.val_path else None)
    active, total = active_connections(model)
    print(f"MLP connections active={active:,}/{total:,} ({active / total:.2%})")
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(args.grad_accum):
            inputs, targets = next(train_batches)
            with autocast(args):
                _, loss = model(inputs, targets)
            (loss / args.grad_accum).backward()
            loss_total += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        enforce_model_masks(model)

        if (step % args.rewire_every == 0 and step < args.steps and
                (args.freeze_after == 0 or step < args.freeze_after)):
            changed = rewire_model(model, optimizer, args.rewire_fraction)
            print(f"step={step} rewired={changed:,} connections")
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            message = f"step={step} train_loss={loss_total:.4f}"
            if val_batches:
                message += f" val_loss={evaluate(model, val_batches, args):.4f}"
            print(message)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.output, model, optimizer, step, args, dims)
            print(f"saved {args.output}")


if __name__ == "__main__":
    main()
