"""Latih satu varian dengan data dan prosedur yang dipakai semua eksperimen."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path

import torch

from small_llm.config import ExperimentConfig, load_config, resolve_device
from small_llm.data import StoryCorpus, load_stories, sample_batch
from small_llm.model import StoryTransformer
from small_llm.sparse import VARIANTS, active_connections, convert_model, moe_aux_loss, rewire_set


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1_000_000 if platform.system() == "Darwin" else value / 1024


@torch.no_grad()
def evaluate(
    model: StoryTransformer, corpus: StoryCorpus, cfg: ExperimentConfig, device: str
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(cfg.train.seed + 2)
    losses = []
    for _ in range(cfg.train.eval_batches):
        x, y = sample_batch(
            corpus.valid,
            batch_size=cfg.train.batch_size,
            context_length=cfg.model.context_length,
            generator=generator,
            device=device,
        )
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def train_variant(cfg: ExperimentConfig, variant: str, *, overwrite: bool = False) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Varian tidak dikenal: {variant}")
    device = resolve_device(cfg.train.device)
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    corpus = load_stories(cfg.train.data_dir, seed=cfg.train.seed,
                          valid_fraction=cfg.train.valid_fraction)
    if min(corpus.train.numel(), corpus.valid.numel()) <= cfg.model.context_length:
        raise ValueError("Data train atau validasi lebih pendek dari context_length")
    out = Path(cfg.train.out_dir) / variant
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} sudah ada; pakai --overwrite jika ingin menggantinya")
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.train.seed)
    model = StoryTransformer(cfg.model)
    model = convert_model(model, variant, asdict(cfg.sparse)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay)
    generator = torch.Generator().manual_seed(cfg.train.seed + 1)
    parameters = sum(p.numel() for p in model.parameters())
    active, total = active_connections(model)
    if variant in ("pruned", "set"):
        mlp_usage = f"{active}/{total} koneksi aktif"
    elif variant == "topk":
        mlp_usage = f"{model.blocks[0].mlp.k}/{4 * cfg.model.n_embed} neuron per token"
    elif variant == "moe":
        mlp_usage = f"{cfg.sparse.top_k}/{cfg.sparse.n_experts} expert per token"
    else:
        mlp_usage = "semua koneksi"
    print(f"{variant}: device={device}, parameter={parameters:,}, "
          f"train={corpus.train.numel():,} byte, valid={corpus.valid.numel():,} byte")
    if total:
        print(f"  koneksi MLP aktif={active:,}/{total:,} ({active / total:.1%})")

    metrics_path = out / "metrics.jsonl"
    training_seconds = 0.0
    last_train_loss = 0.0
    initial_valid_loss = evaluate(model, corpus, cfg, device)
    last_valid_loss = initial_valid_loss
    with metrics_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"step": 0, "train_loss": None,
                              "valid_loss": initial_valid_loss, "step_seconds": 0.0}) + "\n")
        print(f"  langkah 0: valid_loss={initial_valid_loss:.4f}")
        for step in range(1, cfg.train.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            started = time.perf_counter()
            train_loss = 0.0
            for _ in range(cfg.train.grad_accum):
                x, y = sample_batch(
                    corpus.train,
                    batch_size=cfg.train.batch_size,
                    context_length=cfg.model.context_length,
                    generator=generator,
                    device=device,
                )
                _, token_loss = model(x, y)
                loss = token_loss
                if variant == "moe":
                    loss = loss + cfg.sparse.aux_coef * moe_aux_loss(model)
                (loss / cfg.train.grad_accum).backward()
                train_loss += token_loss.item() / cfg.train.grad_accum
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"Loss tidak finite pada langkah {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            if variant == "set":
                for block in model.blocks:
                    block.mlp.enforce_masks()
                if step % cfg.sparse.rewire_every == 0:
                    rewire_set(model, optimizer, cfg.sparse.rewire_fraction)
            _sync(device)
            elapsed = time.perf_counter() - started
            training_seconds += elapsed
            last_train_loss = train_loss

            should_eval = step % cfg.train.eval_every == 0 or step == cfg.train.steps
            if should_eval:
                last_valid_loss = evaluate(model, corpus, cfg, device)
            if step % cfg.train.log_every == 0 or should_eval:
                record = {"step": step, "train_loss": train_loss,
                          "valid_loss": last_valid_loss if should_eval else None,
                          "step_seconds": elapsed}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                message = f"  langkah {step}/{cfg.train.steps}: train_loss={train_loss:.4f}"
                if should_eval:
                    message += f", valid_loss={last_valid_loss:.4f}"
                print(message)

    summary = {
        "variant": variant,
        "device": device,
        "parameters": parameters,
        "mlp_usage": mlp_usage,
        "active_mlp_connections": active if total else None,
        "total_mlp_connections": total if total else None,
        "train_tokens": corpus.train.numel(),
        "valid_tokens": corpus.valid.numel(),
        "train_files": corpus.train_files,
        "valid_files": corpus.valid_files,
        "steps": cfg.train.steps,
        "initial_valid_loss": initial_valid_loss,
        "train_loss": last_train_loss,
        "valid_loss": last_valid_loss,
        "train_seconds": training_seconds,
        "train_tokens_per_second": (
            cfg.train.steps * cfg.train.batch_size * cfg.train.grad_accum
            * cfg.model.context_length / training_seconds
        ),
        "peak_process_rss_mb": _peak_rss_mb(),
        "sparse_config": asdict(cfg.sparse),
    }
    checkpoint = {"model": model.state_dict(), "model_config": asdict(cfg.model),
                  "sparse_config": asdict(cfg.sparse), "variant": variant}
    checkpoint_temp = out / "model.pt.tmp"
    torch.save(checkpoint, checkpoint_temp)
    checkpoint_temp.replace(out / "model.pt")
    summary_temp = out / "summary.json.tmp"
    summary_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    summary_temp.replace(out / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih satu Small LLM cerita pendek")
    parser.add_argument("--config", default="configs/small_m1.json")
    parser.add_argument("--variant", choices=VARIANTS, default="dense")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train_variant(load_config(args.config), args.variant, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

