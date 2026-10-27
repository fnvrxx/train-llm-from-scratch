"""Jalankan tiap varian dalam proses terpisah agar angka memori dapat dibandingkan."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from small_llm.config import load_config
from small_llm.sparse import VARIANTS
from small_llm.train import train_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandingkan dense dan varian sparse")
    parser.add_argument("--config", default="configs/small_m1.json")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--in-process", action="store_true",
                        help="jalankan tanpa subprocess; kolom peak RSS tidak dapat dibandingkan")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.train.out_dir)
    existing = [variant for variant in args.variants if (out / variant).exists()]
    if existing and not args.overwrite:
        parser.error(f"hasil sudah ada untuk {', '.join(existing)} di {out}; "
                     "gunakan --overwrite hanya jika ingin mengganti hasil lama")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in args.variants:
        print(f"\nMenjalankan {variant}", flush=True)
        if args.in_process:
            row = train_variant(cfg, variant, overwrite=args.overwrite)
            row["peak_process_rss_mb"] = None
        else:
            command = [sys.executable, "-m", "small_llm.train", "--config", args.config,
                       "--variant", variant]
            if args.overwrite:
                command.append("--overwrite")
            subprocess.run(command, check=True)
            row = json.loads((out / variant / "summary.json").read_text(encoding="utf-8"))
        rows.append(row)

    columns = ["variant", "parameters", "mlp_usage", "active_mlp_connections", "initial_valid_loss",
               "train_loss", "valid_loss", "train_tokens_per_second", "peak_process_rss_mb"]
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("\nPerbandingan (loss validasi lebih kecil lebih baik):")
    for row in rows:
        rss = (f"{row['peak_process_rss_mb']:.0f} MB"
               if row["peak_process_rss_mb"] is not None else "tidak diukur")
        print(f"  {row['variant']:7} val={row['valid_loss']:.4f} "
              f"tok/s={row['train_tokens_per_second']:.0f} "
              f"RSS={rss}")
    print(f"Tabel: {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()

