# Sparse feed-forward experiments

These three variants replace only `Block.mlp` on a fresh `Transformer`. The
original model and training scripts remain unchanged. Run from the repository
root with Python 3 and PyTorch installed:

```bash
python3 -m experiments.pruned.run --sparsity 0.5
python3 -m experiments.topk.run --active-fraction 0.25
python3 -m experiments.moe.run --num-experts 4 --top-k 2
python3 -m unittest experiments.test_experiments
```

`evolutionary/` is a fourth, separate experiment. It changes sparse
connectivity during training and has a real HDF5 training runner. See its
[README](evolutionary/README.md) before launching it on a training machine.

The three runners above use a tiny synthetic next-token task by default. They print the
parameter count, batch inference latency for the dense baseline and variant,
and a few training losses. These numbers are a functionality check, not a
quality benchmark. Use the same `--device`, dimensions, batch size, sequence
length and `--benchmark-iters` when comparing variants. For a real comparison,
train on the same dataset and compare validation loss, generation quality,
tokens per second, and peak memory on the target device.

`--dense-checkpoint PATH` initializes from a bare Transformer checkpoint. Pass
the matching model dimensions on the CLI. PPO/reward wrapper checkpoints need
their `transformer.` prefix removed first. `--save PATH` writes an experiment
checkpoint; `--load-variant PATH` resumes its model weights with matching
architecture options. The runner does not resume optimizer state or a training
schedule. Experiment checkpoints are not interchangeable with the existing
production loaders.

See each experiment folder for its computational and migration limits.
