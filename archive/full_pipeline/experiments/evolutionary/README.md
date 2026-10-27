# Sparse Evolutionary Training for the MLP

This folder implements a SET-style experiment without changing `src/models`.
It initializes random sparse masks on both MLP weight matrices. Every
`--rewire-every` optimizer steps, it removes a fraction of the smallest active
weights and activates an equal number of randomly chosen dormant positions.
The number of active connections stays constant. New weights start at zero;
their optimizer moments are cleared, so subsequent training steps learn them.

The runner supports training from scratch or fine-tuning an existing dense
Transformer checkpoint on the repository's HDF5 token data. Example command
for a capable training machine, from the repository root:

```bash
python3 -m experiments.evolutionary.train \
  --model-config configs/base.json \
  --train-path data/train/pile_train.h5 \
  --val-path data/val/pile_dev.h5 \
  --dense-checkpoint /path/to/dense_checkpoint.pt \
  --output /path/to/evolutionary.pt \
  --device cuda --amp-bf16 \
  --batch-size 4 --grad-accum 8 --steps 1000 \
  --density 0.2 --rewire-fraction 0.3 --rewire-every 100
```

Omit `--dense-checkpoint` to start from random Transformer weights. If using a
smaller architecture, point `--model-config` at its matching base JSON. The
checkpoint must contain a bare Transformer `model_state_dict` with those exact
dimensions. It must not be a reward or PPO wrapper checkpoint. `--resume PATH`
restores this experiment's model, masks, optimizer, and completed step; keep the
same architecture and rewiring settings. The data iterator restarts and
reshuffles on resume, so resumption is not bit-for-bit identical. The output
checkpoint is specific to this experiment and requires building the same
evolutionary model before loading. The helper
`experiments.evolutionary.model.load_experiment_checkpoint(path, device)`
rebuilds it for generation or evaluation.

To check correctness on the training machine:

```bash
python3 -m unittest experiments.evolutionary.test_evolutionary
```

This implementation multiplies dense matrices by masks before `F.linear`.
It studies changing connectivity and quality; it does not provide sparse-kernel
speed or memory savings for the weight and optimizer tensors. The original SET
paper rewires at epoch boundaries; this runner uses optimizer-step intervals
because the HDF5 pretraining stream has no natural epoch boundary.
