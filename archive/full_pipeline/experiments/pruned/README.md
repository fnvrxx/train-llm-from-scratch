# Magnitude pruning

`model.py` masks the lowest-magnitude weights separately in both MLP matrices.
Masks are buffers in the experiment checkpoint. Masked weights receive zero
gradient from the loss. The tensors and PyTorch linear operations remain dense,
so this is an effective-weight sparsity experiment, not a speed optimization.
The original dense MLP weights can initialize the variant exactly before masks
are applied. Fine-tuning is needed to recover quality after pruning.
