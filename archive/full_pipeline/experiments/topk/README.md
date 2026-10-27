# Per-token top-k neurons

`model.py` computes the full first projection, selects the largest ReLU
activations for each token, then sums only the selected columns of the second
projection. It preserves the dense MLP parameter names and can initialize
directly from dense weights. At `--active-fraction 1`, output matches the
original MLP. Smaller fractions change its function and require fine-tuning.
Gather and indexing overhead can outweigh saved arithmetic, especially for
small models or batches; measure on the target hardware.
