# Mixture of experts

`model.py` partitions the dense MLP hidden neurons into equal-width experts,
then routes each token to `--top-k` of `--num-experts`. The total expert width
matches the original MLP; a router adds a few parameters. The shared runner
adds an auxiliary load-balancing loss during training. With every expert
selected and a uniform router, the variant reconstructs the original dense
MLP. With fewer selected experts, the function changes and needs fine-tuning.

This is an experimental router, with no capacity limit or distributed expert
parallelism. Token dispatch overhead may dominate small runs. Monitor per-expert
load and validation quality before treating it as a training replacement.
