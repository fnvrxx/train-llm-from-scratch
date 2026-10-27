from experiments.common import run
from experiments.pruned.model import convert_model


def add_options(parser):
    parser.add_argument("--sparsity", type=float, default=0.5)


if __name__ == "__main__":
    run("pruned", add_options, lambda model, args: convert_model(model, args.sparsity))
