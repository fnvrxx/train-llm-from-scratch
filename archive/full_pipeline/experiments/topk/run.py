from experiments.common import run
from experiments.topk.model import convert_model


def add_options(parser):
    parser.add_argument("--active-fraction", type=float, default=0.25)


if __name__ == "__main__":
    run("topk", add_options, lambda model, args: convert_model(model, args.active_fraction))
