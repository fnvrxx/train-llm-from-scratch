from experiments.common import run
from experiments.moe.model import convert_model


def add_options(parser):
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)


if __name__ == "__main__":
    run("moe", add_options, lambda model, args: convert_model(model, args.num_experts, args.top_k))
