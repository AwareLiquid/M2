import argparse
import json

import torch

from mt_lnn.model import MTLNNModel

from .recipe import Recipe, model_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Count model parameters on the meta device without allocating weights")
    parser.add_argument("--size", choices=("probe", "2b"), default="probe")
    parser.add_argument("--vocab-size", type=int, default=32768)
    args = parser.parse_args()
    if args.vocab_size < 2:
        parser.error("Vocabulary size must be >= 2")
    config = model_config(Recipe(size=args.size), args.vocab_size, 128)
    with torch.device("meta"):
        model = MTLNNModel(config)
    count = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"size": args.size, "parameters": count, "vocab_size": args.vocab_size,
                      "fp32_adam_training_state_bytes_lower_bound": count * 16,
                      "includes_activations": False}))


if __name__ == "__main__":
    main()
