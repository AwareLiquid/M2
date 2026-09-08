import argparse
import json
from pathlib import Path

from .recipe import Recipe, TRAINABLE_EXPERIMENTS
from .runner import TrainingRun


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 synthetic reasoning: train, resume, evaluate")
    parser.add_argument("action", choices=("train", "resume", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30, help="Total target steps, not additional steps")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--experiment", choices=TRAINABLE_EXPERIMENTS)
    parser.add_argument("--task", choices=("pointer_chase", "mod_chain"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.steps < 1 or args.save_every < 1:
        parser.error("--steps and --save-every must be positive")
    if args.action == "train":
        if args.checkpoint.exists():
            parser.error("Checkpoint already exists; use resume or a new path")
        run = TrainingRun(Recipe(
            task=args.task or "pointer_chase", experiment=args.experiment or "baseline",
            seed=0 if args.seed is None else args.seed,
        ), args.device)
    else:
        if any(value is not None for value in (args.task, args.experiment, args.seed)):
            parser.error("Resume/evaluate use the checkpoint recipe; do not override it")
        run = TrainingRun.restore(args.checkpoint, args.device)
    if args.action != "evaluate":
        if args.steps < run.step:
            parser.error("Target steps precede the checkpoint")
        while run.step < args.steps:
            loss = run.train_until(min(args.steps, run.step + args.save_every))
            run.save(args.checkpoint)
            print(json.dumps({"step": run.step, "loss": loss}), flush=True)
    print(json.dumps({"step": run.step, "accuracy": run.evaluate(), "task": run.recipe.task,
                      "experiment": run.recipe.experiment}), flush=True)


if __name__ == "__main__":
    main()
