import argparse
import json
from pathlib import Path

from .recipe import Recipe, TRAINABLE_EXPERIMENTS
from .runner import TrainingRun


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 reasoning and text: train, resume, evaluate")
    parser.add_argument("action", choices=("train", "resume", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30, help="Total target steps, not additional steps")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--experiment", choices=TRAINABLE_EXPERIMENTS)
    parser.add_argument("--task", choices=("pointer_chase", "mod_chain", "text", "sft"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--size", choices=("probe", "medium", "2b"))
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--rope-scale", type=float,
                        help="NTK/YaRN RoPE 缩放 (>=1.0; 后训练扩上下文用)")
    parser.add_argument("--difficulty", type=int)
    parser.add_argument("--n-values", type=int)
    args = parser.parse_args()
    if args.steps < 1 or args.save_every < 1:
        parser.error("--steps and --save-every must be positive")
    if args.action == "train":
        if args.checkpoint.exists():
            parser.error("Checkpoint already exists; use resume or a new path")
        run = TrainingRun(Recipe(
            task=args.task or "pointer_chase", experiment=args.experiment or "baseline",
            seed=0 if args.seed is None else args.seed,
            corpus=str(args.corpus.resolve()) if args.corpus else None,
            size=args.size or "probe",
            sequence_length=128 if args.sequence_length is None else args.sequence_length,
            batch=128 if args.batch is None else args.batch,
            grad_accum=1 if args.grad_accum is None else args.grad_accum,
            lr=0.0003 if args.lr is None else args.lr,
            difficulty=2 if args.difficulty is None else args.difficulty,
            n_values=8 if args.n_values is None else args.n_values,
        ), args.device)
    else:
        if any(value is not None for value in (
            args.experiment, args.seed, args.size, args.batch, args.grad_accum,
        )):
            parser.error("Resume/evaluate use the checkpoint recipe; overridable "
                         "on resume: --task/--corpus/--lr/--sequence-length/"
                         "--rope-scale")
        if args.action != "resume" and any(value is not None for value in (
            args.task, args.corpus, args.lr, args.sequence_length,
            args.rope_scale,
        )):
            parser.error("task/corpus/lr/sequence-length/rope-scale overrides "
                         "are only valid for resume")
        if args.task is not None and args.corpus is None:
            parser.error("--task override requires --corpus (the new data path)")
        run = TrainingRun.restore(args.checkpoint, args.device,
                                  corpus=args.corpus, task=args.task, lr=args.lr,
                                  sequence_length=args.sequence_length,
                                  rope_scale=args.rope_scale)
    if args.action != "evaluate":
        if args.steps < run.step:
            parser.error("Target steps precede the checkpoint")
        while run.step < args.steps:
            loss = run.train_until(min(args.steps, run.step + args.save_every))
            run.save(args.checkpoint)
            print(json.dumps({"step": run.step, "loss": loss}), flush=True)
    metric = "validation_ppl" if run.recipe.task in ("text", "sft") else "accuracy"
    print(json.dumps({"step": run.step, metric: run.evaluate(), "task": run.recipe.task,
                      "experiment": run.recipe.experiment}), flush=True)


if __name__ == "__main__":
    main()
