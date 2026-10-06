"""Post-SFT eval — held-out instruction generations, base vs SFT side by side.

    python -m m2_training.sft_eval <sft_ckpt> [corpus_manifest] [base_ckpt]

加载 base 与 SFT 两个检查点，对同一批 held-out 短指令做贪心解码对照
（byte 级, eos=换行）。服务器口径见 docs/SFT_RUN1.md。
"""
import sys
from pathlib import Path

import numpy as np
import torch

from .runner import TrainingRun

DEFAULT_CORPUS = "/root/M2/sft_data/corpus128/manifest.json"
DEFAULT_BASE = "/root/M2/runs/t2b_30k.pt"

PROMPTS = [
    "Compute the sum of 9 and 5.",
    "What is the capital of France?",
    "Translate 'good morning' to Spanish.",
    "Convert 1 kilometer to meters.",
    "Give me a synonym for happy.",
    "What color is the sky on a clear day?",
    "Name the largest planet in our solar system.",
    "Rewrite this in past tense: I eat an apple.",
]


def load(path: str, corpus: str) -> TrainingRun:
    run = TrainingRun.restore(Path(path), device="cuda", corpus=corpus,
                              task="sft")
    run.model.eval()
    return run


@torch.no_grad()
def respond(run: TrainingRun, prompt: str, max_new: int = 40) -> str:
    raw = (prompt + "\n\n").encode("utf-8")
    ids = torch.from_numpy(
        np.frombuffer(raw, dtype=np.uint8).astype(np.int64))[None].cuda()
    out = run.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=10, pad_token_id=0)
    gen = out[0].cpu().numpy().tolist()[ids.shape[1]:]
    text = bytes(gen).decode("utf-8", "replace")
    return text.split("\n")[0][:90]


def main() -> None:
    sft_ckpt = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/t2b_sft_v1.pt"
    corpus = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CORPUS
    base_ckpt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_BASE

    print("loading base ...", flush=True)
    base = load(base_ckpt, corpus)
    print("loading sft ...", flush=True)
    sft = load(sft_ckpt, corpus)

    print("\n=== held-out generations (greedy) ===", flush=True)
    for p in PROMPTS:
        print(f"\nQ: {p}")
        print(f"  base: {respond(base, p)}")
        print(f"  sft : {respond(sft, p)}", flush=True)


if __name__ == "__main__":
    main()
