"""Post-SFT eval — held-out instruction generations, base vs SFT side by side.

    python -m m2_training.sft_eval <sft_ckpt> [sft_corpus] [base_ckpt] [base_corpus]

每个检查点按其语料的 sequence_length 装载（manifest 自带），所有模型走
SFT 数据路径（同一任务口径）。byte 级贪心解码（eos=换行）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .runner import TrainingRun

DEFAULT_SFT_CORPUS = "/root/M2/sft_data/corpus512/manifest.json"
DEFAULT_BASE = "/root/M2/runs/t2b_30k.pt"
DEFAULT_BASE_CORPUS = "/root/M2/sft_data/corpus128/manifest.json"

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


def corpus_seq(corpus: str) -> int:
    meta = json.loads(Path(corpus).read_text(encoding="utf-8"))
    return int(meta["sequence_length"])


def load(path: str, corpus: str) -> TrainingRun:
    run = TrainingRun.restore(Path(path), device="cuda", corpus=corpus,
                              task="sft", sequence_length=corpus_seq(corpus))
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
    sft_ckpt = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/t2b_sft_v2.pt"
    sft_corpus = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SFT_CORPUS
    base_ckpt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_BASE
    base_corpus = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_BASE_CORPUS

    print("loading base ...", flush=True)
    base = load(base_ckpt, base_corpus)
    print("loading sft ...", flush=True)
    sft = load(sft_ckpt, sft_corpus)

    print("\n=== held-out generations (greedy) ===", flush=True)
    for p in PROMPTS:
        print(f"\nQ: {p}")
        print(f"  base: {respond(base, p)}")
        print(f"  sft : {respond(sft, p)}", flush=True)


if __name__ == "__main__":
    main()
