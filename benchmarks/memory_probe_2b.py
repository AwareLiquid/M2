"""2b memory probe — cross-window associative recall at 2B scale.

The M1 flagship claim: the recurrent state carries discrete key->value bindings
across a DROPPED KV cache (0.56 recall vs attention's 0.000 by construction).
This probe tests the 200K-step 2b checkpoint on the same protocol:

  seg_a = K1 V1 ... Kn Vn  (byte tokens; keys/values are disjoint byte ranges)
  forward(seg_a, use_cache=True) -> cache
  cross-window: cache = cache.recurrent_only()   (KV dropped, LNN h_prev kept)
  in-window:    full cache                        (attention can copy)
  seg_b = shuffled keys re-queried, each followed by its value (teacher forced)
  accuracy = argmax at each key position == the following value token

Zero-shot: the 2b was trained on byte-level text, NOT this task, so the number
measures what the recurrent state carries by itself. Chance = 1/n_values.

Run (on the server, checkpoint at /root/M2/runs/t2b_30k.pt):
    python benchmarks/memory_probe_2b.py --ckpt runs/t2b_30k.pt
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from m2_training.recipe import Recipe, model_config  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402


def load_model(ckpt: str, device: str) -> tuple[MTLNNModel, dict]:
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    recipe = Recipe(**state["recipe"])
    cfg = model_config(recipe, 256, recipe.sequence_length + 1)
    model = MTLNNModel(cfg)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    return model, {"step": state.get("step"), "recipe": state["recipe"]}


def make_batch(bsz: int, n_pairs: int, key_lo: int, key_hi: int,
               val_lo: int, val_hi: int, g: torch.Generator):
    keys = torch.stack([
        torch.randperm(key_hi - key_lo, generator=g)[:n_pairs] + key_lo
        for _ in range(bsz)
    ])
    vals = torch.randint(val_lo, val_hi, (bsz, n_pairs), generator=g)
    seg_a = torch.stack([keys, vals], dim=-1).reshape(bsz, 2 * n_pairs)
    order = torch.stack([torch.randperm(n_pairs, generator=g) for _ in range(bsz)])
    q_keys = torch.gather(keys, 1, order)
    q_vals = torch.gather(vals, 1, order)
    seg_b = torch.stack([q_keys, q_vals], dim=-1).reshape(bsz, 2 * n_pairs)
    return seg_a, seg_b


@torch.no_grad()
def evaluate(model, device, cross_window: bool, bsz: int = 4, n_pairs: int = 16,
             n_batches: int = 3, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    key_lo, key_hi = 10, 74          # 64 unique keys
    val_lo, val_hi = 100, 164        # 64 value options -> chance 1/64
    accs, losses = [], []
    for _ in range(n_batches):
        seg_a, seg_b = make_batch(bsz, n_pairs, key_lo, key_hi, val_lo, val_hi, g)
        seg_a, seg_b = seg_a.to(device), seg_b.to(device)
        out_a = model(input_ids=seg_a, use_cache=True)
        cache = out_a["cache"]
        if cross_window:
            cache = cache.recurrent_only()      # KV dropped, h_prev kept
        out_b = model(input_ids=seg_b, cache=cache, use_cache=True)
        logits = out_b["logits"]                  # (B, 2n, V)
        key_pos = torch.arange(0, seg_b.shape[1], 2, device=device)
        pred = logits[:, key_pos, :].argmax(-1)
        targets = seg_b[:, key_pos + 1]
        accs.append(float((pred == targets).float().mean()))
        losses.append(float(torch.nn.functional.cross_entropy(
            logits[:, key_pos, :].reshape(-1, logits.shape[-1]),
            targets.reshape(-1))))
    return {"acc": sum(accs) / len(accs), "ce": sum(losses) / len(losses)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_REPO / "runs" / "t2b_200k.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--batches", type=int, default=3)
    args = ap.parse_args()

    model, meta = load_model(args.ckpt, args.device)
    print(f"=== 2b memory probe (ckpt step {meta['step']}, "
          f"{sum(p.numel() for p in model.parameters()):,} params) ===", flush=True)
    inw = evaluate(model, args.device, cross_window=False,
                   bsz=args.bsz, n_pairs=args.n_pairs, n_batches=args.batches)
    xw = evaluate(model, args.device, cross_window=True,
                  bsz=args.bsz, n_pairs=args.n_pairs, n_batches=args.batches)
    chance = 1.0 / 64
    print(f"  in-window  acc {inw['acc']*100:.1f}%  (CE {inw['ce']:.3f})")
    print(f"  cross-window acc {xw['acc']*100:.1f}%  (CE {xw['ce']:.3f})")
    print(f"  chance {chance*100:.1f}%  | M1 reference cross-window 56%")
    verdict = "RECALL" if xw["acc"] > 5 * chance else "NO RECALL"
    print(f"VERDICT: {verdict} (cross-window vs 5x chance)")
    out = {"ckpt": args.ckpt, "step": meta["step"], "in_window": inw,
           "cross_window": xw, "chance": chance, "verdict": verdict,
           "n_pairs": args.n_pairs}
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
