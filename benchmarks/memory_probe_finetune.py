"""Memory probe with task adaptation — can the recurrent state LEARN to carry
key->value bindings across a dropped KV cache?

Protocol (the meaningful version; zero-shot is uninformative for an untrained-
for-recall model):
  train:  seg_a -> cache -> recurrent_only() -> seg_b, CE on the value tokens
  eval:   same cross-window condition + the in-window condition (full cache)

If the cross-window recall rises well above chance, the recurrent state carries
discrete bindings (the M1 claim, at this scale/model). If it stays at chance
while in-window is high, attention does the work and the STATE does not.

Run:
    python benchmarks/memory_probe_finetune.py --ckpt runs/probe_mem.pt \
        --steps 400 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from m2_training.recipe import Recipe, model_config  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402

KEY_LO, KEY_HI = 10, 74       # 64 unique keys
VAL_LO, VAL_HI = 100, 164     # 64 values -> chance 1/64


def make_batch(bsz, n_pairs, g):
    keys = torch.stack([torch.randperm(KEY_HI - KEY_LO, generator=g)[:n_pairs] + KEY_LO
                        for _ in range(bsz)])
    vals = torch.randint(VAL_LO, VAL_HI, (bsz, n_pairs), generator=g)
    seg_a = torch.stack([keys, vals], -1).reshape(bsz, 2 * n_pairs)
    order = torch.stack([torch.randperm(n_pairs, generator=g) for _ in range(bsz)])
    seg_b = torch.stack([keys.gather(1, order), vals.gather(1, order)],
                        -1).reshape(bsz, 2 * n_pairs)
    return seg_a, seg_b


def forward_two_seg(model, seg_a, seg_b, cross_window: bool):
    out_a = model(input_ids=seg_a, use_cache=True)
    cache = out_a["cache"]
    if cross_window:
        cache = cache.recurrent_only()
    out_b = model(input_ids=seg_b, cache=cache, use_cache=True)
    return out_b["logits"]


def value_loss(logits, seg_b):
    key_pos = torch.arange(0, seg_b.shape[1], 2, device=seg_b.device)
    targets = seg_b[:, key_pos + 1]
    return F.cross_entropy(logits[:, key_pos, :].reshape(-1, logits.shape[-1]),
                           targets.reshape(-1))


@torch.no_grad()
def evaluate(model, device, cross_window: bool, n_batches=3, bsz=4,
             n_pairs=16, seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    accs, ces = [], []
    for _ in range(n_batches):
        seg_a, seg_b = make_batch(bsz, n_pairs, g)
        seg_a, seg_b = seg_a.to(device), seg_b.to(device)
        logits = forward_two_seg(model, seg_a, seg_b, cross_window)
        key_pos = torch.arange(0, seg_b.shape[1], 2, device=device)
        pred = logits[:, key_pos, :].argmax(-1)
        targets = seg_b[:, key_pos + 1]
        accs.append(float((pred == targets).float().mean()))
        ces.append(float(value_loss(logits, seg_b)))
    return sum(accs) / len(accs), sum(ces) / len(ces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_REPO / "runs" / "probe_mem.pt"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cross-train", action="store_true", default=True)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    recipe = Recipe(**state["recipe"])
    cfg = model_config(recipe, 256, recipe.sequence_length + 1)
    model = MTLNNModel(cfg)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(args.device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== memory adaptation (ckpt step {state.get('step')}, "
          f"{n_params:,} params, cross-window training) ===", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(1)
    for step in range(args.steps):
        seg_a, seg_b = make_batch(args.bsz, args.n_pairs, g)
        seg_a, seg_b = seg_a.to(args.device), seg_b.to(args.device)
        logits = forward_two_seg(model, seg_a, seg_b, cross_window=True)
        loss = value_loss(logits, seg_b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1} loss {loss.item():.3f}", flush=True)

    inw = evaluate(model, args.device, cross_window=False, n_pairs=args.n_pairs)
    xw = evaluate(model, args.device, cross_window=True, n_pairs=args.n_pairs)
    chance = 1.0 / (VAL_HI - VAL_LO)
    print(f"\nin-window  acc {inw[0]*100:.1f}% (CE {inw[1]:.3f})")
    print(f"cross-window acc {xw[0]*100:.1f}% (CE {xw[1]:.3f})")
    print(f"chance {chance*100:.1f}%")
    verdict = "STATE CARRIES BINDINGS" if xw[0] > max(5 * chance, 0.10) else "STATE DOES NOT CARRY"
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
