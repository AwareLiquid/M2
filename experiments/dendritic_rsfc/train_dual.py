"""RSFC v2 — dual task with rule-invariant sensory output (interference test).

Hypothesis (from the Science 2026 paper): the point of the two-compartment
division is that the RULE update (apical) must not corrupt the SENSORY
representation (soma). v2 therefore trains BOTH outputs:

  category  — high/low under the CURRENT rule (rule-dependent)
  sensory   — is the tone in the upper half? (FIXED mapping, rule-invariant)

The interference metric: SENSORY accuracy on the first trials after a rule
switch. A single shared state must update the rule and (transiently) loses the
sensory mapping; the two-compartment model reads the sensory from the
rule-free soma, so its sensory accuracy should hold.

Success (pre-registered): dendritic post-switch sensory acc > single-LTC by
>= 3 pts in >= 2/3 seeds, with comparable category accuracy.

Run:
    python -m experiments.dendritic_rsfc.train_dual --steps 300 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import DendriticRSFC, SingleLTC
from .train import make_batch, n_params

HERE = Path(__file__).resolve().parent


def masks(switch: torch.Tensor, K: int = 5):
    sw = switch.bool()
    post = torch.zeros_like(sw)
    for k in range(K):
        if k == 0:
            post |= sw
        else:
            post[:, k:] |= sw[:, :-k]
    near = torch.zeros_like(sw)
    for k in range(min(10, sw.shape[1])):
        if k == 0:
            near |= sw
        else:
            near[:, k:] |= sw[:, :-k]
    return post, ~near


@torch.no_grad()
def evaluate(model, episodes, dual: bool) -> dict:
    x, label, switch, _ = episodes
    sens_label = (x > 0.5).long()
    model.eval()
    if dual:
        cat_logits, sens_logits = model.forward_dual(x, label)
    else:
        cat_logits, sens_logits = model(x, label), None
    post, steady = masks(switch)
    out = {}
    cat_ok = (cat_logits.argmax(-1) == label).float()
    out["cat_overall"] = float(cat_ok.mean())
    out["cat_post"] = float(cat_ok[post].mean())
    out["cat_steady"] = float(cat_ok[steady].mean())
    if sens_logits is not None:
        sens_ok = (sens_logits.argmax(-1) == sens_label).float()
        out["sens_overall"] = float(sens_ok.mean())
        out["sens_post"] = float(sens_ok[post].mean())
        out["sens_steady"] = float(sens_ok[steady].mean())
    return out


def train_dual(model, steps, batch, T, lr, seed, device, tag=""):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for step in range(steps):
        x, label, _, _ = make_batch(batch, T, seed=seed * 100000 + step, device=device)
        sens_label = (x > 0.5).long()
        cat_logits, sens_logits = model.forward_dual(x, label)
        loss = F.cross_entropy(cat_logits.reshape(-1, 2), label.reshape(-1)) \
            + F.cross_entropy(sens_logits.reshape(-1, 2), sens_label.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"  [{tag}] step {step+1} loss {loss.item():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=240)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    print(f"=== RSFC v2 dual task (category + rule-invariant sensory), "
          f"T={args.T}, {args.seeds} seeds ===")
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        ev = make_batch(64, args.T, seed=90000 + seed, device=device)
        for name, ctor in (("dendritic", lambda: DendriticRSFC(32, 32)),
                           ("single_ltc", lambda: SingleLTC(48))):
            model = ctor().to(device)
            t0 = time.time()
            train_dual(model, args.steps, args.batch, args.T, args.lr, seed,
                       device, tag=name)
            r = {"seed": seed, "model": name, "params": n_params(model),
                 **evaluate(model, ev, dual=True)}
            rows.append(r)
            print(f"  seed {seed} {name:10s} cat {r['cat_overall']*100:.1f}% "
                  f"(post {r['cat_post']*100:.1f}%) | sens {r['sens_overall']*100:.1f}% "
                  f"(post {r['sens_post']*100:.1f}%, steady {r['sens_steady']*100:.1f}%)")

    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float):
                agg[r["model"]][k].append(v)
    summary = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in agg.items()}
    for m, s in summary.items():
        print(f"\n{m}: cat {s['cat_overall']*100:.1f}% (post {s['cat_post']*100:.1f}%) | "
              f"sens {s['sens_overall']*100:.1f}% (post {s['sens_post']*100:.1f}%, "
              f"steady {s['sens_steady']*100:.1f}%)")

    per_seed = []
    for seed in range(args.seeds):
        d = [r for r in rows if r["seed"] == seed and r["model"] == "dendritic"][0]
        s = [r for r in rows if r["seed"] == seed and r["model"] == "single_ltc"][0]
        per_seed.append({"seed": seed,
                         "sens_post_delta_pts": (d["sens_post"] - s["sens_post"]) * 100,
                         "cat_post_delta_pts": (d["cat_post"] - s["cat_post"]) * 100})
    wins = sum(1 for p in per_seed if p["sens_post_delta_pts"] >= 3.0)
    print(f"\npost-switch SENSORY delta (dendritic - single): "
          f"{[round(p['sens_post_delta_pts'], 1) for p in per_seed]}")
    print("VERDICT:", "PASS" if wins >= 2 else "FAIL")

    out = {"config": vars(args), "rows": rows, "summary": summary,
           "per_seed": per_seed, "pass": wins >= 2}
    (HERE / "report_dual.json").write_text(json.dumps(out, indent=1),
                                           encoding="utf-8")
    lines = ["# RSFC v2 — dual task (interference)", "",
             "| model | cat | cat post | sens | sens post | sens steady |",
             "|---|---|---|---|---|---|"]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['cat_overall']*100:.1f}% | {s['cat_post']*100:.1f}% | "
                     f"{s['sens_overall']*100:.1f}% | {s['sens_post']*100:.1f}% | "
                     f"{s['sens_steady']*100:.1f}% |")
    lines += ["", f"Verdict: {'PASS' if wins >= 2 else 'FAIL'} "
              f"(sens post delta {[round(p['sens_post_delta_pts'],1) for p in per_seed]} pts)"]
    (HERE / "report_dual.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
