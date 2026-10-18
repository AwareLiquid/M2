"""Train + evaluate the dendritic two-compartment unit vs point-neuron baselines
on the RSFC (rule-switching flexible categorization) task.

Metrics:
  overall   accuracy over all trials
  post_sw   accuracy on the first K=5 trials after a rule switch (rule must be
            re-inferred from reward feedback)
  steady    accuracy on trials >10 after a switch (rule established)
Ablations on the trained dendritic model: soma_only / apical_only / sst
("dendritic inhibition" — the paper's causal manipulation).

Run:
    python -m experiments.dendritic_rsfc.train --steps 300 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import feedback, rsfc_episodes
from .model import DendriticRSFC, GRURSFC, MLPRSFC, SingleLTC

HERE = Path(__file__).resolve().parent


def make_batch(batch: int, T: int, seed: int, device: str):
    x, _, label, rule, switch = rsfc_episodes(batch, T, seed=seed)
    return (torch.from_numpy(x).to(device), torch.from_numpy(label).to(device),
            torch.from_numpy(switch).to(device), torch.from_numpy(rule).to(device))


def rule_probe(states: torch.Tensor, rule: torch.Tensor, steps: int = 300,
               lr: float = 1e-2) -> float:
    """Linear probe: decode the hidden rule from a state sequence (B,T,D)."""
    b, T, d = states.shape
    probe = torch.nn.Linear(d, 2).to(states.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    x = states.reshape(b * T, d).detach()
    y = rule.reshape(-1)
    for _ in range(steps):
        loss = F.cross_entropy(probe(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((probe(x).argmax(-1) == y).float().mean())


@torch.no_grad()
def evaluate(model, episodes: tuple, mode: str = "full", K: int = 5) -> dict:
    x, label, switch, rule = episodes
    model.eval()
    logits = model(x, label, mode=mode)
    pred = logits.argmax(-1)
    correct = (pred == label).float()
    sw = switch.bool()
    post_mask = torch.zeros_like(sw)
    for k in range(K):                       # first K trials after each switch
        sh = torch.zeros_like(sw)
        if k > 0:
            sh[:, k:] = sw[:, :-k]
        else:
            sh = sw
        post_mask |= sh
    # steady: >10 trials after a switch
    steady_mask = torch.zeros_like(sw)
    steady_mask[:, 10:] = ~sw[:, :-10] if sw.shape[1] > 10 else steady_mask[:, 10:]
    # simpler: steady = not within 10 of a switch
    near = torch.zeros_like(sw)
    for k in range(min(10, sw.shape[1])):
        if k == 0:
            near |= sw
        else:
            near[:, k:] |= sw[:, :-k]
    steady_mask = ~near
    return {"overall": float(correct.mean()),
            "post_sw": float(correct[post_mask].mean()) if post_mask.any() else float("nan"),
            "steady": float(correct[steady_mask].mean()) if steady_mask.any() else float("nan"),
            "n_post": int(post_mask.sum())}


def train_model(model, steps: int, batch: int, T: int, lr: float,
                seed: int, device: str, tag: str = "") -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    last = None
    for step in range(steps):
        x, label, _, _ = make_batch(batch, T, seed=seed * 100000 + step, device=device)
        logits = model(x, label)
        loss = F.cross_entropy(logits.reshape(-1, 2), label.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss)
        if (step + 1) % 100 == 0:
            print(f"  [{tag}] step {step+1} loss {loss.item():.4f}", flush=True)
    return last


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


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
    probes = []
    print(f"=== RSFC: dendritic two-compartment vs point-neuron baselines "
          f"(T={args.T}, {args.seeds} seeds) ===")
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        eval_eps = make_batch(64, args.T, seed=90000 + seed, device=device)
        for name, ctor in (
            ("dendritic", lambda: DendriticRSFC(32, 32)),
            ("single_ltc", lambda: SingleLTC(64)),
            ("single_ltc_48", lambda: SingleLTC(48)),
            ("gru", lambda: GRURSFC(64)),
            ("mlp", lambda: MLPRSFC(64)),
        ):
            model = ctor().to(device)
            t0 = time.time()
            train_model(model, args.steps, args.batch, args.T, args.lr, seed,
                        device, tag=name)
            res = evaluate(model, eval_eps, mode="full")
            r = {"seed": seed, "model": name, "params": n_params(model),
                 "train_sec": round(time.time() - t0, 1), **res}
            rows.append(r)
            print(f"  seed {seed} {name:10s} params {r['params']:,}  "
                  f"overall {r['overall']*100:.1f}%  post_sw {r['post_sw']*100:.1f}%  "
                  f"steady {r['steady']*100:.1f}%")
            if name == "dendritic":
                for mode in ("soma_only", "apical_only", "sst"):
                    abl = evaluate(model, eval_eps, mode=mode)
                    rows.append({"seed": seed, "model": f"dendritic_{mode}",
                                 "params": r["params"], "train_sec": 0, **abl})
                    print(f"           ablation {mode:11s} overall {abl['overall']*100:.1f}% "
                          f"post_sw {abl['post_sw']*100:.1f}% steady {abl['steady']*100:.1f}%")
                # rule-probe (paper's claim): rule decodable from the apical
                # state, largely absent from the somatic state
                with torch.no_grad():
                    _, s_states, a_states = model.forward_states(eval_eps[0],
                                                                 eval_eps[1])
                p_ap = rule_probe(a_states, eval_eps[3])
                p_so = rule_probe(s_states, eval_eps[3])
                probes.append({"seed": seed, "apical": p_ap, "soma": p_so})
                print(f"           rule probe: apical {p_ap*100:.1f}%  soma {p_so*100:.1f}%")

    # aggregate
    from collections import defaultdict
    agg = defaultdict(lambda: {"overall": [], "post_sw": [], "steady": []})
    for r in rows:
        agg[r["model"]]["overall"].append(r["overall"])
        agg[r["model"]]["post_sw"].append(r["post_sw"])
        agg[r["model"]]["steady"].append(r["steady"])
    summary = {}
    for m, d in agg.items():
        summary[m] = {k: float(np.mean(v)) for k, v in d.items()}
        print(f"\n{m:16s} mean overall {summary[m]['overall']*100:.1f}%  "
              f"post_sw {summary[m]['post_sw']*100:.1f}%  steady {summary[m]['steady']*100:.1f}%")

    # verdict: dendritic post-switch > best point-neuron baseline by >=5 pts
    d = summary["dendritic"]["post_sw"]
    base = max(summary["single_ltc"]["post_sw"], summary["gru"]["post_sw"])
    sst = summary["dendritic_sst"]["post_sw"]
    verdict = {
        "dendritic_post_sw": d, "best_baseline_post_sw": base,
        "gain_pts": (d - base) * 100,
        "sst_drop_pts": (d - sst) * 100,
        "pass": bool((d - base) >= 0.05 and (d - sst) >= 0.05),
    }
    print(f"\nDendritic vs best baseline (post-switch): "
          f"{(d-base)*100:+.1f} pts; SST ablation drop: {(d-sst)*100:.1f} pts")
    print("VERDICT:", "PASS" if verdict["pass"] else "FAIL")

    if probes:
        probes_mean = {k: float(np.mean([p[k] for p in probes]))
                       for k in ("apical", "soma")}
        print(f"rule probe mean: apical {probes_mean['apical']*100:.1f}%  "
              f"soma {probes_mean['soma']*100:.1f}%")
    else:
        probes_mean = {}
    out = {"config": vars(args), "rows": rows, "summary": summary,
           "probes": probes, "probes_mean": probes_mean,
           "verdict": verdict}
    (HERE / "report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = ["# RSFC — dendritic vs point-neuron", ""]
    lines.append("| model | overall | post-switch | steady |")
    lines.append("|---|---|---|---|")
    for m, s in summary.items():
        lines.append(f"| {m} | {s['overall']*100:.1f}% | {s['post_sw']*100:.1f}% | "
                     f"{s['steady']*100:.1f}% |")
    lines += ["", f"Verdict: {'PASS' if verdict['pass'] else 'FAIL'} "
              f"(dendritic vs best baseline {verdict['gain_pts']:+.1f} pts; "
              f"SST drop {verdict['sst_drop_pts']:.1f} pts)"]
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
