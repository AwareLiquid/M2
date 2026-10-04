"""experiments/liquid_pc/order_sensitivity.py — 任务顺序敏感性（RESEARCH_RECOVERY
登记的第 4 条：扩展测试任务顺序敏感性）。

三任务的 6 种排列 × PC/GRU × 3 seeds，生成式回放（scholar 协议，同 abc_continual），
检验"PC 的梦优势"是否顺序稳健（预注册主指标：worst_final；顺序稳健 = 6 种排列的
PC−GRU 差值全部同号）。

Run:  python -m experiments.liquid_pc.order_sensitivity
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.liquid_pc.data import train_test_split, three_regimes
from experiments.liquid_pc.model import PCLiquidCore, GRUBaseline
from experiments.liquid_pc.run import train, one_step_mse, generate_synthetic

KINDS = ["PC", "GRU"]
ORDERS = list(itertools.permutations(range(3)))   # 6 permutations


def make_model(kind: str, d_in: int, seed: int):
    torch.manual_seed(seed)
    if kind == "PC":
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    return GRUBaseline(d_in, hidden=70)


def run_order(perm, seed: int, kind: str, n_seq: int = 320, seq_len: int = 96,
              epochs: int = 40, batch: int = 64, lr: float = 3e-3,
              buf: int = 16, replay_batch: int = 16,
              warmup: int = 16) -> float:
    """按给定任务顺序跑 scholar 协议，返回 worst_final。"""
    regimes = three_regimes(n_seq, seq_len, 1, seed=seed)
    splits = [train_test_split(r, 0.8) for r in regimes]
    m = make_model(kind, 1, seed)
    prev_final = []
    after = []
    dbuf = None
    for stage, idx in enumerate(perm):
        tr, te = splits[idx]
        train(m, tr, epochs=epochs, batch=batch, lr=lr, seed=seed + stage,
              replay_x=dbuf, replay_batch=min(replay_batch, buf))
        after.append(one_step_mse(m, te))
        dbuf = generate_synthetic(m, n_syn=buf, seq_len=seq_len, d_in=1,
                                  seed=seed + 7 * stage, warmup=warmup)
    # forgetting per task: final - right-after-learned (last task has none)
    finals = [one_step_mse(m, splits[i][1]) for i in range(3)]
    worst = max(finals)
    return worst


def main():
    t0 = time.time()
    seeds = (0, 1, 2)
    rec = {kind: {} for kind in KINDS}
    for perm in ORDERS:
        key = "".join(str(i) for i in perm)
        for kind in KINDS:
            vals = [run_order(perm, seed, kind) for seed in seeds]
            rec[kind][key] = {"mean": statistics.fmean(vals),
                              "vals": [round(v, 4) for v in vals]}
            print(f"  order {key} {kind}: worst {rec[kind][key]['mean']:.4f} "
                  f"(seeds {rec[kind][key]['vals']})", flush=True)

    # 顺序稳健性：每个排列的 PC-GRU 差值
    diffs = {}
    for perm in ORDERS:
        key = "".join(str(i) for i in perm)
        diffs[key] = round(rec["PC"][key]["mean"] - rec["GRU"][key]["mean"], 4)
    signs = {k: (v < 0) for k, v in diffs.items()}
    robust = all(signs.values()) or not any(signs.values())
    print(f"\nPC-GRU (负 = PC 更好): {diffs}")
    print(f"同号: {robust}  ->  VERDICT: "
          f"{'PASS (顺序稳健)' if robust else 'FAIL (顺序敏感)'}")

    out = {"protocol": "order_sensitivity", "seeds": list(seeds),
           "orders": rec, "pc_minus_gru": diffs, "robust": robust,
           "seconds": round(time.time() - t0, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "report_order_sensitivity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"report -> {path}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
