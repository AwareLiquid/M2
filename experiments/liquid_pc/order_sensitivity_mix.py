"""experiments/liquid_pc/order_sensitivity_mix.py — 顺序敏感性的替代方案。

基线结论（order_sensitivity.py）：PC 在 5/6 任务顺序更好，但顺序 201 反转
（GRU 更好）→ FAIL（顺序敏感）。反转画像：**最难任务放最后**时，PC 的梦缓冲
只保留"上一阶段"的梦 → 早期任务的遗忘未被回放覆盖。

本脚本 = 同一协议，唯一变量：**梦池改为跨任务混合（reservoir）**——每阶段
结束后把新梦追加进池，回放时从"全部历史梦"里采样（混合旧+新），而不是
只用最新一阶段的梦。标准生成式回放实践（DGR reservoir）。

判据（同基线协议）：6 种排列的 PC−GRU 差值全部同号 = 顺序稳健。

Run:  python -m experiments.liquid_pc.order_sensitivity_mix
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
from experiments.liquid_pc.order_sensitivity import make_model
from experiments.liquid_pc.run import train, one_step_mse, generate_synthetic

KINDS = ["PC", "GRU"]
ORDERS = list(itertools.permutations(range(3)))


def run_order_mix(perm, seed: int, kind: str, n_seq: int = 320,
                  seq_len: int = 96, epochs: int = 40, batch: int = 64,
                  lr: float = 3e-3, buf: int = 16, replay_batch: int = 16,
                  warmup: int = 16) -> float:
    """scholar 协议 + 跨任务混合梦池，返回 worst_final。"""
    regimes = three_regimes(n_seq, seq_len, 1, seed=seed)
    splits = [train_test_split(r, 0.8) for r in regimes]
    m = make_model(kind, 1, seed)
    pool: list[torch.Tensor] = []
    for stage, idx in enumerate(perm):
        tr, te = splits[idx]
        replay_x = torch.cat(pool, dim=0) if pool else None   # 全部历史梦
        train(m, tr, epochs=epochs, batch=batch, lr=lr, seed=seed + stage,
              replay_x=replay_x, replay_batch=replay_batch)
        dbuf = generate_synthetic(m, n_syn=buf, seq_len=seq_len, d_in=1,
                                  seed=seed + 7 * stage, warmup=warmup)
        pool.append(dbuf)
    finals = [one_step_mse(m, splits[i][1]) for i in range(3)]
    return max(finals)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of seeds per (order, kind)")
    args = ap.parse_args()
    t0 = time.time()
    seeds = tuple(range(args.seeds))
    rec = {kind: {} for kind in KINDS}
    for perm in ORDERS:
        key = "".join(str(i) for i in perm)
        for kind in KINDS:
            vals = [run_order_mix(perm, seed, kind) for seed in seeds]
            rec[kind][key] = {"mean": statistics.fmean(vals),
                              "vals": [round(v, 4) for v in vals]}
            print(f"  order {key} {kind}: worst {rec[kind][key]['mean']:.4f} "
                  f"(seeds {rec[kind][key]['vals']})", flush=True)

    diffs = {}
    for perm in ORDERS:
        key = "".join(str(i) for i in perm)
        diffs[key] = round(rec["PC"][key]["mean"] - rec["GRU"][key]["mean"], 4)
    signs = {k: (v < 0) for k, v in diffs.items()}
    robust = all(signs.values()) or not any(signs.values())
    n_pc = sum(1 for v in diffs.values() if v < 0)
    print(f"\nPC-GRU (负 = PC 更好): {diffs}")
    print(f"PC 更好的顺序数: {n_pc}/6  |  全部同号: {robust}")
    print("VERDICT:", "PASS (顺序稳健)" if robust else
          f"FAIL — 但混合梦池下 PC 占优 {n_pc}/6")

    out = {"protocol": "order_sensitivity_mix(reservoir)", "seeds": list(seeds),
           "orders": rec, "pc_minus_gru": diffs, "robust": robust,
           "pc_wins": n_pc, "seconds": round(time.time() - t0, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "report_order_sensitivity_mix.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"report -> {path}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
