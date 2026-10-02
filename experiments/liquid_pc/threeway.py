"""experiments/liquid_pc/threeway.py — A->B->C 三对照回放：
真实回放 vs 生成式回放 vs 无回放（相同容量/数据/预算，预注册指标）。

RESEARCH_RECOVERY.md 登记的下一实验："新实验同时包含真实回放、生成式回放、
无回放对照；相同容量/数据/预算。主指标预注册最差任务、平均遗忘、新任务学习
质量；报告逐 seed 和失败率。" 本件兑现它：

- none：顺序训练，无任何回放（灾难遗忘上界）
- real：真实储备池回放（旧任务真实样本，buf=16，随阶段增长 A→A+B）
- dream：生成式回放（模型梦 buffer，scholar 协议，同 abc_continual）

三模式 × PC/GRU × 5 seeds，同 buffer 容量、同 batch、同 epochs。
主指标：worst_final（全系统是否仍可用）、mean_forget、c_final（新任务质量）。
逐 seed 全存 JSON。

Run:  python -m experiments.liquid_pc.threeway
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Dict, List

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.liquid_pc.data import train_test_split, three_regimes
from experiments.liquid_pc.model import PCLiquidCore, GRUBaseline
from experiments.liquid_pc.run import train, one_step_mse, generate_synthetic

KINDS = ["PC", "GRU"]
MODES = ["none", "real", "dream"]


def make_model(kind: str, d_in: int, seed: int):
    torch.manual_seed(seed)
    if kind == "PC":
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    return GRUBaseline(d_in, hidden=70)


def real_buffer(*parts: torch.Tensor, buf: int, seed: int) -> torch.Tensor:
    """真实储备池：从旧任务训练集各取一半，拼到 buf 条。"""
    g = torch.Generator().manual_seed(seed)
    out: List[torch.Tensor] = []
    per = max(1, buf // len(parts))
    for x in parts:
        idx = torch.randperm(len(x), generator=g)[:min(per, len(x))]
        out.append(x[idx])
    return torch.cat(out, dim=0)


def main(
    *,
    seeds=(0, 1, 2, 3, 4),
    buf: int = 16,
    n_seq: int = 320,
    seq_len: int = 96,
    d_in: int = 1,
    epochs: int = 40,
    batch: int = 64,
    lr: float = 3e-3,
    replay_batch: int = 16,
    warmup: int = 16,
) -> Dict:
    t0 = time.time()
    rec = {(mode, kind): {"forget_A": [], "forget_B": [], "mean_forget": [],
                          "a_final": [], "b_final": [], "c_final": [],
                          "worst_final": []}
           for mode in MODES for kind in KINDS}

    for seed in seeds:
        a, b, c = three_regimes(n_seq, seq_len, d_in, seed=seed)
        a_tr, a_te = train_test_split(a, 0.8)
        b_tr, b_te = train_test_split(b, 0.8)
        c_tr, c_te = train_test_split(c, 0.8)

        for kind in KINDS:
            for mode in MODES:
                m = make_model(kind, d_in, seed)

                # --- task A ------------------------------------------------
                train(m, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
                a_after_A = one_step_mse(m, a_te)
                d1 = None
                if mode == "dream":
                    d1 = generate_synthetic(m, n_syn=buf, seq_len=seq_len,
                                            d_in=d_in, seed=seed + 7,
                                            warmup=warmup)
                elif mode == "real":
                    d1 = real_buffer(a_tr, buf=buf, seed=seed + 7)

                # --- task B ------------------------------------------------
                train(m, b_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 1,
                      replay_x=d1, replay_batch=min(replay_batch, buf))
                b_after_B = one_step_mse(m, b_te)
                d2 = None
                if mode == "dream":
                    d2 = generate_synthetic(m, n_syn=buf, seq_len=seq_len,
                                            d_in=d_in, seed=seed + 17,
                                            warmup=warmup)
                elif mode == "real":
                    d2 = real_buffer(a_tr, b_tr, buf=buf, seed=seed + 17)

                # --- task C ------------------------------------------------
                train(m, c_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 2,
                      replay_x=d2, replay_batch=min(replay_batch, buf))

                a_final = one_step_mse(m, a_te)
                b_final = one_step_mse(m, b_te)
                c_final = one_step_mse(m, c_te)
                fA = a_final - a_after_A
                fB = b_final - b_after_B
                worst = max(a_final, b_final, c_final)

                r = rec[(mode, kind)]
                r["forget_A"].append(fA)
                r["forget_B"].append(fB)
                r["mean_forget"].append(0.5 * (fA + fB))
                r["a_final"].append(a_final)
                r["b_final"].append(b_final)
                r["c_final"].append(c_final)
                r["worst_final"].append(worst)

    # ---- 汇总（mean ± std，逐 seed 全存）--------------------------------
    summary: Dict = {}
    for (mode, kind), r in rec.items():
        summary[f"{mode}/{kind}"] = {
            "worst_final": f"{statistics.fmean(r['worst_final']):.4f} ± "
                           f"{statistics.pstdev(r['worst_final']):.4f}",
            "mean_forget": f"{statistics.fmean(r['mean_forget']):.4f} ± "
                           f"{statistics.pstdev(r['mean_forget']):.4f}",
            "c_final": f"{statistics.fmean(r['c_final']):.4f} ± "
                       f"{statistics.pstdev(r['c_final']):.4f}",
            "worst_seeds": [round(v, 4) for v in r["worst_final"]],
        }

    # 三对照对比（同模型内 mode 间最差任务差）
    for kind in KINDS:
        wn = statistics.fmean(rec[("none", kind)]["worst_final"])
        wr = statistics.fmean(rec[("real", kind)]["worst_final"])
        wd = statistics.fmean(rec[("dream", kind)]["worst_final"])
        summary[f"contrast/{kind}"] = {
            "none_minus_real": round(wn - wr, 4),
            "none_minus_dream": round(wn - wd, 4),
            "dream_minus_real": round(wd - wr, 4),
        }

    out = {"protocol": "threeway", "seeds": list(seeds), "buf": buf,
           "summary": summary, "seconds": round(time.time() - t0, 1)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "report_threeway.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nreport -> {path}  ({out['seconds']}s)")
    return out


if __name__ == "__main__":
    main()
