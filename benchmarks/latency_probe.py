"""Latency/memory probe: pure-liquid vs hybrid vs transformer (p0c-latency).

端侧延迟轴最小探针(轮 363 换轴推导;零训练,纯推理):
三种配置同宽(d_model=104):①纯液体核 MT-LNN(attention_layers=(),
O(1) 状态,O(T) 计算)②混合 MT-LNN(默认注意力层,轮 362/363 描述性
证据=超线性)③transformer(O(T²) 注意力)。T 扫描 {512, 2048, 8192,
16384},batch=1(端侧单请求场景),每点预热 1 次+测 3 次取中位。

假设(预注册 benchmarks/verdicts/p0c_latency.prereg.json):纯液体核
延迟增长 ~线性(16× token → ~16× 延迟),transformer ~二次(→ ~256×
或 OOM)⇒ 长流斜率分离。MPS OOM=记录结局(端侧内存墙)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.baselines import BaselineConfig, ModernCausalTransformer
from benchmarks.reasoning_depth import _append_jsonl
from mt_lnn import MTLNNConfig, MTLNNModel

D_MODEL = 104


def build(kind: str, vocab: int, seq_len: int, device):
    torch.manual_seed(0)
    if kind == "pure_liquid":
        cfg = MTLNNConfig(vocab_size=vocab, max_seq_len=seq_len,
                          d_model=D_MODEL, n_layers=2, n_heads=4,
                          n_kv_heads=4, d_head=26, dropout=0.0,
                          attention_dropout=0.0, gwtb_n_heads=1,
                          core_iterations=2, attention_layers=(),
                          use_global_coherence=False)
        m = MTLNNModel(cfg)
    elif kind == "hybrid":
        cfg = MTLNNConfig(vocab_size=vocab, max_seq_len=seq_len,
                          d_model=D_MODEL, n_layers=2, n_heads=4,
                          n_kv_heads=4, d_head=26, dropout=0.0,
                          attention_dropout=0.0, gwtb_n_heads=1,
                          core_iterations=2)
        m = MTLNNModel(cfg)
    else:
        cfg = BaselineConfig(vocab_size=vocab, max_seq_len=seq_len,
                             d_model=D_MODEL, n_layers=2, n_heads=4,
                             d_ff=256)
        m = ModernCausalTransformer(cfg)
    return m.to(device).eval()


@torch.no_grad()
def measure(model, T, vocab, device, reps=3):
    """batch=1 随机 token 前向,预热 1+测 3 取中位;返回 (秒, 峰值MB)。"""
    ids = torch.randint(0, vocab, (1, T), device=device)
    model(ids)  # warmup
    if device.type == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.time()
        model(ids)
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.time() - t0)
    mb = torch.mps.current_allocated_memory() / 1e6 if device.type == "mps" \
        else 0.0
    return float(np.median(times)), round(mb, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[512, 2048, 8192, 16384])
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default="p0c_latency")
    p.add_argument("--results_dir", default=None)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        dev = torch.device("cpu")
        m = build("pure_liquid", 64, 64, dev)
        ids = torch.randint(0, 64, (1, 64), device=dev)
        assert m(ids)["logits"].shape == (1, 64, 64)
        print("latency_probe selftest OK")
        return
    device = torch.device("mps" if args.device in ("auto", "mps")
                          and torch.backends.mps.is_available() else "cpu")
    rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    results = os.path.join(args.results_dir or rd, "latency_probe.jsonl")
    vocab = 64
    for kind in ["pure_liquid", "hybrid", "transformer"]:
        for T in args.lengths:
            try:
                m = build(kind, vocab, T, device)
                sec, mb = measure(m, T, vocab, device)
                row = {"mode": "latency", "config": kind, "T": T,
                       "fwd_s_median": round(sec, 4), "alloc_mb": mb,
                       "oom": False, "tag": args.tag,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                print(f"  {kind:12s} T={T:6d}: {sec:.4f}s  alloc={mb:.0f}MB",
                      flush=True)
                del m
                if device.type == "mps":
                    torch.mps.empty_cache()
            except (RuntimeError, MemoryError) as e:
                row = {"mode": "latency", "config": kind, "T": T,
                       "fwd_s_median": None, "alloc_mb": None, "oom": True,
                       "err": str(e)[:200], "tag": args.tag,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                print(f"  {kind:12s} T={T:6d}: OOM/{type(e).__name__}",
                      flush=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
            _append_jsonl(results, row)


if __name__ == "__main__":
    main()
