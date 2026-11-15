"""Streaming key-value recall: length-extrapolation probe (p0c-stream-len).

长流式记忆轴最小探针(轮 361 换轴推导,四栏见 DISTILL/GOALS 队列):
32 条 (key → value) 事实以 [K k][V v] 对的形式埋进含噪声填充的流里
(键互异,事实数固定);流长由填充密度变化。模型在 T=256 上训练,
在更长的流上评估回忆准确率(查询流末尾 [Q k_q],答案位预测 v_q)。

长流 = 回忆距离变长(同一批事实,更稀疏的呈现):
  T=256  → 填充间隔 ~6
  T=1024 → 填充间隔 ~30
  T=4096 → 填充间隔 ~127

对照:MT-LNN(液体核,O(1) 递归状态)vs ModernCausalTransformer
(O(T²) 全注意力)。两模型同 max_seq_len(4096,RoPE 不作弊)。
同时透明报告各长度的前向墙钟(延迟平坦性,描述性指标不入判据——
测量噪声大,铁律"合轮收尾显式判定退出码"要求数字稳定)。

自测:_selftest 含黄金回放(从 token 行重建 KV 表,核验答案)、
布局不变式、确定性。结果追加 benchmarks/results/streaming_recall.jsonl。
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
from benchmarks.reasoning_depth import _append_jsonl, train_model
from mt_lnn import MTLNNConfig, MTLNNModel

# ── vocab ────────────────────────────────────────────────────────────────────
BOS, K_MARK, V_MARK, Q_MARK = range(4)
N_SPECIAL = 4
N_KEYS, N_VALS, N_NOISE = 32, 16, 8
KEY_BASE = N_SPECIAL                    # keys: KEY_BASE .. KEY_BASE+31
VAL_BASE = KEY_BASE + N_KEYS            # values: VAL_BASE .. VAL_BASE+15
NOISE_BASE = VAL_BASE + N_VALS          # noise: NOISE_BASE .. NOISE_BASE+7
VOCAB = NOISE_BASE + N_NOISE            # 60

# 事实数固定(键互异 ⇒ 无歧义回忆);流长只由填充密度变化
N_FACTS = 32
TRAIN_LEN = 256


def _fillers(rng: np.random.Generator, n: int) -> np.ndarray:
    return NOISE_BASE + rng.integers(0, N_NOISE, size=n)


def gen_stream(batch: int, T: int, rng: np.random.Generator,
               n_facts: int = N_FACTS):
    """Returns (tokens (B,T) int64, answer (B,) value ids, ans_pos).

    布局: [BOS] {[K k][V v] filler*}* [Q k_q] ANS
    键互异(n_facts 条事实全部在流中);查询键必在流内。
    e8 易变体(轮 363): n_facts=8(回忆容量降,距离测试保留)。
    """
    pair_tok = 4            # [K k][V v] = 4 token
    budget = T - 1 - 3      # 减 BOS(1) 和 [Q k_q](2)([ANS] 占 T-1 位)
    assert budget >= n_facts * pair_tok, f"T={T} 装不下 {n_facts} 条事实"
    assert n_facts <= N_KEYS, f"n_facts {n_facts} > 键池 {N_KEYS}"
    slack = budget - n_facts * pair_tok          # 全部分配给填充
    gaps = n_facts + 1                           # 每事实后一个 gap + 末尾一个
    per = rng.integers(0, slack // gaps + 1, size=gaps)
    per[-1] += slack - per.sum()                 # 余数全给最后一个 gap(长流=长尾填充)

    toks = np.full((batch, T), PAD := 0, dtype=np.int64)
    ans = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        keys = rng.permutation(N_KEYS)[:n_facts]
        vals = rng.integers(0, N_VALS, size=n_facts)
        q = int(rng.integers(0, n_facts))         # 查询第 q 条事实
        row = [BOS]
        for i in range(n_facts):
            row += [K_MARK, KEY_BASE + int(keys[i]),
                    V_MARK, VAL_BASE + int(vals[i])]
            row += _fillers(rng, int(per[i])).tolist()
        row += _fillers(rng, int(per[-1])).tolist()   # 末 gap:最后事实→查询的回忆距离
        k_q = KEY_BASE + int(keys[q])
        v_q = VAL_BASE + int(vals[q])
        row += [Q_MARK, k_q]
        assert len(row) == T - 1, f"layout {len(row)} != {T-1}"
        toks[b] = row + [0]                        # [ANS] 占位 0(损失只看 ans_pos-1 → 上一位是 k_q)
        ans[b] = v_q
    return toks, ans, T - 1


def make_batch(tokens, answer, ans_pos, device):
    ids = torch.from_numpy(tokens).to(device)
    labels = torch.full_like(ids, -100)
    labels[:, ans_pos] = torch.from_numpy(answer).to(device)
    return ids, labels, ans_pos


@torch.no_grad()
def eval_len(model, T, device, rng, batches=4, batch=16, timed=False,
             n_facts=N_FACTS):
    """准确率 + (可选) 前向墙钟(描述性)。

    轮 362 教训: T=4096 时 global_coherence 稀疏分数张量 MPS OOM
    (4GB 峰值)——长 T 自适应减 batch + 长度间清缓存。
    """
    if T >= 4096:
        batch = min(batch, 4)
    if device.type == "mps":
        torch.mps.empty_cache()
    model.eval()
    correct = total = 0
    wall = 0.0
    for _ in range(batches):
        toks, ans, pos = gen_stream(batch, T, rng, n_facts=n_facts)
        ids, labels, ans_pos = make_batch(toks, ans, pos, device)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        logits = model(ids)["logits"]
        if device.type == "mps":
            torch.mps.synchronize()
        wall += time.time() - t0
        pred = logits[:, ans_pos - 1, :].argmax(-1)
        correct += (pred == labels[:, ans_pos]).sum().item()
        total += batch
    model.train()
    acc = correct / total
    return (acc, wall / batches) if timed else acc


def build(model_kind, vocab, seq_len, seed, device):
    if model_kind == "mtlnn":
        torch.manual_seed(seed)
        cfg = MTLNNConfig(vocab_size=vocab, max_seq_len=seq_len, d_model=104,
                          n_layers=2, n_heads=4, n_kv_heads=4, d_head=26,
                          dropout=0.0, attention_dropout=0.0, gwtb_n_heads=1,
                          core_iterations=2)
        m = MTLNNModel(cfg)
    else:
        torch.manual_seed(seed)
        cfg = BaselineConfig(vocab_size=vocab, max_seq_len=seq_len,
                             d_model=104, n_layers=2, n_heads=4, d_ff=256)
        m = ModernCausalTransformer(cfg)
    return m.to(device)


def run(model_kind, seeds, steps, train_len, eval_lens, device,
        tag, results_path, n_facts=N_FACTS):
    from types import SimpleNamespace
    vocab, seq_cap = VOCAB, max(eval_lens)
    print(f"== streaming_recall {model_kind} train_len={train_len} "
          f"n_facts={n_facts} eval={eval_lens} device={device} ==",
          flush=True)

    def gen_train(b, r):
        toks, ans, pos = gen_stream(b, train_len, r, n_facts=n_facts)
        return SimpleNamespace(tokens=toks, answer=ans, ans_pos=pos)

    for seed in seeds:
        t0 = time.time()
        rng = np.random.default_rng(1000 + seed)
        m = build(model_kind, vocab, seq_cap, seed, device)
        train_model(m, gen_train, device, steps, 16, 3e-4, seed)
        acc, lat = {}, {}
        for T in eval_lens:
            a, w = eval_len(m, T, device, np.random.default_rng(10_000 + seed),
                            timed=True, n_facts=n_facts)
            acc[T], lat[T] = round(a, 4), round(w, 4)
            print(f"  [seed {seed}] T={T}: acc={a:.4f} fwd={w:.3f}s",
                  flush=True)
        _append_jsonl(results_path, {
            "mode": "stream_len", "model": model_kind, "seed": seed,
            "steps": steps, "train_len": train_len, "n_facts": n_facts,
            "eval_lens": eval_lens,
            "acc_by_len": acc, "fwd_s_by_len": lat,
            "params": m.get_num_params(),
            "wall_s": round(time.time() - t0, 1), "tag": tag,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        del m


def _selftest():
    rng = np.random.default_rng(0)
    toks, ans, pos = gen_stream(16, 256, rng)
    assert toks.shape == (16, 256) and toks[:, 0].min() >= BOS
    assert (toks[:, -1] == 0).all(), "答案位占位须为 PAD"
    for r in range(16):
        row = toks[r]
        kv = {}
        i = 1
        while i < len(row) - 2:
            if row[i] == K_MARK:
                kv[int(row[i + 1])] = int(row[i + 3])
                i += 4
            elif row[i] == Q_MARK:
                break
            else:
                i += 1
        k_q = int(row[i + 1])
        assert k_q in kv, "查询键必须在流内"
        assert len(kv) == N_FACTS, "键须互异且全部出现"
        assert ans[r] == kv[k_q], "黄金回放不匹配"
        assert all(KEY_BASE <= t < KEY_BASE + N_KEYS
                   for t in row[i + 1:i + 2])
    for T in (1024, 4096):
        t2, a2, p2 = gen_stream(4, T, np.random.default_rng(1))
        assert t2.shape[1] == T
    r1 = gen_stream(4, 256, np.random.default_rng(5))
    r2 = gen_stream(4, 256, np.random.default_rng(5))
    assert (r1[0] == r2[0]).all() and (r1[1] == r2[1]).all()
    print("streaming_recall selftest OK")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["mtlnn", "transformer"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--train_len", type=int, default=TRAIN_LEN)
    p.add_argument("--eval_lens", type=int, nargs="+",
                   default=[256, 1024, 4096])
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default="p0c_stream_len")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--results_dir", default=None)
    p.add_argument("--n_facts", type=int, default=N_FACTS)
    args = p.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    results = os.path.join(args.results_dir or rd, "streaming_recall.jsonl")
    run(args.model, args.seeds, args.steps, args.train_len,
        args.eval_lens, device, args.tag, results, n_facts=args.n_facts)


if __name__ == "__main__":
    main()
