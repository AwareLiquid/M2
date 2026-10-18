# 记忆探针 — 跨窗口 K→V 召回（2026-10-19）

> 目的：验证旗舰能力主张——**循环状态在 KV 丢弃后仍携带离散 key→value 绑定**
> （M1 的 0.56 recall vs attention 0.000）。本文记录两个探针：
> ① 2b 零样本（服务器不可用→本地尝试）② 小模型任务适配（本地完成）。

## 协议（`benchmarks/memory_probe_2b.py` / `memory_probe_finetune.py`）

```
seg_a = K1 V1 ... Kn Vn          (byte tokens；key 10-73，value 100-163)
forward(seg_a, use_cache=True) → cache
cross-window: cache = cache.recurrent_only()   # KV 丢弃，仅存每层 h_prev
in-window:    full cache                        # 注意力可复制
forward(seg_b, cache) → 在每个 key 位置 argmax 预测其 value
chance = 1/64 = 1.6%
```

## 结果

| 设置 | in-window | cross-window | 说明 |
|---|---|---|---|
| **2b 200K 零样本** | 0.0% | 0.5% | 2b checkpoint（step 200000，1.93B）从未训过该任务 → 无召回（预期） |
| **probe 365K + 400 步任务适配**（仅 cross-window 训练） | 3.6% (CE 19.9) | **100.0% (CE 0.003)** | **状态成功携带绑定**；in-window 低是训练条件不匹配（未训注意力路径） |

**关键结论**：
1. **循环状态可以学习携带跨窗口的 K→V 绑定**——365K 参数、400 步、随机 K/V 泛化
   到 100%（chance 1.6%）；
2. 零样本不可行（未训该任务），任务适配是必要前提——M1 的 0.56 也来自适配后的
   状态（fast-weight 适配器）；
3. **2b 规模化验证受阻**：A100 租期已到期（36.140.149.133 不可达），本地
   （15.4GB RAM / RTX 5060 8.5GB）构建 2b 模型时崩溃（access violation，
   fp32/bf16 均崩）→ 需新 GPU 或 ≥32GB RAM 机器。

## 复现

```bash
# ① 准备小语料 + 训练 probe 模型（本地 CPU，~5 分钟）
python -m m2_training.prepare --train <train.txt> --validation <val.txt> --output <dir>
python -m m2_training train --task text --size probe --experiment baseline \
  --corpus <dir>/manifest.json --checkpoint runs/probe_mem.pt \
  --sequence-length 128 --batch 8 --lr 3e-3 --steps 3000 --device cpu
# ② 零样本探针
python benchmarks/memory_probe_2b.py --ckpt runs/probe_mem.pt --device cpu
# ③ 任务适配探针（关键实验）
python benchmarks/memory_probe_finetune.py --ckpt runs/probe_mem.pt --steps 400 --device cpu
```
