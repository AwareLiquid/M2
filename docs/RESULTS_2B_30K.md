# 2b 训练结果 — 30K 步完成（2026-09-16~17，A100 40GB）

> 这是 M2 的 2b preset（2080×34×16/4，2.06B 参数）首次完整 30K 步训练。
> 可行性先例：2026-09-10 的 10K 步验证（val PPL 4.53，见 M1 backlog "2b 拆分训练可行性"）。

## 配置

| 项 | 值 |
|---|---|
| 模型 | 2b preset：2080 宽 / 34 层 / 16 头 / 4 KV（2,064,984,584 参数 @ vocab 32768） |
| 语料 | WikiText-103 **byte-level**（vocab 256，UTF-8 字节 token） |
| 序列/批 | seq 128，batch 8 × grad_accum 4（有效 32） |
| 优化 | AdamW8bit（β 0.9/0.95，wd 0.01），lr 1e-4 |
| 精度 | bf16 模型 + bf16 checkpoint（**无 optimizer state，不可精确 resume**——已知限制） |
| 通道 | A100-PCIE-40GB（36.140.149.133），训练过程三次 GPU 挤兑 OOM 中断，checkpoint 纪律（每 500 步原子保存）保住全部进度 |

## 结果

| 里程碑 | val PPL |
|---|---|
| 10K 步（09-10 可行性） | 4.53 |
| **30K 步（本次完成）** | **2.73** |
| 末步 loss | 1.00 |

## 备注

- **checkpoint**：服务器 `/root/M2/runs/t2b_30k.pt`（4.1GB bf16 参数，无 optimizer state）
- **诚实边界**：byte-level PPL 2.73 ≈ BPB 1.45——这是 2b 规模在 byte 语料上的收敛值，
  不是与 gpt2-vocab 模型直接可比的数值；不做跨 vocab 的 PPL 对比宣传
- **下一步候选**：① 更长预算（60K+ 步）；② 记忆探针评估（cross-window recall @2b 规模，
  把 M1 已验证的记忆优势在 2b 规模复测）；③ checkpoint 下载归档（4.1GB，磁盘有限）
