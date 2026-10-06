# 2B SFT — Run 2 (2026-10-06, seq512 + rope_scale 4; **欠训练，续训中**)

**配置**：基座 `t2b_30k.pt`（200K, seq128, rope_scale 1.0）→ resume 覆盖：
`--task sft --corpus corpus512（19,600 train / 400 val, ≤512B, 响应中位 92B）
--lr 2e-5 --sequence-length 512 --rope-scale 4.0 --batch 2 --grad-accum 4`。
1 epoch = 2,450 步，~100 分钟（0.41 步/s，24GB 显存）。代码：见本仓
`m2_training`（resume 覆盖已入 main，195 tests 绿）。

## 训练读数（1 epoch）

| 指标 | Run 1（5.4k/seq128/3ep） | **Run 2（19.6k/seq512/1ep）** |
|---|---|---|
| train loss | 0.019-0.035（记忆化） | **1.025-1.604（平滑，未坍缩）** |
| val PPL | 3.947 | **3.538** |

## Held-out 生成对照（v2 vs base）

| 指令 | base | v2 |
|---|---|---|
| 9+5 | 重复循环 | "The sum of 2 and 2 is 1."（格式对、内容错） |
| France 首都 | "wwww , which was…" | "The capital of France is a capital of Fr"（退化） |
| good morning→西语 | 维基腔 | "The Spanish are Spanish are Spanish are"（退化） |
| km→m | 维基腔 | "The convert of the converted probert is"（蜕变词） |
| 过去式 I eat an apple | "( 1997 ) ." | "I eat an apple."（未变换） |
| **正确率** | 0/8 | **0/8**（格式 8/8） |

## 诊断（10-07 中期）

**v2 = 欠训练**：1 epoch × lr 2e-5 的有效更新远少于 v1（3 epochs × 1e-4）；
train loss 1.18 仍在高位（v1 已记忆化）。格式/模板学会了，内容没有。
另有两个可疑变量叠加：seq512 + rope_scale=4 的**位置语义切换**（基座在
128/scale1.0 训成）可能在短训内未适配（蜕变词提示预训练表征被扰动）。

**决定**：续训 Run 2b（同配置 +5,000 步 ≈ 3 epochs，`sft_run2b.log`）——
若 loss 持续下行且内容出现 → 确认欠训练；若平台或继续退化 → 下一轮回
seq128 原生 regime（Run 3 候选：seq128 + 更大数据 + lr 5e-5）。

产物：`/dev/shm/t2b_sft_v2.pt`（3.86GB bf16）。
