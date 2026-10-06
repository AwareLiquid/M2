# 2B SFT — Run 1 (2026-10-06, first post-training pass)

**配置**：基座 = `t2b_30k.pt`（byte-level 34L×d2080, 200K 步, val PPL 2.4106）；
SFT = `m2_training` resume + `--task sft` 切换，语料 = alpaca-cleaned 过滤
≤128 字节（5,435 train / 111 val），3 epochs = 2,040 步，batch 8, lr 1e-4
（基座 recipe），bf16 + AdamW8bit，40G A100，~83 分钟。

## 训练读数

| 指标 | 值 |
|---|---|
| train loss | 0.024 → 0.035 → 0.019（3 个 save 点） |
| **val PPL（SFT 中）** | **3.947**（SFT 前同口径 = 2.531） |

**train loss ~0.02 = 对 5.4k 条明显记忆化**（3 epochs × lr 1e-4 偏大）。

## Held-out 生成对照（贪心解码, 8 条短指令）

| 指令 | base（200K） | SFT |
|---|---|---|
| Compute the sum of 9 and 5. | 重复循环 | **The sum of 9 and 5 is 14.** ✓ |
| What is the capital of France? | "wwww , which was a series..." | **The capital of France is Paris.** ✓ |
| Convert 1 kilometer to meters. | 维基腔续写 | **1 kilometer is equivalent to 1000 meters** ✓ |
| Rewrite past tense: I eat an apple. | "( 1997 ) ." | **I ate an apple.** ✓ |
| Translate 'good morning' to Spanish. | 乱 | The good morning.（格式对、内容错） |
| Give me a synonym for happy. | 乱 | Sadness for happiness...（错） |
| What color is the sky...? | 乱 | The sky on a clear day is beautiful day.（半通） |
| Name the largest planet... | "v" | The largest planet in our solar system i...（截断，方向对） |

**结论**：SFT 前后 = **0/8 → 4/8 正确、格式合规 8/8**。指令跟随质变；
val PPL 的"恶化"是指标错配（base 的 PPL 反映通用 LM 能力、SFT 后分布变尖 +
小数据记忆化），不代表质量回退。

## 下一轮（Run 2）准备

1. **语料升级**：alpaca ≤512 字节（19,600 train / 400 val，已建好
   `corpus512`）——数据量 3.6×、响应中位 92B；
2. **lr 降档**：1e-4 → 2e-5（典型 SFT 区间）；epochs 3 → 1-2；
3. **seq 512 + rope_scale=4**：需给 resume 增加 `--sequence-length`/
   `--rope-scale` 覆盖（基座 rope_scale=1.0, seq128；YaRN 外推路径已在
   M2 embedding 实现）；
4. 评估口径：held-out 生成正确率（人工/规则）+ 选择题式评测，PPL 仅记录。

产物：`/dev/shm/t2b_sft_v1.pt`（3.86GB, bf16; 已回拉本地备份）。
