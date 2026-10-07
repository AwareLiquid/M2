# M2 handoff

Owner: Everest.

2026-09-09：新增 `mt_lnn/research/rl/dpo_grpo.py` —— DPO / GRPO 对齐损失参考实现
（标准公式，数学经 8 个合成测试验证；**未在真实训练验证**，仅供未来 2B 线 RL
阶段使用，默认不接入任何训练路径）。全仓测试 185 passed（177+8），无回归。

2026-09-09：先读 [历史候选与训练准入](docs/RESEARCH_RECOVERY.md)。
融合式 PCLiquidCore 已从 M1 迁移到 `experiments/liquid_pc/`（保留 M1 原始副本），
数值等价验证通过（nchain smoke：PC 最差任务 0.149 vs GRU 0.502，防遗忘优势复现）。
完整 nchain（5 seeds×5 tasks）与 genreplay（生成式重放对照）在服务器后台运行中。
2026-09-09 训练器修复：`batch` 默认 8→128（小批量梯度噪声是 pointer_chase
学不会的根因，batch=128 后 accuracy 0.09→0.71）、`weight_decay=0.01`（对齐 M1）、
新增 `medium` 规模（416 宽 8 层 17.7M）。完整测试 177 passed。

Local checkout: `E:\AwareLiquid\M2`.
Remote: https://github.com/AwareLiquid/M2.

## Immediate work

- Independent training entry point: `python -m m2_training`.
- Train/checkpoint/resume/evaluate verified; five configurations pass exact CPU resume tests.
- GPU selective-state train and cross-process resume/evaluation verified.
- Unwired modules and the remaining physical source split are listed in `docs/TRAINING.md`.
- Ten independent research implementations now live in `mt_lnn/research/`; old import paths alias them.
- Text corpus preparation, SHA-256 identity checks, gradient accumulation and text PPL evaluation are implemented.
- The `2b` preset counts 2,064,984,584 parameters at vocab 32768; only meta-device sizing was performed, not full-scale training.
- CPU text resume with accumulation is exact; CUDA text train/resume/evaluate was exercised at probe size.

1. Experimental sources and their shared runtime are imported; provenance is pinned in `docs/MIGRATION.md`.
2. Direct tests pass (156); the reasoning-depth GPU smoke completes training and evaluation.
3. Keep future research changes here; graduate validated mechanisms to M1 through reviewed changes.
4. M1 PR #46 is closed. Its compatibility snapshot supplies this initial migration.

## Preservation

The former document-QA checkout is now `E:\AwareLiquid\RAG`, remote
`AwareLiquid/AwareLiquid-RAG`. Its untracked `local_verifier_server.py` was
preserved during the directory rename. Do not copy it or QA scores into this model repository.

Experimental code, direct tests, reasoning benchmark and research documents
have been imported. See `docs/MIGRATION.md` for the exact source revision.
The imported suite passed 156 tests locally. No new capability claim follows
from this migration. M1 compatibility code remains in its source repository.
2026-09-11：新增 [数据形态规范](docs/DATA_FORMS.md) —— 数据形态必须匹配训练目标
(分线匹配表/情节流格式/反模式清单)。核心结论: 世界模型线与类脑评估线改用
情节流(观测-动作-时间戳), 语言主线语料不动。

2026-10-06：**2B 后训练管线打通（SFT Run 1）**。`m2_training` 新增 SFT 任务
（JSONL→byte 级记录 + 响应掩码 + sha256 校验/恢复语义, 7+1 tests）；resume 支持
`--task/--corpus` 覆盖（text 基座 → SFT 后训练入口）。README 级结论见
[docs/SFT_RUN1.md](docs/SFT_RUN1.md)：held-out 生成 **0/8 → 4/8 正确、格式
8/8**（指令跟随质变）；train loss 0.02 = 5.4k 条 3 epoch 过拟合；val PPL
2.53→3.95 为指标错配。Run 2 准备：19.6k 条 seq-512 语料（corpus512）+ lr 2e-5
+ `--sequence-length/--rope-scale` 覆盖（待加）。
**同批抢救**：训练机上仅存未入库的 2B 基建（bf16+bitsandbytes AdamW8bit、
rope_scale/YaRN 频表、resume corpus 覆盖）已入库 → 分支 `feat/2b-sft`
（设备条件化：CUDA=bf16+8bit / CPU=fp32+AdamW, 193 tests 全绿）。

2026-10-07：**JevBench 官方基准首测（M2 模型两枚）**。经由官方 harness
（fstandhartinger/jevbench）+ 原生 /v1/systemone 服务：
- M2-2B SFT v2（2 万步）：public tiers raw **35.8%**（48 易 + 72 原题），
  ECE 0.34-0.50，p50 0.71-0.92s，schema 100%；
- 续训版 v2b（+5k 步，val PPL 3.20）：raw **31.3%**——指令数据多训不提升
  决策题（分布不同）；
- 结论：速度/成本轴已到顶（O1-Flash 决策头 16-21ms），准确率是开放问题。
  产物：`docs/JEVBENCH_RUN1.md`（M2）+ O1-Flash 仓 `docs/JEVBENCH_RESULTS.md`。
- **2B 冻结核 + 决策头**（`m2_training/decision_2b.py`，final_norm hook + readout）：
  官方 easy 39.6% / orig 34.7%、**ECE 0.20**（同批最优）、p50 0.168s；
  自家语料 val 0.50（从零训平台 0.41）。2B 表征对决策显著优于 4.5M 从零。

2026-10-07 运维：**训练机（被墙）可经 hf-mirror 直传 HuggingFace**——`HF_ENDPOINT=https://hf-mirror.com` + huggingface_hub ≥1.x（`/root/M2/.venv`
已装），上传 `t2b_sft_v2.pt` 成功（~1.6MB/s）。直连 huggingface.co 不可达
（000），hf-mirror.com 200/0.24s。基座 `t2b_30k.pt` 与 HF `t2b_200k.pt`
sha256 逐字节一致（e10af131…），无需重复传。\n
2026-10-07 晚：**决策线双训练在跑**（服务器 A100）——A) 2B+头长效
（warm-start 8k 头 +20k 步, lr 5e-4, \decision_head_2b_v2.pt\）；B) v2b(SFT)
核 A/B（fresh 头 8k 步, \decision_head_v2b_core.pt\）。工具已入库
（\decision_2b.py\/\it_temp_2b.py\/\jev_service_2b.py\）。队列待排：
决策语料 v2（CLINC150/FinancialPhraseBank/ContractNLI 真实族）、O1-Sound 数据
经镜像解锁、readout 变体、SFT v3。
