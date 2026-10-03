# M2 handoff

Owner: Everest.

2026-09-27（轮 2）：[循环体系](docs/loop/GOAL-PROMPT-M2.md) 瘦身
canonical 化（AMM-002，参照 Physic AMM-037 先例）——v2 精简版（~20 行
指针+铁律）升为唯一点火源，v1 全文降级 `GOAL-PROMPT-M2-v1-ARCHIVED.md`
（废止横幅）；承重铁律机械门禁 `tests/test_goal_prompt_invariants.py`
（17 条，每条对应一次真实事故）；细则（状态机/阶梯/算力四档/结论分级）
下沉 GOALS.md 细则区；阶梯④新增 RSI 夜账到期检查（防断喂）；高危词
扫描修掉"两因结束 vs PARKED 结束会话"矛盾（改显式三因）。全仓
207+20=227 passed。

2026-09-27：新增 [循环体系](docs/loop/GOAL-PROMPT-M2.md) —— 移植
AwareLiquid-Physic 循环体系（GOAL-PROMPT-v8/AMM-035 语义）：`GOALS.md`
=程序计数器+goal_queue（4 条初始队列：P0-C′ 深度复测/2B 120K 腿/情节流/
RL 接线），`scripts/goal_check`=轮次路由器（每轮全队列达成检测/深位
弹出/current_variable 锚点回显，`--audit` 数数锚），`scripts/marathon_guard`
=锁检，`scripts/direction_gate`=提交前判单门（四轴耦合声明），驱动=用户
Desktop Go 拉式点火（AMM-003+AMM-010：点火一次=一个单目标连续循环，goal
校验驱动轮次直至队列清空或 BLOCKED-HUMAN；禁 cron/定时，ignite.sh 已于轮 11 归档），
同步=PR 流（AMM-004：fork 分支+PR，禁直推远端 main）。账本三件：
`RSI-INDEX.md`=定量指数+十轮节拍唯一定义处，`DISTILL.md`=四栏经验账本
（AMM-005 蒸馏门），`RSI-HORIZON.md`=外部对标+社区蒸馏门协议（AMM-008
问表）。训练腿跨轮（>30min 训练=status doing，轮次=判读落盘）。机制改动只走
[AMENDMENTS](docs/loop/AMENDMENTS.md) 提案（AMM-001 已采纳）。马拉松重启
=粘贴 GOAL-PROMPT-M2.md 的 text 围栏。全仓测试 206 passed（185+21）。

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

