# M2 handoff

Owner: Everest.

2026-09-09：先读 [历史候选与训练准入](docs/RESEARCH_RECOVERY.md)。
融合式 PCLiquidCore＋生成式重放列回研究候选，不等于已接入本仓主模型；
旧快照中的测试计数和“所有旧路径均兼容”不能代表当前状态。

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
