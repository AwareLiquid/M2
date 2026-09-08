# AwareLiquid M2

Next-generation model architecture research, owned by Everest.

## Repository boundaries

- [M1](https://github.com/AwareLiquid/M1): stable model core, training and inference.
- [AwareLiquid-RAG](https://github.com/AwareLiquid/AwareLiquid-RAG): document retrieval and compression around an external frozen model. Previously named AwareLiquid-M2.
- **M2 (this repository)**: experimental recurrent reasoning, latent workspaces, world models and cognitive mechanisms.

## Current status

Repository established on 2026-09-08. Experimental source migration from M1
is pending dependency mapping and regression verification. No trained M2
checkpoint or benchmark improvement is claimed by this initialization.

## Direction

Develop a compact reasoning engine with persistent memory and optional external
knowledge retrieval. The 2B target is a research objective; parity with 70B
models on selected tasks requires measured evidence, not a parameter-count claim.

Experimental mechanisms graduate into M1 only after reproducible multi-seed
ablations, strong baselines and explicit parameter, compute and memory accounting.

Existing M1 imports and historical checkpoints remain supported during migration.
The RAG adapter's benchmark results do not establish M2 architecture performance.
