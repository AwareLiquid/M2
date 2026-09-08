# AwareLiquid M2

Next-generation model architecture research, owned by Everest.

## Repository boundaries

- [M1](https://github.com/AwareLiquid/M1): stable model core, training and inference.
- [AwareLiquid-RAG](https://github.com/AwareLiquid/AwareLiquid-RAG): document retrieval and compression around an external frozen model. Previously named AwareLiquid-M2.
- **M2 (this repository)**: experimental recurrent reasoning, latent workspaces, world models and cognitive mechanisms.

## Current status

Experimental sources and their fixed-version runtime dependencies are now
imported from M1. See [migration provenance and usage](docs/MIGRATION.md).
No trained M2 checkpoint or benchmark improvement is claimed by this migration.

## Training and checkpoint resume

An independent synthetic-reasoning and pretokenized-text training entry point is available:

```sh
python -m m2_training train --checkpoint runs/probe.pt --steps 30
python -m m2_training resume --checkpoint runs/probe.pt --steps 60
python -m m2_training evaluate --checkpoint runs/probe.pt
```

See [training instructions and module wiring](docs/TRAINING.md) for supported
experiments, text preparation, GPU usage and limitations. A roughly 2B model
configuration is available for appropriately provisioned machines; no trained
2B checkpoint is provided. Extracted experiments live in `mt_lnn/research/`.

## Research direction

Develop a compact reasoning engine with persistent memory and optional external
knowledge retrieval. The 2B target is a research objective; parity with 70B
models on selected tasks requires measured evidence, not a parameter-count claim.

Experimental mechanisms graduate into M1 only after reproducible multi-seed
ablations, strong baselines and explicit parameter, compute and memory accounting.

Existing M1 imports and historical checkpoints remain supported during migration.
The RAG adapter's benchmark results do not establish M2 architecture performance.
