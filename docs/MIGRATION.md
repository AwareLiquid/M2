# M1 experiment migration

Source: https://github.com/everest-an/M1
Source commit: `1e0114d10cc9b4edb3b60cf3aec428769a03b698`.
This is the preserved PR #46 branch, not a claim that PR #46 merged into M1.

The `mt_lnn/` package is imported verbatim with its shared runtime dependencies
because experimental loops are integrated into the model and configuration.
Keeping its import paths supports historical checkpoints and existing tests.
M2 owns future experimental development; this initial snapshot is not a new
model training result. M1 compatibility copies remain until downstream callers
and checkpoints can be migrated safely.

Migrated mechanisms include core/stack loops, GWT workspace, coherence,
predictive coding, world models, top-down modulation, rhythm, Hebbian plasticity,
astrocyte, neuromodulation and sleep consolidation. Research roadmaps and the
reasoning-depth benchmark accompany the package.

The accompanying tests cover model execution, scans, profiles, core/workspace
iterations, selective state, world models, top-down signals and Hebbian/rhythm
paths. Historical documents retain old paths and results; consult the source
commit for evidence files not included here.

Use a separate environment from M1: both distributions currently expose
`mt_lnn`. Do not install both into the same interpreter.

```sh
python -m pip install -e ".[test]"
python -m pytest tests -q
python benchmarks/reasoning_depth.py --smoke
```

No weights, secrets, training datasets or RAG project code are migrated.
