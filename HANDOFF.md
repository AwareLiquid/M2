# M2 handoff

Owner: Everest.

Local checkout: `E:\AwareLiquid\M2`.
Remote: https://github.com/AwareLiquid/M2.

## Immediate work

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
