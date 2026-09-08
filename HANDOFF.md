# M2 handoff

Owner: Everest.

Local checkout: `E:\AwareLiquid\M2`.
Remote: https://github.com/AwareLiquid/M2.

## Immediate work

1. Inventory M1 experimental modules and their shared-core dependencies.
2. Move experiments, direct tests and research documentation into this repository.
3. Pin the required M1 core revision and verify imports, forward/backward and caches.
4. Replace the earlier same-repository proposal in M1 PR #46 with the cross-repository migration plan.

## Preservation

The former document-QA checkout is now `E:\AwareLiquid\RAG`, remote
`AwareLiquid/AwareLiquid-RAG`. Its untracked `local_verifier_server.py` was
preserved during the directory rename. Do not copy it or QA scores into this model repository.

No experiment migration or new training has been completed by this initial commit.
