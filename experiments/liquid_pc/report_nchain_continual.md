# PC-Liquid-Core 2-task generative-replay continual learning (deep chain)

Seeds: `[0]` (n=1)  |  2 tasks  |  chained dream buffer=8 (warmup 16)  |  runtime 2.9s

Deep Generative Replay (scholar) over 2 well-separated timescale regimes (slow->fast). After each task the current model dreams a buffer of all tasks seen so far; the next task rehearses it. Forgetting is measured vs each task's error right after it was learned. `worst_final` = max held-out 1-step MSE across all tasks = the 'is the whole system still usable' number (lower=better).

| kind | mean forget | WORST task final MSE |
|---|---:|---:|
| PC | -0.30017 +/- 0.00000 | 0.14891 +/- 0.00000 (max 0.14891) |
| GRU | -0.29607 +/- 0.00000 | 0.50213 +/- 0.00000 (max 0.50213) |

**Paired PC-GRU mean forgetting:** diff -0.00411, PC lower in 1/1.

**Paired PC-GRU WORST-task final MSE:** diff -0.35322, PC lower in 1/1.  (worst-task is the usability number; large positive GRU worst = collapse.)

Per-seed worst-task index (which timescale regime is the system's weakest link):
- PC : [1]
- GRU: [1]
