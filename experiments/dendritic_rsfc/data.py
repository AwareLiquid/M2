"""RSFC — rule-switching flexible categorization task (Science 2026, adz5323).

Synthetic mirror of the mouse task: on each trial a tone x_t is drawn from
U(0,1) and must be categorized "high"/"low" relative to a boundary that flips
between two hidden rules at random block boundaries. The model sees the current
tone plus the PREVIOUS trial's reward feedback (+1 correct / -1 wrong) and must
INFER the current rule from the reward history, then apply it.

Outputs per episode:
  x       (B,T)  tone value
  rew_in  (B,T)  feedback of trial t-1 (0 at t=0) — the only rule information
  label   (B,T)  1 = "high", 0 = "low" under the CURRENT rule
  rule    (B,T)  0 normal (high if x>0.5) / 1 reversed
  switch  (B,T)  1 if the rule changed at this trial (must re-infer)
"""
from __future__ import annotations

import numpy as np


def rsfc_episodes(batch: int, T: int = 240, block_lo: int = 25,
                  block_hi: int = 55, seed: int = 0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((batch, T)).astype(np.float32)
    rule = np.zeros((batch, T), dtype=np.int64)
    switch = np.zeros((batch, T), dtype=np.float32)
    for b in range(batch):
        r = 0
        t = 0
        first = True
        while t < T:
            end = min(t + int(rng.integers(block_lo, block_hi)), T)
            rule[b, t:end] = r
            if not first:
                switch[b, t] = 1.0
            first = False
            t = end
            r = 1 - r
    label = ((x > 0.5).astype(np.int64) ^ rule).astype(np.int64)  # rule 1 reverses
    rew_in = np.zeros((batch, T), dtype=np.float32)               # filled online
    return x, rew_in, label, rule, switch


def feedback(prev_label_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reward of the previous trial: +1 if the prediction was right else -1."""
    return np.where(prev_label_pred == labels, 1.0, -1.0).astype(np.float32)
