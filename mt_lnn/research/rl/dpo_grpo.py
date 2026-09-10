"""Reference implementations of alignment losses: DPO and GRPO.

Adapted for the M2 experimental line from the standard formulations
(Rafailov et al. 2023 for DPO; Shao et al. 2024 for GRPO). The math is
unit-tested on synthetic tensors — correctness of the LOSS MATH is verified;
training-time benefit on a real M2 model is NOT yet measured (no RL run
exists in this repo). Use only behind an experiment flag.

Honest status: these are reference building blocks for a future RL stage of
the 2B line, not a claim that RL works on the liquid core. See HANDOFF.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss over already-summed per-sequence log-probs.

    Each input is (B,) — the sequence-summed (optionally masked) log-prob
    of the chosen/rejected completion under the policy and the frozen
    reference model.

    loss = -logsigmoid( beta * ((p_chosen - p_rejected) - (r_chosen - r_rejected)) )

    Properties (unit-tested):
      * policy == reference  -> loss = log(2) ~ 0.693 (no preference signal)
      * policy prefers chosen -> loss < log(2); the stronger, the lower
      * swapping chosen/rejected flips the direction of improvement
    """
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()


def grpo_advantages(rewards: torch.Tensor, num_generations: int, eps: float = 1e-4) -> torch.Tensor:
    """Group-wise advantage normalization (GRPO).

    rewards: (B * G,) flattened per-prompt rewards, grouped G at a time.
    Returns advantages (B * G,) with per-group mean 0 and std ~1: group
    statistics are repeated across each group's members.
    """
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    adv = (grouped - mean) / (std + eps)
    return adv.reshape(-1)


def grpo_per_token_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    epsilon: float,
    loss_type: str = "clip",
) -> torch.Tensor:
    """GRPO per-token objective with ratio clipping and a k3 KL estimator.

    per_token_logps / old / ref: (B, T). advantages: (B,) per sequence.

    clip (PPO-style): -min(ratio * A, clip(ratio, 1-e, 1+e) * A) + beta * k3
    cispo:           -(clamp(ratio, max=e_high).detach() * A * logps - beta * k3)

    k3 = exp(ref - policy) - (ref - policy) - 1  (always >= 0, low-variance)
    """
    kl_div = ref_per_token_logps - per_token_logps
    k3 = torch.exp(kl_div) - kl_div - 1.0
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    A = advantages.unsqueeze(1)  # (B, 1)

    if loss_type == "cispo":
        clamped = torch.clamp(ratio, max=epsilon).detach()
        per_token = -(clamped * A * per_token_logps - beta * k3)
    else:
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        per_token = -torch.min(ratio * A, clipped * A) + beta * k3
    return per_token.mean()


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """(B, T, V) logits + (B, T) labels -> (B, T) per-token log-probs."""
    return torch.gather(F.log_softmax(logits, dim=-1), -1, labels.unsqueeze(-1)).squeeze(-1)
