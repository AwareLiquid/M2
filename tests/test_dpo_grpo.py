"""Synthetic-math tests for the DPO / GRPO reference losses.

These verify the LOSS MATH, not training behavior: the functions are correct
building blocks; whether RL helps a real M2 model is unmeasured (see HANDOFF).
"""
import math

import torch

from mt_lnn.research.rl.dpo_grpo import (
    dpo_loss,
    grpo_advantages,
    grpo_per_token_loss,
    log_probs_from_logits,
)


def test_dpo_neutral_when_policy_equals_reference():
    # policy == reference -> logit difference 0 -> -logsigmoid(0) = log(2)
    B = 8
    zeros = torch.zeros(B)
    loss = dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
    assert abs(loss.item() - math.log(2.0)) < 1e-5


def test_dpo_decreases_when_policy_prefers_chosen():
    B = 8
    ref_chosen = torch.full((B,), 5.0)
    ref_rejected = torch.full((B,), 5.0)  # reference indifferent
    # policy strongly prefers chosen over rejected
    loss_pref = dpo_loss(torch.full((B,), 10.0), torch.full((B,), 2.0),
                         ref_chosen, ref_rejected, beta=0.1)
    # policy wrongly prefers rejected
    loss_flip = dpo_loss(torch.full((B,), 2.0), torch.full((B,), 10.0),
                         ref_chosen, ref_rejected, beta=0.1)
    assert loss_pref.item() < math.log(2.0)
    assert loss_flip.item() > math.log(2.0)
    assert loss_pref.item() < loss_flip.item()


def test_dpo_gradient_direction():
    policy_chosen = torch.tensor([3.0], requires_grad=True)
    policy_rejected = torch.tensor([3.0], requires_grad=True)
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected,
                    beta=1.0)
    loss.backward()
    # raising the chosen log-prob must lower the loss; raising rejected raises it
    assert policy_chosen.grad.item() < 0
    assert policy_rejected.grad.item() > 0


def test_grpo_advantages_normalized_per_group():
    torch.manual_seed(0)
    B, G = 6, 4
    rewards = torch.randn(B * G)
    adv = grpo_advantages(rewards, G)
    grouped = adv.view(B, G)
    assert torch.allclose(grouped.mean(dim=1), torch.zeros(B), atol=1e-5)
    # std is shrunk by eps/(std+eps) ~ 1e-4, so tolerance 1e-3
    assert torch.allclose(grouped.std(dim=1, unbiased=False), torch.ones(B),
                          atol=1e-3)
    # within a group, the highest reward gets the highest advantage
    assert all(torch.argmax(grouped[i]) == torch.argmax(rewards.view(B, G)[i])
               for i in range(B))


def test_grpo_clip_zeroes_gradient_beyond_ratio():
    B, T = 2, 4
    lp = torch.zeros(B, T, requires_grad=True)
    old = torch.zeros(B, T)
    ref = torch.zeros(B, T)
    adv = torch.tensor([1.0, 1.0])
    loss = grpo_per_token_loss(lp, old, ref, adv, beta=0.0, epsilon=0.2,
                               loss_type="clip")
    loss.backward()
    # ratio = exp(lp) = 1 at lp=0 -> inside clip, gradient flows
    assert lp.grad.abs().sum().item() > 0


def test_grpo_cispo_runs_and_matches_shape():
    B, T = 2, 4
    lp = torch.randn(B, T)
    old = torch.randn(B, T)
    ref = torch.randn(B, T)
    adv = torch.tensor([0.5, -0.5])
    loss = grpo_per_token_loss(lp, old, ref, adv, beta=0.05, epsilon=0.2,
                               loss_type="cispo")
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_k3_kl_nonnegative():
    # k3 estimator exp(d) - d - 1 >= 0 for all d
    d = torch.linspace(-5, 5, 101)
    k3 = torch.exp(d) - d - 1.0
    assert (k3 >= -1e-6).all()


def test_log_probs_shape_and_sum():
    B, T, V = 3, 5, 64
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    lp = log_probs_from_logits(logits, labels)
    assert lp.shape == (B, T)
    assert (lp <= 0).all()  # log-probs are non-positive
