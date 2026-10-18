"""Two-compartment dendritic unit vs point-neuron baselines for RSFC.

DendriticRSFC mirrors the Science 2026 finding: within one unit, the apical
(slow) compartment carries the INFERRED RULE and the somatic (fast) compartment
carries the SENSORY stimulus; the rule multiplicatively gates the sensory
readout (nonlinear integration). ``mode`` implements the causal ablations:
  full        normal
  soma_only   apical ignored (sensory-only readout)
  apical_only rule-only readout (no sensory)
  sst         "dendritic inhibition" at test: rule gate forced to 0
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DendriticRSFC(nn.Module):
    def __init__(self, d_soma: int = 32, d_apical: int = 32,
                 tau_soma: float = 1.5, tau_apical: float = 40.0):
        super().__init__()
        self.d_s, self.d_a = d_soma, d_apical
        self.decay_s = float(torch.exp(torch.tensor(-1.0 / tau_soma)))
        self.decay_a = float(torch.exp(torch.tensor(-1.0 / tau_apical)))
        # soma: fast sensory state — SENSORY input only (thalamic pathway)
        self.w_in_s = nn.Linear(1, d_soma)
        self.u_s = nn.Linear(d_soma, d_soma, bias=False)
        # apical: slow rule state — FEEDBACK input only (top-down rule pathway)
        self.w_in_a = nn.Linear(2, d_apical)
        self.u_a = nn.Linear(d_apical, d_apical, bias=False)
        # readouts
        self.w_out = nn.Linear(d_soma, 2)          # rule-gated sensory
        self.w_out_a = nn.Linear(d_apical, 2)      # apical-only control
        # dual-task v2: rule-INVARIANT sensory head (reads the soma only — the
        # paper's sensory compartment; must stay stable across rule switches)
        self.w_out_sens = nn.Linear(d_soma, 2)

    def forward_states(self, x: torch.Tensor, labels: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward that also returns per-step soma/apical states — for the
        rule-probe test (the paper: rule info lives at the dendrites and is
        largely absent at the soma)."""
        b, T = x.shape
        s = torch.zeros(b, self.d_s, device=x.device)
        a = torch.zeros(b, self.d_a, device=x.device)
        rew = torch.zeros(b, device=x.device)
        out, ss, aa = [], [], []
        for t in range(T):
            s = self.decay_s * s + (1 - self.decay_s) * torch.tanh(
                self.w_in_s(x[:, t:t+1]) + self.u_s(s))
            a = self.decay_a * a + (1 - self.decay_a) * torch.tanh(
                self.w_in_a(torch.stack([x[:, t], rew], dim=-1)) + self.u_a(a))
            logits = self.w_out(s * torch.tanh(a))
            out.append(logits)
            ss.append(s)
            aa.append(a)
            pred = logits.argmax(-1)
            rew = torch.where(pred == labels[:, t],
                              torch.ones_like(pred, dtype=torch.float32),
                              -torch.ones_like(pred, dtype=torch.float32))
        return torch.stack(out, 1), torch.stack(ss, 1), torch.stack(aa, 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor,
                mode: str = "full", teacher_reward: bool = True) -> torch.Tensor:
        """x (B,T), labels (B,T) -> logits (B,T,2). Reward input is the
        feedback of the PREVIOUS step's choice (online; used as an input)."""
        b, T = x.shape
        s = torch.zeros(b, self.d_s, device=x.device)
        a = torch.zeros(b, self.d_a, device=x.device)
        rew = torch.zeros(b, device=x.device)
        out = []
        for t in range(T):
            # anatomical input split (Science 2026): soma gets the sensory
            # stream, apical dendrites get the feedback/rule stream only.
            s = self.decay_s * s + (1 - self.decay_s) * torch.tanh(
                self.w_in_s(x[:, t:t+1]) + self.u_s(s))
            a = self.decay_a * a + (1 - self.decay_a) * torch.tanh(
                self.w_in_a(torch.stack([x[:, t], rew], dim=-1)) + self.u_a(a))
            if mode == "full":
                # bipolar rule gate: tanh(a) in [-1,1] can SCALE AND INVERT the
                # sensory readout — required because a rule switch reverses the
                # high/low mapping (a positive-only gate cannot express that).
                gate = torch.tanh(a)
                logits = self.w_out(s * gate)
            elif mode == "soma_only":
                logits = self.w_out(s)
            elif mode == "apical_only":
                logits = self.w_out_a(a)
            elif mode == "sst":                                # dendritic inhibition
                logits = self.w_out(s)
            else:
                raise ValueError(mode)
            out.append(logits)
            pred = logits.argmax(-1)
            rew = torch.where(pred == labels[:, t],
                              torch.ones_like(pred, dtype=torch.float32),
                              -torch.ones_like(pred, dtype=torch.float32))
        return torch.stack(out, dim=1)


    def forward_dual(self, x: torch.Tensor, labels: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual task v2: (category logits, rule-invariant sensory logits).
        The sensory head reads the SOMA only — the paper's sensory
        compartment — so it should stay stable across rule switches."""
        b, T = x.shape
        s = torch.zeros(b, self.d_s, device=x.device)
        a = torch.zeros(b, self.d_a, device=x.device)
        rew = torch.zeros(b, device=x.device)
        cat_out, sens_out = [], []
        for t in range(T):
            s = self.decay_s * s + (1 - self.decay_s) * torch.tanh(
                self.w_in_s(x[:, t:t+1]) + self.u_s(s))
            a = self.decay_a * a + (1 - self.decay_a) * torch.tanh(
                self.w_in_a(torch.stack([x[:, t], rew], dim=-1)) + self.u_a(a))
            cat_out.append(self.w_out(s * torch.tanh(a)))
            sens_out.append(self.w_out_sens(s))
            pred = cat_out[-1].argmax(-1)
            rew = torch.where(pred == labels[:, t],
                              torch.ones_like(pred, dtype=torch.float32),
                              -torch.ones_like(pred, dtype=torch.float32))
        return torch.stack(cat_out, 1), torch.stack(sens_out, 1)


class SingleLTC(nn.Module):
    """Point-neuron baseline: ONE liquid state, same inputs."""

    def __init__(self, d: int = 64, tau_init: float = 10.0):
        super().__init__()
        self.d = d
        self.log_tau = nn.Parameter(torch.full((d,), float(torch.log(torch.tensor(tau_init)))))
        self.w_in = nn.Linear(2, d)
        self.u = nn.Linear(d, d, bias=False)
        self.w_out = nn.Linear(d, 2)
        self.w_out_sens = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor, labels: torch.Tensor,
                mode: str = "full", teacher_reward: bool = True) -> torch.Tensor:
        b, T = x.shape
        decay = torch.exp(-1.0 / torch.nn.functional.softplus(self.log_tau))
        h = torch.zeros(b, self.d, device=x.device)
        rew = torch.zeros(b, device=x.device)
        out = []
        for t in range(T):
            inp = torch.stack([x[:, t], rew], dim=-1)
            h = decay * h + (1 - decay) * torch.tanh(self.w_in(inp) + self.u(h))
            logits = self.w_out(h)
            out.append(logits)
            pred = logits.argmax(-1)
            rew = torch.where(pred == labels[:, t],
                              torch.ones_like(pred, dtype=torch.float32),
                              -torch.ones_like(pred, dtype=torch.float32))
        return torch.stack(out, dim=1)


    def forward_dual(self, x: torch.Tensor, labels: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        b, T = x.shape
        decay = torch.exp(-1.0 / torch.nn.functional.softplus(self.log_tau))
        h = torch.zeros(b, self.d, device=x.device)
        rew = torch.zeros(b, device=x.device)
        cat_out, sens_out = [], []
        for t in range(T):
            inp = torch.stack([x[:, t], rew], dim=-1)
            h = decay * h + (1 - decay) * torch.tanh(self.w_in(inp) + self.u(h))
            cat_out.append(self.w_out(h))
            sens_out.append(self.w_out_sens(h))
            pred = cat_out[-1].argmax(-1)
            rew = torch.where(pred == labels[:, t],
                              torch.ones_like(pred, dtype=torch.float32),
                              -torch.ones_like(pred, dtype=torch.float32))
        return torch.stack(cat_out, 1), torch.stack(sens_out, 1)


class GRURSFC(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.gru = nn.GRUCell(2, d)
        self.w_out = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor, labels: torch.Tensor,
                mode: str = "full", teacher_reward: bool = True) -> torch.Tensor:
        b, T = x.shape
        h = torch.zeros(b, self.gru.hidden_size, device=x.device)
        rew = torch.zeros(b, device=x.device)
        out = []
        for t in range(T):
            inp = torch.stack([x[:, t], rew], dim=-1)
            h = self.gru(inp, h)
            logits = self.w_out(h)
            out.append(logits)
            pred = logits.argmax(-1)
            rew = torch.where(pred == labels[:, t],
                              torch.ones_like(pred, dtype=torch.float32),
                              -torch.ones_like(pred, dtype=torch.float32))
        return torch.stack(out, dim=1)


class MLPRSFC(nn.Module):
    """Memoryless point-neuron: cannot infer a rule from reward history."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, x: torch.Tensor, labels: torch.Tensor,
                mode: str = "full", teacher_reward: bool = True) -> torch.Tensor:
        b, T = x.shape
        rew = torch.zeros(b, T, device=x.device)
        # teacher-free: the MLP has no memory, so the reward channel carries the
        # previous trial's feedback (input only)
        inp = torch.stack([x, rew], dim=-1)
        return self.net(inp)
