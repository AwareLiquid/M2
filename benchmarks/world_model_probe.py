#!/usr/bin/env python3
"""World-model paradigm probe (入口 B): state prediction, no next-token anywhere.

Trains a liquid backbone + JEPA-style PredictiveStateHead on physics state
trajectories (harmonic oscillator), then evaluates the multi-step IMAGINATION
rollout against the true future and a "nothing changes" static baseline.

Judgment (pre-registered): imagined trajectory error at H=16 < 0.5 x static
baseline error, and error growth slope < baseline slope -> the model LEARNED
the dynamics from states alone.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# M1 components (uploaded alongside): physics_ops / world_model / imagination
from mt_lnn.physics_ops import rollout  # noqa: E402
from mt_lnn.world_model import PredictiveStateHead  # noqa: E402
from mt_lnn.imagination import LatentImagination  # noqa: E402


# ---------- world: harmonic oscillator trajectories ----------
def gen_oscillator_trajectories(n_traj, T, dt=0.05, seed=0):
    rng = np.random.default_rng(seed)
    k = 4.0  # spring constant -> period ~pi
    trajs = []
    for _ in range(n_traj):
        x0 = rng.uniform(-1.5, 1.5)
        v0 = rng.uniform(-1.0, 1.0)
        x, v = x0, v0
        states = []
        for _ in range(T):
            states.append([x, v])
            a = -k * x
            v = v + a * dt
            x = x + v * dt
        trajs.append(np.asarray(states, dtype=np.float32))
    return np.stack(trajs)  # (n_traj, T, 2)


# ---------- minimal liquid backbone (multi-scale decay scan, O(1) state) ----------
class LiquidBlock(nn.Module):
    """Minimal liquid recurrent block: per-scale exponential decay scan + blend.
    h_s[t] = decay_s * h_s[t-1] + (1 - decay_s) * W(x[t]);  y[t] = sum_s a_s h_s[t].
    Carried state = (n_scales, d) per layer — O(1), independent of T."""
    def __init__(self, d, n_scales=5):
        super().__init__()
        self.d = d
        self.n_scales = n_scales
        self.W = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d)
        self.log_tau = nn.Parameter(torch.linspace(-2.0, 3.0, n_scales))
        self.blend = nn.Parameter(torch.zeros(n_scales))

    def forward(self, x, h_prev=None):
        B, T, d = x.shape
        u = torch.tanh(self.W(x))  # (B, T, d)
        tau = F.softplus(self.log_tau)  # (S,)
        decay = torch.exp(-1.0 / tau)  # (S,)
        a = torch.softmax(self.blend, dim=0)  # (S,)
        hs = []
        h = torch.zeros(B, self.n_scales, d, device=x.device) if h_prev is None else h_prev
        out = []
        for t in range(T):
            h = decay[None, :, None] * h + (1 - decay[None, :, None]) * u[:, t, None, :]
            y = torch.einsum("s,bsd->bd", a, h)
            out.append(y)
        y = torch.stack(out, dim=1)  # (B, T, d)
        return y, h


class LiquidBackbone(nn.Module):
    def __init__(self, d_state, d_model, n_layers=2):
        super().__init__()
        self.inp = nn.Linear(d_state, d_model)
        self.blocks = nn.ModuleList(
            [LiquidBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.inp(s)
        for blk in self.blocks:
            x, _ = blk(x)
        return self.norm(x)  # (B, T, d_model)


def train(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = gen_oscillator_trajectories(args.n_traj, args.T, seed=args.seed)
    data = torch.from_numpy(data).to(device)

    backbone = LiquidBackbone(2, args.d_model, args.n_layers).to(device)
    head = PredictiveStateHead(args.d_model).to(device)
    opt = torch.optim.Adam(
        list(backbone.parameters()) + [p for p in head.parameters() if p.requires_grad],
        lr=args.lr)

    # ---- train: state prediction (JEPA) — NO next-token ----
    for step in range(args.steps):
        idx = torch.randint(0, args.n_traj, (args.batch,))
        s = data[idx]  # (B, T, 2)
        h = backbone(s)  # (B, T, d)
        _, loss = head(h, compute_loss=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"  step {step} wm_loss {loss.item():.4f}", flush=True)

    # ---- eval: imagination rollout vs true future vs static ----
    backbone.eval()
    head.eval()
    imag = LatentImagination(head)
    with torch.no_grad():
        s = data[:args.eval_traj]
        h = backbone(s)

        # true future in the imagination's own latent space (normalised online_proj)
        z_true = []
        for k in range(1, args.H + 1):
            z_true.append(torch.nn.functional.normalize(
                head.online_proj(h[:, args.eval_start + k]), dim=-1))
        z_true = torch.stack(z_true, dim=1)  # (B, H, P)

        # imagined future (already normalised)
        imagined = imag.imagine(h[:, args.eval_start], args.H)
        z_imag = imagined.latents  # (B, H, P)

        # static baseline: "nothing changes" = repeat the present latent
        z0 = torch.nn.functional.normalize(
            head.online_proj(h[:, args.eval_start]), dim=-1)
        z_static = z0.unsqueeze(1).repeat(1, args.H, 1)

        def err(z):
            d = (z - z_true).norm(dim=-1).mean(dim=0)  # per-horizon mean error
            return d

        e_imag = err(z_imag).cpu().numpy()
        e_static = err(z_static).cpu().numpy()

    print("imagination err @H=1/4/16: %.4f / %.4f / %.4f" %
          (e_imag[0], e_imag[3], e_imag[15]))
    print("static     err @H=1/4/16: %.4f / %.4f / %.4f" %
          (e_static[0], e_static[3], e_static[15]))
    ratio16 = e_imag[15] / max(e_static[15], 1e-9)
    slope_imag = e_imag[15] - e_imag[0]
    slope_static = e_static[15] - e_static[0]
    verdict = (ratio16 < 0.5) and (slope_imag < slope_static)
    print("ratio16=%.3f  slope_imag=%.4f slope_static=%.4f -> H %s" %
          (ratio16, slope_imag, slope_static, "SUPPORTED" if verdict else "REJECTED"))
    return verdict


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_traj", type=int, default=256)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--eval_traj", type=int, default=32)
    ap.add_argument("--eval_start", type=int, default=64)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
