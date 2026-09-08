"""Fast-weight associative memory (content-addressable, causal, in-graph).

Extracted from ``llama_adapter.py`` so the model core (``mt_lnn.model``) can
use the associative-memory primitive without importing the serving/adapter
stack. See the ``FastWeightMemory`` docstring for the math and why it exists.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastWeightMemory(nn.Module):
    """Causal fast-weight associative memory (Ba et al. 2016; gated linear attn).

    Within a sequence it WRITES key->value associations into a per-sample fast-
    weight matrix F and READS them back by content, all inside the autograd graph
    so the main task loss (CE) trains it end-to-end. This is the architectural
    answer to two findings at once:

      * Hebbian-as-side-loss is inert (gradient ~1e-12, drowned by CE). A forward-
        pass memory has a first-order effect on the output, so it is learnable at
        0.5B scale -- no need to wait for 4-8B.
      * The served adapter had no content-addressable memory, hence the honest
        needle-in-haystack 0% result. Fast weights are the principled fix.

    Recurrence (per position t, strictly causal):
        k_t, q_t = phi(W_k x_t), phi(W_q x_t)     # phi = elu+1 -> positive features
        v_t      = W_v x_t
        F_t = lam * F_{t-1} + k_t (outer) v_t      # write
        z_t = lam * z_{t-1} + k_t                  # running key normaliser
        r_t = (q_t @ F_t) / (q_t . z_t + eps)      # associative read
        out = W_o r_t
    lam in (0,1) is a learnable per-head decay (sigmoid of a raw param): how long
    a written association survives. F_t depends only on positions <= t, so the
    read is causal and safe for autoregressive LM.

    NOTE: this reference forward scans T sequentially (O(T) python steps) for
    clarity and exactness. A chunked/parallel scan is the production speed
    optimisation and does not change the math.
    """

    def __init__(self, d_model: int, d_mem: int = 64, n_heads: int = 1,
                 init_decay: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        inner = n_heads * d_mem
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)
        # Learnable decay per head, initialised near `init_decay` via logit.
        init_decay = min(max(init_decay, 1e-3), 1 - 1e-3)
        raw = math.log(init_decay / (1.0 - init_decay))
        self.decay_raw = nn.Parameter(torch.full((n_heads,), float(raw)))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, T, d_model). Returns (out (B,T,d_model), (F, z)) where the
        returned state can be fed back in to continue the memory across calls
        (used later for streaming / cross-session persistence)."""
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_mem
        k = (F.elu(self.W_k(x)) + 1.0).view(B, T, H, D)
        q = (F.elu(self.W_q(x)) + 1.0).view(B, T, H, D)
        v = self.W_v(x).view(B, T, H, D)
        decay = torch.sigmoid(self.decay_raw)              # (H,)
        dF = decay.view(1, H, 1, 1)
        dz = decay.view(1, H, 1)

        if state is None:
            Fmat = x.new_zeros(B, H, D, D)                  # (B,H,d_k,d_v)
            zvec = x.new_zeros(B, H, D)                     # (B,H,d_k)
        else:
            Fmat, zvec = state

        reads = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]          # each (B,H,D)
            Fmat = dF * Fmat + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            zvec = dz * zvec + kt
            num = torch.einsum("bhd,bhde->bhe", qt, Fmat)   # (B,H,d_v)
            den = torch.einsum("bhd,bhd->bh", qt, zvec).clamp_min(1e-6).unsqueeze(-1)
            reads.append((num / den).reshape(B, H * D))
        r = torch.stack(reads, dim=1)                       # (B,T,H*D)
        return self.W_o(r), (Fmat, zvec)
