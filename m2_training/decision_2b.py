"""决策头训练 — 2B 冻结核 + 可训练 readout（JevBench 形状语料）。

核（MTLNNModel, 2B）冻结在 bf16；forward hook 抓 final_norm 输出 (B,T,d)；
可训练 readout = 选项文本嵌入(冻结的 embed_tokens, 均值池) ⊗ 状态逐位置
相似度的均值池化（与 O1-Flash 决策头同构，但表征来自 2B）。

    python -m m2_training.decision_2b --ckpt runs/t2b_30k.pt \
        --corpus decision_corpus.jsonl --steps 8000 --batch 8 \
        --out runs/decision_head_2b.pt [--size probe --no-ckpt 用于冒烟]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt_lnn.model import MTLNNModel  # noqa: E402
from m2_training.recipe import Recipe, model_config  # noqa: E402


def option_text(row: dict, lab: str) -> str:
    crit = row.get("criteria") or {}
    desc = crit.get(lab) if isinstance(crit, dict) else None
    return f"{lab}: {desc}" if desc else lab


def core_text(row: dict) -> str:
    state = str(row.get("state", ""))
    instr = row.get("instructions", "") or ""
    return (state + "\n" + instr).strip()


class DecisionReadout(nn.Module):
    def __init__(self, d_core: int = 2080, d_read: int = 512):
        super().__init__()
        self.opt_proj = nn.Linear(d_core, d_read, bias=False)
        self.state_proj = nn.Linear(d_core, d_read, bias=False)
        self.scale = d_read ** -0.5
        self.d_read = d_read

    def forward(self, y: torch.Tensor, opt_vecs: torch.Tensor,
                opt_mask: torch.Tensor,
                state_mask: torch.Tensor) -> torch.Tensor:
        """y (B,T,d_core); opt_vecs (B,K,d_core); masks -> scores (B,K)."""
        yp = self.state_proj(y)                     # (B,T,d_read)
        op = self.opt_proj(opt_vecs)                # (B,K,d_read)
        sim = torch.einsum("bkd,btd->bkt", op, yp) * self.scale
        sim = sim.masked_fill(~state_mask.unsqueeze(1), 0.0)
        denom = state_mask.sum(dim=1, keepdim=True).clamp(min=1)
        scores = sim.sum(dim=-1) / denom
        return scores.masked_fill(~opt_mask, float("-inf"))


def encode_bytes(text: str, max_len: int) -> list[int]:
    return list(text.encode("utf-8"))[: max_len - 1]


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.no_ckpt:
            recipe = Recipe(task="text", corpus="x.json", size="probe",
                            sequence_length=args.max_len)
        else:
            st0 = torch.load(args.ckpt, map_location="cpu", weights_only=True)
            base = Recipe(**st0["recipe"])
            recipe = Recipe(task="text", corpus="x.json", size=base.size,
                            sequence_length=args.max_len,
                            rope_scale=base.rope_scale)
        cfg = model_config(recipe, vocab=256, length=args.max_len)
        self.model = MTLNNModel(cfg)
        if not args.no_ckpt:
            st = torch.load(args.ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(
                st["model"], strict=False)
            print(f"core loaded (missing {len(missing)}, "
                  f"unexpected {len(unexpected)})", flush=True)
        self.model = self.model.to(torch.bfloat16).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.d_core = cfg.d_model

        # capture final_norm output
        self._captured = None

        def hook(_mod, _inp, out):
            self._captured = out

        self.model.final_norm.register_forward_hook(hook)

        self.head = DecisionReadout(self.d_core, args.d_read).to(self.device)
        self.head = self.head.to(torch.float32)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=args.lr)

    @torch.no_grad()
    def core_forward(self, texts: list[str]):
        seqs = [torch.tensor(encode_bytes(t, self.args.max_len),
                             dtype=torch.long) for t in texts]
        ids = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0).to(self.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, s in enumerate(seqs):
            mask[i, :len(s)] = True
        self.model(input_ids=ids)
        y = self._captured.float()                  # (B,T,d_core)
        return y, mask

    @torch.no_grad()
    def embed_options(self, rows: list[dict]):
        k_max = max(len(r["labels"]) for r in rows)
        opts = torch.zeros(len(rows), k_max, self.d_core, device=self.device)
        omask = torch.zeros(len(rows), k_max, dtype=torch.bool,
                            device=self.device)
        for b, r in enumerate(rows):
            for k, lab in enumerate(r["labels"]):
                ids = torch.tensor(encode_bytes(option_text(r, lab),
                                                self.args.max_len),
                                   dtype=torch.long, device=self.device)
                with torch.no_grad():
                    emb = self.model.embed_tokens(ids.unsqueeze(0))
                opts[b, k] = emb.float().mean(dim=1)[0]
                omask[b, k] = True
        return opts, omask

    def batch(self, rows: list[dict]):
        texts = [core_text(r) for r in rows]
        y, smask = self.core_forward(texts)
        opts, omask = self.embed_options(rows)
        targets = torch.tensor(
            [r["labels"].index(r["expected"]) for r in rows],
            device=self.device)
        return y, smask, opts, omask, targets

    def evaluate(self, rows: list[dict], batch: int = 16) -> float:
        correct = 0
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            y, smask, opts, omask, targets = self.batch(chunk)
            scores = self.head(y, opts, omask, smask)
            correct += int((scores.argmax(dim=-1) == targets).sum())
        return correct / max(len(rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/t2b_30k.pt")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--d-read", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--init-head", default="",
                    help="warm-start the readout from this checkpoint")
    ap.add_argument("--out", default="runs/decision_head_2b.pt")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")
            if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(64, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    print(f"corpus: {len(train)} train / {len(val)} val", flush=True)

    t = Trainer(args)
    if args.init_head:
        spec = torch.load(args.init_head, map_location="cpu", weights_only=True)
        t.head.load_state_dict(spec["head"])
        print(f"head warm-started from {args.init_head} "
              f"(prev val_acc {spec.get('val_acc')})", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        if step < args.warmup:
            cur = args.lr * (step + 1) / max(args.warmup, 1)
        else:
            prog = (step - args.warmup) / max(1, args.steps - args.warmup)
            cur = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in t.opt.param_groups:
            g["lr"] = cur
        chunk = rng.sample(train, args.batch)
        y, smask, opts, omask, targets = t.batch(chunk)
        scores = t.head(y, opts, omask, smask)
        loss = F.cross_entropy(scores, targets)
        t.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(t.head.parameters(), 1.0)
        t.opt.step()
        if (step + 1) % args.log_every == 0:
            print(f"step {step+1}/{args.steps} loss {loss.item():.4f} "
                  f"lr {cur:.2e} {(step+1)/(time.time()-t0):.2f} step/s",
                  flush=True)
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            acc = t.evaluate(val)
            print(f"  [val] step {step+1} acc {acc:.4f}", flush=True)
            if args.save_every and (step + 1) % args.save_every == 0:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save({"head": t.head.state_dict(),
                            "steps": step + 1, "val_acc": acc,
                            "d_core": t.d_core, "d_read": args.d_read,
                            "max_len": args.max_len}, args.out)
                print(f"  [saved] {args.out} @ {step+1}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"head": t.head.state_dict(), "steps": args.steps,
                "d_core": t.d_core, "d_read": args.d_read,
                "max_len": args.max_len}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
