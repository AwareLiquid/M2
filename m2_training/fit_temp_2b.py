"""Fit the decision-head temperature on the corpus val split (NLL-optimal).

    /root/M2/.venv/bin/python fit_temp_2b.py

Writes /root/decision/head2b_temperature.json {temperature, ece_before,
ece_after, n}. The service applies it when present; the checkpoint is not
mutated.
"""
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/M2")
from mt_lnn.model import MTLNNModel  # noqa: E402
from m2_training.recipe import Recipe, model_config  # noqa: E402
from m2_training.decision_2b import (  # noqa: E402
    DecisionReadout, core_text, encode_bytes, option_text)

CORPUS = "/root/decision/decision_corpus.jsonl"
HEAD = "/root/decision/decision_head_2b.pt"
CORE = "/root/M2/runs/t2b_30k.pt"
OUT = "/root/decision/head2b_temperature.json"


def ece(probs: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> float:
    conf, pred = probs.max(dim=-1)
    acc = (pred == targets).float()
    total = 0.0
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf <= hi)
        if mask.sum() == 0:
            continue
        total += float(mask.float().mean()) * abs(
            float(acc[mask].mean()) - float(conf[mask].mean()))
    return total


def main() -> None:
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")
            if l.strip()]
    rng = random.Random(0)
    rng.shuffle(rows)
    n_val = max(64, int(len(rows) * 0.02))
    val = rows[:n_val]

    spec = torch.load(HEAD, map_location="cpu", weights_only=True)
    max_len = int(spec.get("max_len", 384))
    st = torch.load(CORE, map_location="cpu", weights_only=True)
    recipe = Recipe(**st["recipe"])
    recipe = Recipe(task="text", corpus="x.json", size=recipe.size,
                    sequence_length=max_len)
    cfg = model_config(recipe, vocab=256, length=max_len)
    model = MTLNNModel(cfg)
    model.load_state_dict(st["model"], strict=False)
    model = model.to(torch.bfloat16).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cap = {}
    model.final_norm.register_forward_hook(
        lambda _m, _i, out: cap.__setitem__("y", out))
    head = DecisionReadout(cfg.d_model, int(spec.get("d_read", 512))).cuda()
    head.load_state_dict(spec["head"])
    head.eval()

    all_scores, all_targets = [], []
    with torch.no_grad():
        for i in range(0, len(val), 16):
            chunk = val[i:i + 16]
            seqs = [torch.tensor(encode_bytes(core_text(r), max_len),
                                 dtype=torch.long) for r in chunk]
            ids = torch.nn.utils.rnn.pad_sequence(
                seqs, batch_first=True, padding_value=0).cuda()
            smask = torch.zeros_like(ids, dtype=torch.bool)
            for j, s in enumerate(seqs):
                smask[j, :len(s)] = True
            model(input_ids=ids)
            y = cap["y"].float()
            k = max(len(r["labels"]) for r in chunk)
            opts = torch.zeros(len(chunk), k, cfg.d_model, device="cuda")
            omask = torch.zeros(len(chunk), k, dtype=torch.bool,
                                device="cuda")
            for b, r in enumerate(chunk):
                for j, lab in enumerate(r["labels"]):
                    oids = torch.tensor(
                        encode_bytes(option_text(r, lab), max_len),
                        dtype=torch.long, device="cuda")
                    opts[b, j] = model.embed_tokens(
                        oids[None]).float().mean(dim=1)[0]
                    omask[b, j] = True
            scores = head(y, opts, omask, smask)
            targets = torch.tensor(
                [r["labels"].index(r["expected"]) for r in chunk],
                device="cuda")
            all_scores.append(scores.masked_fill(~omask, -1e4).cpu())
            all_targets.append(targets.cpu())

    k = max(s.shape[1] for s in all_scores)
    S = torch.full((len(val), k), -1e4)
    T = torch.zeros(len(val), dtype=torch.long)
    idx = 0
    for s, t in zip(all_scores, all_targets):
        S[idx:idx + len(s), :s.shape[1]] = s
        T[idx:idx + len(t)] = t
        idx += len(s)

    probs0 = F.softmax(S, dim=-1)
    ece0 = ece(probs0, T)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(S / log_T.exp(), T)
        loss.backward()
        return loss

    opt.step(closure)
    temp = float(log_T.exp().item())
    probs1 = F.softmax(S / temp, dim=-1)
    ece1 = ece(probs1, T)
    acc = float((probs1.argmax(-1) == T).float().mean())
    print(f"temp {temp:.3f}  ECE {ece0:.4f} -> {ece1:.4f}  acc {acc:.4f}  "
          f"n={len(val)}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"temperature": temp, "ece_before": ece0,
                   "ece_after": ece1, "acc": acc, "n": len(val)}, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
