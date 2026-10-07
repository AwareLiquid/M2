"""AwareLiquid 2B+决策头服务 — TypeSafe /v1/systemone（冻结核 + readout 单前向）。

核 = M2-2B（bf16 冻结）；head = m2_training.decision_2b 的 DecisionReadout。
口径与训练一致（option_text/core_text/encode_bytes）。

    /root/M2/.venv/bin/python jev_service_2b.py <head_ckpt> [port]
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/M2")
from mt_lnn.model import MTLNNModel  # noqa: E402
from m2_training.recipe import Recipe, model_config  # noqa: E402
from m2_training.decision_2b import (  # noqa: E402
    DecisionReadout, core_text, encode_bytes, option_text)

HEAD_CKPT = sys.argv[1] if len(sys.argv) > 1 else "/root/decision/decision_head_2b.pt"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8299
CORE_CKPT = os.environ.get("CORE_CKPT", "/root/M2/runs/t2b_30k.pt")
MODEL_NAME = Path(HEAD_CKPT).stem

print(f"loading head spec {HEAD_CKPT} ...", flush=True)
spec = torch.load(HEAD_CKPT, map_location="cpu", weights_only=True)
MAX_LEN = int(spec.get("max_len", 384))
D_READ = int(spec.get("d_read", 512))

st = torch.load(CORE_CKPT, map_location="cpu", weights_only=True)
recipe = Recipe(**st["recipe"])
recipe = Recipe(task="text", corpus="x.json", size=recipe.size,
                sequence_length=MAX_LEN)
cfg = model_config(recipe, vocab=256, length=MAX_LEN)
model = MTLNNModel(cfg)
missing, unexpected = model.load_state_dict(st["model"], strict=False)
print(f"core loaded (missing {len(missing)}, unexpected {len(unexpected)})",
      flush=True)
model = model.to(torch.bfloat16).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)

_captured = {}


def hook(_m, _i, out):
    _captured["y"] = out


model.final_norm.register_forward_hook(hook)

head = DecisionReadout(cfg.d_model, D_READ).cuda().eval()
head.load_state_dict(spec["head"])

# post-hoc temperature (fit on the corpus val split; optional sidecar)
TEMP = 1.0
TEMP_PATH = "/root/decision/head2b_temperature.json"
try:
    with open(TEMP_PATH, encoding="utf-8") as f:
        TEMP = float(json.load(f)["temperature"])
    print(f"temperature {TEMP:.3f} loaded from {TEMP_PATH}", flush=True)
except Exception:
    print("no temperature sidecar; serving raw softmax", flush=True)
print("ready.", flush=True)


def derive_labels(qtype: str, criteria):
    if qtype == "noul":
        return ["no", "yes"]
    if isinstance(criteria, dict):
        return [str(k) for k in criteria.keys()]
    if isinstance(criteria, list):
        return [str(i) for i in range(len(criteria))]
    return []


@torch.no_grad()
def decide(payload: dict) -> dict:
    q = (payload.get("questions") or {}).get("decision") or {}
    qtype = q.get("type")
    criteria = q.get("criteria")
    labels = derive_labels(qtype, criteria)
    if not labels:
        raise ValueError("no labels derivable")
    state = payload.get("state", "")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    row = {"state": state, "instructions": q.get("instructions", "") or "",
           "criteria": criteria if isinstance(criteria, dict) else None,
           "labels": labels}

    text = core_text(row)
    ids = torch.tensor(encode_bytes(text, MAX_LEN), dtype=torch.long,
                       device="cuda")[None]
    model(input_ids=ids)
    y = _captured["y"].float()
    smask = torch.ones_like(ids, dtype=torch.bool)

    k = len(labels)
    opts = torch.zeros(1, k, cfg.d_model, device="cuda")
    for j, lab in enumerate(labels):
        oids = torch.tensor(encode_bytes(option_text(row, lab), MAX_LEN),
                            dtype=torch.long, device="cuda")
        opts[0, j] = model.embed_tokens(oids[None]).float().mean(dim=1)[0]
    omask = torch.ones(1, k, dtype=torch.bool, device="cuda")

    scores = head(y, opts, omask, smask)
    probs = F.softmax(scores / TEMP, dim=-1)[0]
    prob_map = {lab: float(p) for lab, p in zip(labels, probs)}
    if qtype == "noul":
        answer = {"type": "noul", "noul": float(probs[labels.index("yes")])}
    elif qtype == "choice":
        best = max(sorted(prob_map), key=lambda kk: prob_map[kk])
        answer = {"type": "choice", "choice": best, "probabilities": prob_map}
    else:
        answer = {"type": qtype, "probabilities": prob_map}
    return {"answers": {"decision": answer}, "model": MODEL_NAME,
            "usage": {"input_tokens": int(ids.shape[1]), "output_tokens": 0}}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, {"status": "ok", "model": MODEL_NAME})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/systemone":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            t0 = time.time()
            resp = decide(payload)
            dt = (time.time() - t0) * 1000
            ans = resp["answers"]["decision"]
            print(f"[{ans.get('type')}] {dt:.1f}ms "
                  f"{ans.get('choice', ans.get('noul'))}", flush=True)
            self._send(200, resp)
        except Exception as ex:  # noqa: BLE001
            print("ERROR:", type(ex).__name__, str(ex)[:200], flush=True)
            self._send(500, {"error": f"{type(ex).__name__}: {str(ex)[:200]}"})

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    print(f"serving on 0.0.0.0:{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
