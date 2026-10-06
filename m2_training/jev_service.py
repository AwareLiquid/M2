"""AwareLiquid decision service — native TypeSafe /v1/systemone wire format.

Reads the model's own distribution over the exact label set (per-label
sequence-likelihood softmax under our byte-level model — the "native
distribution" an open-weights rebuild exposes), so the official JevBench
harness can score it as a native system.

    /root/M2/.venv/bin/python jev_service.py <ckpt> [port]
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/M2")
from m2_training.runner import TrainingRun  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/t2b_sft_v2.pt"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8099
CORPUS = "/root/M2/sft_data/corpus512/manifest.json"
MODEL_NAME = Path(CKPT).stem

print(f"loading {CKPT} ...", flush=True)
run = TrainingRun.restore(Path(CKPT), device="cuda", corpus=CORPUS,
                          task="sft", sequence_length=512)
MODEL = run.model.eval()
print("ready.", flush=True)


def derive_labels(qtype: str, criteria):
    if qtype == "noul":
        return ["no", "yes"]
    if isinstance(criteria, dict):
        return [str(k) for k in criteria.keys()]
    if isinstance(criteria, list):
        return [str(i) for i in range(len(criteria))]
    return []


def criterion_text(qtype: str, criteria, label: str) -> str:
    if criteria is None:
        return ""
    if isinstance(criteria, list):
        try:
            return str(criteria[int(label)])
        except (ValueError, IndexError):
            return ""
    if label in criteria:
        return str(criteria[label])
    low = label.lower()
    if low in criteria:
        return str(criteria[low])
    if qtype == "noul":
        if low == "no" and "false" in criteria:
            return str(criteria["false"])
        if low == "yes" and "true" in criteria:
            return str(criteria["true"])
    return ""


def build_prompt(state: str, instructions: str, qtype: str, criteria,
                 labels: list) -> str:
    opts = []
    for lab in labels:
        desc = criterion_text(qtype, criteria, lab)
        opts.append(f"- {lab}" + (f": {desc}" if desc else ""))
    return (state + "\n\n" + (instructions + "\n" if instructions else "")
            + "Options:\n" + "\n".join(opts) + "\n\nAnswer with the label.\n\n")


@torch.no_grad()
def label_logprob(prompt_bytes: bytes, label_bytes: bytes,
                  max_len: int = 512) -> float:
    full = (prompt_bytes + label_bytes)[:max_len]
    if len(full) <= len(prompt_bytes):
        return float("-inf")
    ids = torch.from_numpy(
        np.frombuffer(full, dtype=np.uint8).astype(np.int64))[None].cuda()
    logits = MODEL(ids)["logits"][0]
    lp = 0.0
    for i, b in enumerate(label_bytes):
        pos = len(prompt_bytes) - 1 + i
        if pos >= logits.shape[0]:
            break
        lp += float(torch.log_softmax(logits[pos].float(), dim=-1)[b])
    return lp


@torch.no_grad()
def decide(payload: dict) -> dict:
    q = (payload.get("questions") or {}).get("decision") or {}
    qtype = q.get("type")
    instructions = q.get("instructions", "") or ""
    criteria = q.get("criteria")
    state = payload.get("state", "")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    labels = derive_labels(qtype, criteria)
    if not labels:
        raise ValueError("no labels derivable")
    prompt = build_prompt(state, instructions, qtype, criteria, labels)
    pb = prompt.encode("utf-8")[:500]
    lps = torch.tensor([label_logprob(pb, lab.encode("utf-8"))
                        for lab in labels], dtype=torch.float64)
    probs = torch.softmax(lps, dim=-1).numpy()
    if qtype == "noul":
        p_yes = float(probs[labels.index("yes")])
        answer = {"type": "noul", "noul": p_yes}
    elif qtype == "choice":
        prob_map = {lab: float(p) for lab, p in zip(labels, probs)}
        best = max(sorted(prob_map), key=lambda k: prob_map[k])
        answer = {"type": "choice", "choice": best,
                  "probabilities": prob_map}
    else:  # score
        answer = {"type": qtype,
                  "probabilities": {lab: float(p)
                                    for lab, p in zip(labels, probs)}}
    return {"answers": {"decision": answer}, "model": MODEL_NAME,
            "usage": {"input_tokens": len(pb), "output_tokens": 0}}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # health
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
            print(f"[{resp['answers']['decision'].get('type')}] "
                  f"{dt:.0f}ms {list(resp['answers']['decision'].get('probabilities', {}).items())[:2]}",
                  flush=True)
            self._send(200, resp)
        except Exception as ex:  # noqa: BLE001
            print("ERROR:", type(ex).__name__, str(ex)[:200], flush=True)
            self._send(500, {"error": f"{type(ex).__name__}: {str(ex)[:200]}"})

    def log_message(self, *args) -> None:  # silence per-request logs
        pass


if __name__ == "__main__":
    print(f"serving on 0.0.0.0:{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
