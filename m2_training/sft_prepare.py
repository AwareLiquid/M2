"""Build SFT records from a JSONL of {"prompt", "response"} pairs.

Layout per record (utf8-byte-v1 tokens):
    prompt | b"\\n\\n" | response | b"\\n"
The loss mask covers the response bytes and the trailing newline; the prompt
and separator are context only. Records are truncated/padded to
--sequence-length: the response is trimmed from the RIGHT first, then the
prompt is trimmed from the LEFT (keeping the context nearest the response).
Padding uses byte 0 with mask False, after the trailing newline, so it can
only affect forward context, never a supervised position.

The validation split is --validation-fraction of the records, shuffled with
--seed. Manifest + npz mirror text_data.Corpus (sha256-checked on open).

    python -m m2_training.sft_prepare --input data.jsonl --output corpus_sft \\
        --sequence-length 512
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .text_data import fingerprint

SEPARATOR = b"\n\n"
EOS = b"\n"


def record(prompt: bytes, response: bytes, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length < 8:
        raise ValueError("sequence_length must be >= 8 to hold separator, eos and bytes")
    if not response:
        raise ValueError("SFT records need a non-empty response")
    response = response[: sequence_length - len(SEPARATOR) - len(EOS)]
    prompt_budget = sequence_length - len(response) - len(SEPARATOR) - len(EOS)
    prompt = prompt[-prompt_budget:] if prompt_budget > 0 else b""
    body = prompt + SEPARATOR + response + EOS
    tokens = np.zeros(sequence_length, dtype=np.int64)
    tokens[: len(body)] = np.frombuffer(body, dtype=np.uint8)
    mask = np.zeros(sequence_length, dtype=bool)
    response_start = len(prompt) + len(SEPARATOR)
    mask[response_start:len(body)] = True
    return tokens, mask


def _read_records(input_path: Path) -> list[tuple[bytes, bytes]]:
    if not input_path.exists():
        raise ValueError(f"Input JSONL not found: {input_path}")
    parsed = []
    for number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Line {number} is not valid JSON") from error
        if not isinstance(item, dict) or "response" not in item:
            raise ValueError(f"Line {number} needs a 'response' field")
        prompt = item.get("prompt", "")
        if not isinstance(prompt, str) or not isinstance(item["response"], str):
            raise ValueError(f"Line {number}: prompt/response must be strings")
        parsed.append((prompt.encode("utf-8"), item["response"].encode("utf-8")))
    if len(parsed) < 2:
        raise ValueError("Need at least two records to form train and validation splits")
    return parsed


def prepare(input_path: Path, output: Path, sequence_length: int = 512,
            validation_fraction: float = 0.02, seed: int = 0) -> Path:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    records = _read_records(input_path)
    order = np.random.default_rng(seed).permutation(len(records))
    n_validation = max(1, int(round(len(records) * validation_fraction)))
    if n_validation >= len(records):
        raise ValueError("Validation split would leave no training records")
    validation_idx, train_idx = order[:n_validation], order[n_validation:]
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "format_version": 1, "tokenizer": "utf8-byte-v1", "vocab_size": 256,
        "sequence_length": sequence_length,
        "records": {"train": len(train_idx), "validation": len(validation_idx)},
    }
    for name, index in (("train", train_idx), ("validation", validation_idx)):
        tokens = np.zeros((len(index), sequence_length), dtype=np.int64)
        mask = np.zeros((len(index), sequence_length), dtype=bool)
        for row, record_index in enumerate(index):
            tokens[row], mask[row] = record(*records[record_index], sequence_length)
        path = output / f"{name}.npz"
        np.savez_compressed(path, tokens=tokens, mask=mask)
        metadata[name] = {"file": path.name, "sha256": fingerprint(path)}
    path = output / "manifest.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT records with response-only loss masks")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = prepare(args.input, args.output, args.sequence_length,
                   args.validation_fraction, args.seed)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"manifest": str(path), "records": metadata["records"],
                      "sequence_length": metadata["sequence_length"]}))


if __name__ == "__main__":
    main()
