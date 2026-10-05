"""SFT records with response-only loss masks (manifest format v1).

Companion to text_data.Corpus: same byte-level tokenizer, but records are
independent (prompt | separator | response) instead of one flat stream, and
the training loss covers only the response bytes. Batches are drawn by record
index, so every batch is uniform over records rather than over byte offsets.

Manifest:
    {"format_version": 1, "tokenizer": "utf8-byte-v1", "vocab_size": 256,
     "sequence_length": T, "records": {"train": N, "validation": M},
     "train": {"file": "train.npz", "sha256": ...},
     "validation": {"file": "validation.npz", "sha256": ...}}
Each npz stores `tokens` (N, T) int and `mask` (N, T) bool — mask True marks
positions whose next-token prediction is trained. Labels are built as
tokens where mask, else -100, matching the model's shifted loss convention:
label position t is trained from logit position t-1, so the first response
byte (label index response_start) is predicted from the separator context.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .text_data import fingerprint


@dataclass(frozen=True)
class SFTData:
    train_tokens: np.ndarray
    train_mask: np.ndarray
    validation_tokens: np.ndarray
    validation_mask: np.ndarray
    vocab: int
    sequence_length: int
    identity: str

    @classmethod
    def open(cls, manifest: str, sequence_length: int) -> "SFTData":
        path = Path(manifest)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or metadata.get("tokenizer") != "utf8-byte-v1":
            raise ValueError("Unsupported SFT manifest or tokenizer")
        vocab = int(metadata["vocab_size"])
        if int(metadata["sequence_length"]) != sequence_length:
            raise ValueError("SFT manifest sequence_length differs from the recipe")
        splits = []
        for split in ("train", "validation"):
            file = path.parent / metadata[split]["file"]
            if fingerprint(file) != metadata[split]["sha256"]:
                raise ValueError(f"SFT checksum mismatch: {split}")
            with np.load(file, allow_pickle=False) as archive:
                tokens = archive["tokens"]
                mask = archive["mask"]
            if tokens.ndim != 2 or tokens.shape != mask.shape:
                raise ValueError("SFT arrays must be matching (N, T) matrices")
            if tokens.shape[1] != sequence_length or tokens.shape[0] < 1:
                raise ValueError("SFT record length differs from sequence_length")
            if tokens.min() < 0 or tokens.max() >= vocab:
                raise ValueError("SFT tokens outside the vocabulary")
            if not mask.any():
                raise ValueError(f"SFT {split} split has no supervised positions")
            splits.append((tokens, mask))
        if metadata["train"]["sha256"] == metadata["validation"]["sha256"]:
            raise ValueError("Train and validation splits must differ")
        identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        return cls(splits[0][0], splits[0][1], splits[1][0], splits[1][1], vocab,
                   sequence_length, identity)

    def sample(self, rng: np.random.Generator, batch: int, *,
               validation: bool = False) -> tuple[np.ndarray, np.ndarray]:
        tokens, mask = ((self.validation_tokens, self.validation_mask) if validation
                        else (self.train_tokens, self.train_mask))
        index = rng.integers(0, tokens.shape[0], size=batch)
        ids = tokens[index].astype(np.int64)
        labels = np.where(mask[index], ids, -100)
        return ids, labels
