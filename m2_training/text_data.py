import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Corpus:
    train: np.ndarray
    validation: np.ndarray
    vocab: int
    identity: str

    @classmethod
    def open(cls, manifest: str, sequence_length: int) -> "Corpus":
        path = Path(manifest)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["format_version"] != 1 or not metadata["tokenizer"]:
            raise ValueError("Unsupported corpus manifest or missing tokenizer identity")
        vocab = int(metadata["vocab_size"])
        arrays = []
        for split in ("train", "validation"):
            file = path.parent / metadata[split]["file"]
            if fingerprint(file) != metadata[split]["sha256"]:
                raise ValueError(f"Corpus checksum mismatch: {split}")
            tokens = np.load(file, mmap_mode="r", allow_pickle=False)
            if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
                raise ValueError("Corpus must contain one-dimensional integer token arrays")
            if tokens.size < sequence_length + 1 or tokens.min() < 0 or tokens.max() >= vocab:
                raise ValueError("Corpus is too short or contains tokens outside the vocabulary")
            arrays.append(tokens)
        if metadata["train"]["sha256"] == metadata["validation"]["sha256"]:
            raise ValueError("Train and validation splits must differ")
        identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        return cls(arrays[0], arrays[1], vocab, identity)

    def sample(self, rng: np.random.Generator, batch: int, length: int, *, validation: bool = False) -> np.ndarray:
        tokens = self.validation if validation else self.train
        starts = rng.integers(0, tokens.size - length, size=batch)
        return np.stack([tokens[start:start + length + 1] for start in starts]).astype(np.int64)
