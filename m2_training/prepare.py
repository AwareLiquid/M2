import argparse
import json
from pathlib import Path

import numpy as np

from .text_data import fingerprint


def prepare(train: Path, validation: Path, output: Path) -> Path:
    if train.resolve() == validation.resolve():
        raise ValueError("Use separate training and validation text files")
    raw_train = train.read_text(encoding="utf-8").encode("utf-8")
    raw_validation = validation.read_text(encoding="utf-8").encode("utf-8")
    if len(raw_train) < 2 or len(raw_validation) < 2 or raw_train == raw_validation:
        raise ValueError("Text splits must be nonempty and different")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"format_version": 1, "tokenizer": "utf8-byte-v1", "vocab_size": 256}
    for name, raw in (("train", raw_train), ("validation", raw_validation)):
        path = output / f"{name}.npy"
        np.save(path, np.frombuffer(raw, dtype=np.uint8))
        metadata[name] = {"file": path.name, "sha256": fingerprint(path)}
    path = output / "manifest.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare separate UTF-8 text splits using byte tokens")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.train, args.validation, args.output))


if __name__ == "__main__":
    main()
