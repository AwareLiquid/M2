import json
import os
import subprocess
import sys
from pathlib import Path


def test_text_cli_prepares_trains_resumes_and_evaluates(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("training text " * 20, encoding="utf-8")
    validation.write_text("different held out text " * 20, encoding="utf-8")
    corpus = tmp_path / "corpus"
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    subprocess.run(
        [sys.executable, "-m", "m2_training.prepare", "--train", str(train),
         "--validation", str(validation), "--output", str(corpus)],
        check=True, capture_output=True, env=environment,
    )
    checkpoint = tmp_path / "state.pt"
    base = [sys.executable, "-m", "m2_training"]
    commands = [
        ["train", "--checkpoint", str(checkpoint), "--task", "text", "--corpus",
         str(corpus / "manifest.json"), "--sequence-length", "8", "--batch", "2", "--steps", "2"],
        ["resume", "--checkpoint", str(checkpoint), "--steps", "4"],
        ["evaluate", "--checkpoint", str(checkpoint)],
    ]
    results = []
    for command in commands:
        result = subprocess.run(base + command, check=True, capture_output=True, text=True, env=environment)
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert [result["step"] for result in results] == [2, 4, 4]
    assert results[1]["validation_ppl"] == results[2]["validation_ppl"]
    rejected = subprocess.run(base + commands[0], capture_output=True, env=environment)
    assert rejected.returncode == 2
