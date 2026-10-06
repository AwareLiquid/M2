import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from m2_training.recipe import Recipe
from m2_training.runner import TrainingRun
from m2_training.sft_prepare import prepare, record


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    lines = [json.dumps({"prompt": f"Question {i}?", "response": f"Answer {i}."})
             for i in range(12)]
    source = tmp_path / "sft.jsonl"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return prepare(source, tmp_path / "corpus_sft", sequence_length=32,
                   validation_fraction=0.25, seed=0)


def test_record_layout_and_mask() -> None:
    tokens, mask = record(b"Q", b"A", 8)
    # prompt | \n\n | response | \n  ->  Q \n \n A \n
    assert tokens[:5].tolist() == [ord("Q"), 10, 10, ord("A"), 10]
    assert mask[:5].tolist() == [False, False, False, True, True]
    assert not mask[5:].any()


def test_record_truncation_keeps_response_after_separator() -> None:
    tokens, mask = record(b"x" * 100, b"R" * 30, 16)
    # response trimmed to 16 - len("\n\n") - len("\n") = 13 bytes; prompt dropped
    assert len(mask) == 16
    assert mask[2:16].all()
    assert not mask[:2].any()
    assert tokens[2:15].tolist() == [ord("R")] * 13


def test_sft_supervises_only_response_positions(corpus: Path) -> None:
    run = TrainingRun(Recipe(task="sft", corpus=str(corpus), sequence_length=32, batch=2))
    ids, labels = run.batch(np.random.default_rng(0))
    supervised = labels != -100
    assert supervised.any() and (~supervised).any()
    assert torch.equal(labels[supervised], ids[supervised])
    for row in range(ids.shape[0]):
        first = int(supervised[row].nonzero()[0])
        assert ids[row, first - 2:first].tolist() == [10, 10]


def test_sft_resume_matches_continuous_with_gradient_accumulation(corpus: Path, tmp_path: Path) -> None:
    recipe = Recipe(task="sft", corpus=str(corpus), sequence_length=32, batch=2, grad_accum=2)
    continuous = TrainingRun(recipe)
    continuous.train_until(4)
    segmented = TrainingRun(recipe)
    segmented.train_until(2)
    path = tmp_path / "sft.pt"
    segmented.save(path)
    restored = TrainingRun.restore(path)
    restored.train_until(4)
    for name, expected in continuous.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], expected, rtol=0, atol=0)
    assert restored.evaluate() == continuous.evaluate()
    assert restored.evaluate() > 1


def test_modified_sft_split_is_rejected_on_resume(corpus: Path, tmp_path: Path) -> None:
    run = TrainingRun(Recipe(task="sft", corpus=str(corpus), sequence_length=32))
    path = tmp_path / "sft.pt"
    run.save(path)
    with (corpus.parent / "train.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        TrainingRun.restore(path)


def test_prepare_rejects_empty_response(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"prompt": "Q", "response": ""}\n'
                      '{"prompt": "Q", "response": "ok"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty response"):
        prepare(source, tmp_path / "out", sequence_length=16)


def test_resume_can_switch_task_to_sft_with_corpus_override(tmp_path: Path) -> None:
    """后训练入口：基座（task=text）检查点 → 以 --task sft + 新 corpus 续训。"""
    from m2_training.prepare import prepare as prepare_text
    text_in = tmp_path / "base.txt"
    val_in = tmp_path / "val.txt"
    text_in.write_text("base corpus text " * 40, encoding="utf-8")
    val_in.write_text("held out base text " * 40, encoding="utf-8")
    text_manifest = prepare_text(text_in, val_in, tmp_path / "text_corpus")

    lines = [json.dumps({"prompt": f"q{i}?", "response": f"a{i}."})
             for i in range(8)]
    source = tmp_path / "sft.jsonl"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sft_manifest = prepare(source, tmp_path / "sft_corpus", sequence_length=16)

    ckpt = tmp_path / "base.pt"
    run = TrainingRun(Recipe(task="text", corpus=str(text_manifest),
                             sequence_length=16, batch=2))
    run.train_until(2)
    run.save(ckpt)

    restored = TrainingRun.restore(ckpt, corpus=str(sft_manifest), task="sft")
    assert restored.recipe.task == "sft"
    restored.train_until(4)
    assert restored.step == 4
    assert restored.evaluate() > 1
    assert restored.batch(np.random.default_rng(0))[0].shape[0] == 2


def test_resume_accepts_post_training_overrides(tmp_path: Path) -> None:
    """--lr/--sequence-length/--rope-scale 覆盖 = 后训练档位（Run 2 口径）。"""
    from m2_training.prepare import prepare as prepare_text
    text_in = tmp_path / "base.txt"
    val_in = tmp_path / "val.txt"
    text_in.write_text("base corpus text " * 40, encoding="utf-8")
    val_in.write_text("held out base text " * 40, encoding="utf-8")
    text_manifest = prepare_text(text_in, val_in, tmp_path / "text_corpus")

    lines = [json.dumps({"prompt": f"question {i}?", "response": f"reply {i}."})
             for i in range(8)]
    source = tmp_path / "sft.jsonl"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sft_manifest = prepare(source, tmp_path / "sft_corpus", sequence_length=32)

    ckpt = tmp_path / "base.pt"
    run = TrainingRun(Recipe(task="text", corpus=str(text_manifest),
                             sequence_length=16, batch=2))
    run.train_until(2)
    run.save(ckpt)

    restored = TrainingRun.restore(
        ckpt, corpus=str(sft_manifest), task="sft", lr=2e-5,
        sequence_length=32, rope_scale=4.0)
    assert restored.recipe.task == "sft"
    assert restored.recipe.lr == 2e-5
    assert restored.recipe.sequence_length == 32
    assert restored.recipe.rope_scale == 4.0
    ids, labels = restored.batch(np.random.default_rng(0))
    assert ids.shape[1] == 32
    assert (labels != -100).any()
    restored.train_until(3)
    assert restored.step == 3


def test_rope_scale_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="rope_scale"):
        Recipe(task="sft", corpus="x.json", rope_scale=0.5)


def test_sft_cli_trains_resumes_and_evaluates(tmp_path: Path) -> None:
    source = tmp_path / "sft.jsonl"
    source.write_text("\n".join(
        json.dumps({"prompt": f"prompt {i}", "response": f"reply {i}"})
        for i in range(8)) + "\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    subprocess.run(
        [sys.executable, "-m", "m2_training.sft_prepare", "--input", str(source),
         "--output", str(corpus), "--sequence-length", "16"],
        check=True, capture_output=True, env=environment,
    )
    checkpoint = tmp_path / "sft.pt"
    base = [sys.executable, "-m", "m2_training"]
    result = None
    for command in (
        ["train", "--checkpoint", str(checkpoint), "--task", "sft", "--corpus",
         str(corpus / "manifest.json"), "--sequence-length", "16", "--batch", "2",
         "--steps", "2"],
        ["resume", "--checkpoint", str(checkpoint), "--steps", "4"],
        ["evaluate", "--checkpoint", str(checkpoint)],
    ):
        result = subprocess.run(base + command, check=True, capture_output=True,
                                env=environment, text=True)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["step"] == 4
    assert payload["task"] == "sft"
    assert payload["validation_ppl"] > 1
