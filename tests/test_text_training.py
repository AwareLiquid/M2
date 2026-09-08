import importlib
from pathlib import Path

import pytest
import torch

from m2_training.prepare import prepare
from m2_training.recipe import Recipe
from m2_training.runner import TrainingRun


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("æ¨¡åž‹è®­ç»ƒ uses separate documents. " * 20, encoding="utf-8")
    validation.write_text("è¿™æ˜¯è¯„ä¼°æ–‡æœ¬ã€‚Hold out validation. " * 20, encoding="utf-8")
    return prepare(train, validation, tmp_path / "corpus")


def test_text_resume_matches_continuous_with_gradient_accumulation(corpus: Path, tmp_path: Path) -> None:
    recipe = Recipe(task="text", corpus=str(corpus), sequence_length=8, batch=2, grad_accum=2)
    continuous = TrainingRun(recipe)
    continuous.train_until(4)
    segmented = TrainingRun(recipe)
    segmented.train_until(2)
    path = tmp_path / "resume.pt"
    segmented.save(path)
    restored = TrainingRun.restore(path)
    restored.train_until(4)
    for name, expected in continuous.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], expected, rtol=0, atol=0)
    assert restored.evaluate() == continuous.evaluate()
    assert restored.evaluate() > 1


def test_modified_corpus_is_rejected_on_resume(corpus: Path, tmp_path: Path) -> None:
    run = TrainingRun(Recipe(task="text", corpus=str(corpus), sequence_length=8))
    path = tmp_path / "resume.pt"
    run.save(path)
    with (corpus.parent / "train.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        TrainingRun.restore(path)


@pytest.mark.parametrize("name", ["world_model", "gwtb", "global_coherence", "rhythm", "hamiltonian_head", "predictive_coding", "hebbian_plasticity"])
def test_research_modules_keep_legacy_module_identity(name: str) -> None:
    assert importlib.import_module(f"mt_lnn.{name}") is importlib.import_module(f"mt_lnn.research.{name}")
