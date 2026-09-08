from pathlib import Path

import pytest
import torch

from m2_training.recipe import Recipe
from m2_training.runner import TrainingRun


@pytest.mark.parametrize("experiment", ["baseline", "selective_state", "fast_weight_memory", "latent_core", "latent_stack", "workspace", "competitive_workspace", "predictive_coding", "world_model", "hamiltonian_world_model", "global_rhythm", "global_coherence", "top_down"])
def test_segmented_training_matches_uninterrupted(experiment: str, tmp_path: Path) -> None:
    continuous = TrainingRun(Recipe(experiment=experiment, batch=2))
    continuous.train_until(4)
    segmented = TrainingRun(Recipe(experiment=experiment, batch=2))
    segmented.train_until(2)
    path = tmp_path / "state.pt"
    segmented.save(path)
    resumed = TrainingRun.restore(path)
    assert resumed.step == 2
    resumed.train_until(4)
    for name, expected in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], expected, rtol=0, atol=0)
    assert resumed.evaluate() == continuous.evaluate()


def test_unwired_hebbian_is_rejected() -> None:
    with pytest.raises(ValueError, match="no verified training wiring"):
        Recipe(experiment="hebbian")
