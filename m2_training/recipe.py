from dataclasses import dataclass, replace
from typing import Final

from mt_lnn.config import MTLNNConfig
from mt_lnn.profiles import M2Experiment, m2_research_config

TRAINABLE_EXPERIMENTS: Final = (
    "baseline", "selective_state", "latent_core", "latent_stack", "workspace",
)


@dataclass(frozen=True)
class Recipe:
    task: str = "pointer_chase"
    experiment: str = "baseline"
    seed: int = 0
    batch: int = 8
    difficulty: int = 2
    n_values: int = 8
    lr: float = 0.0003

    def __post_init__(self) -> None:
        if self.task not in ("pointer_chase", "mod_chain"):
            raise ValueError(f"Unsupported task: {self.task}")
        if self.experiment not in TRAINABLE_EXPERIMENTS:
            raise ValueError(f"Experiment has no verified training wiring: {self.experiment}")
        if self.batch < 1 or self.difficulty < 1 or self.n_values < 2 or self.lr <= 0:
            raise ValueError("Batch/difficulty/lr must be positive; n_values >= 2")
        if self.task == "pointer_chase" and self.difficulty >= self.n_values:
            raise ValueError("Single-cycle pointer difficulty must be below n_values")


def model_config(recipe: Recipe, vocab: int, length: int) -> MTLNNConfig:
    experiments = () if recipe.experiment == "baseline" else (M2Experiment(recipe.experiment),)
    return replace(
        m2_research_config(experiments),
        vocab_size=vocab, max_seq_len=length, d_model=104, n_layers=2,
        n_heads=4, n_kv_heads=2, d_head=26, gwtb_n_heads=1,
        dropout=0.0, attention_dropout=0.0,
    )
