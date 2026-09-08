from dataclasses import dataclass, replace
from typing import Final

from mt_lnn.config import MTLNNConfig
from mt_lnn.profiles import M2Experiment, m2_research_config

TRAINABLE_EXPERIMENTS: Final = (
    "baseline", "selective_state", "fast_weight_memory", "latent_core",
    "latent_stack", "workspace", "competitive_workspace", "predictive_coding",
    "world_model", "hamiltonian_world_model", "global_rhythm", "global_coherence",
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
    corpus: str | None = None
    sequence_length: int = 128
    size: str = "probe"
    grad_accum: int = 1

    def __post_init__(self) -> None:
        if self.task not in ("pointer_chase", "mod_chain", "text"):
            raise ValueError(f"Unsupported task: {self.task}")
        if self.experiment not in TRAINABLE_EXPERIMENTS:
            raise ValueError(f"Experiment has no verified training wiring: {self.experiment}")
        if self.batch < 1 or self.difficulty < 1 or self.n_values < 2 or self.lr <= 0:
            raise ValueError("Batch/difficulty/lr must be positive; n_values >= 2")
        if self.task == "pointer_chase" and self.difficulty >= self.n_values:
            raise ValueError("Single-cycle pointer difficulty must be below n_values")
        if (self.task == "text") != (self.corpus is not None):
            raise ValueError("A corpus manifest is required only for text training")
        if self.sequence_length < 2 or self.grad_accum < 1:
            raise ValueError("sequence_length >= 2 and grad_accum >= 1 are required")
        if self.size not in ("probe", "2b"):
            raise ValueError("Unknown model size")
        if self.task == "text" and self.experiment not in ("baseline", "selective_state"):
            raise ValueError("Text training supports baseline/selective_state only; causal loop paths need separate validation")


def model_config(recipe: Recipe, vocab: int, length: int) -> MTLNNConfig:
    experiments = () if recipe.experiment == "baseline" else (M2Experiment(recipe.experiment),)
    width, layers, heads, kv = (104, 2, 4, 2) if recipe.size == "probe" else (2080, 34, 16, 4)
    return replace(
        m2_research_config(experiments),
        vocab_size=vocab, max_seq_len=length, d_model=width, n_layers=layers,
        n_heads=heads, n_kv_heads=kv, d_head=width // heads, gwtb_n_heads=1,
        dropout=0.0, attention_dropout=0.0,
    )
