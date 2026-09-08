from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from benchmarks.reasoning_tasks import make_generator
from mt_lnn.model import MTLNNModel

from .recipe import Recipe, model_config
from .text_data import Corpus


class TrainingRun:
    def __init__(self, recipe: Recipe, device: str = "cpu") -> None:
        self.recipe = recipe
        self.device = torch.device(device)
        torch.manual_seed(recipe.seed)
        self.rng = np.random.default_rng(recipe.seed)
        self.corpus = Corpus.open(recipe.corpus, recipe.sequence_length) if recipe.corpus else None
        if self.corpus is not None:
            vocab, length = self.corpus.vocab, recipe.sequence_length + 1
        else:
            self.generator, vocab, _ = make_generator(
                recipe.task, recipe.difficulty, recipe.n_values, recipe.seed,
            )
            length = self.generator(1, np.random.default_rng(recipe.seed)).tokens.shape[1]
        self.model = MTLNNModel(model_config(recipe, vocab, length)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=recipe.lr, betas=(0.9, 0.95), weight_decay=0.01)
        self.step = 0

    def train_until(self, total_steps: int) -> float | None:
        if total_steps < self.step:
            raise ValueError(f"Total steps {total_steps} is below restored step {self.step}")
        self.model.train()
        last_loss = None
        while self.step < total_steps:
            self.optimizer.zero_grad(set_to_none=True)
            last_loss = 0.0
            for _ in range(self.recipe.grad_accum):
                ids, labels = self.batch(self.rng)
                loss = self.model(ids, labels=labels)["loss"] / self.recipe.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {self.step}")
                loss.backward()
                last_loss += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.step += 1
        return last_loss

    def batch(self, rng: np.random.Generator, *, validation: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if self.corpus is not None:
            tokens = self.corpus.sample(rng, self.recipe.batch, self.recipe.sequence_length, validation=validation)
            ids = torch.from_numpy(tokens).to(self.device)
            return ids, ids.clone()
        batch = self.generator(self.recipe.batch, rng)
        ids = torch.from_numpy(batch.tokens).to(self.device)
        labels = torch.full_like(ids, -100)
        labels[:, batch.ans_pos] = torch.from_numpy(batch.answer).to(self.device)
        return ids, labels

    @torch.no_grad()
    def evaluate(self, batches: int = 4) -> float:
        if batches < 1:
            raise ValueError("Evaluation batches must be positive")
        self.model.eval()
        rng = np.random.default_rng(self.recipe.seed + 1_000_000)
        if self.corpus is not None:
            total_loss = 0.0
            for _ in range(batches):
                ids, labels = self.batch(rng, validation=True)
                total_loss += self.model(ids, labels=labels)["loss"].item()
            return math.exp(total_loss / batches)
        correct = 0
        for _ in range(batches):
            batch = self.generator(self.recipe.batch, rng)
            logits = self.model(torch.from_numpy(batch.tokens).to(self.device))["logits"]
            predicted = logits[:, batch.ans_pos - 1].argmax(-1).cpu()
            correct += int((predicted == torch.from_numpy(batch.answer)).sum())
        return correct / (batches * self.recipe.batch)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 1, "recipe": asdict(self.recipe), "step": self.step,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "data_rng": json.dumps(self.rng.bit_generator.state),
            "torch_rng": torch.get_rng_state(), "device_type": self.device.type,
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "corpus_identity": self.corpus.identity if self.corpus else None,
        }
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        os.close(handle)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def restore(cls, path: Path, device: str = "cpu") -> TrainingRun:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format_version"] != 1:
            raise ValueError("Unsupported checkpoint version")
        run = cls(Recipe(**state["recipe"]), device)
        if state.get("corpus_identity") != (run.corpus.identity if run.corpus else None):
            raise ValueError("Checkpoint corpus identity differs from current data")
        if state["device_type"] != run.device.type:
            raise ValueError("Resume requires the same device type for RNG reproducibility")
        run.model.load_state_dict(state["model"], strict=True)
        run.optimizer.load_state_dict(state["optimizer"])
        run.rng.bit_generator.state = json.loads(state["data_rng"])
        torch.set_rng_state(state["torch_rng"])
        if run.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        run.step = state["step"]
        return run
