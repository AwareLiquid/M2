from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from benchmarks.reasoning_tasks import make_generator
from mt_lnn.model import MTLNNModel

from .recipe import Recipe, model_config


class TrainingRun:
    def __init__(self, recipe: Recipe, device: str = "cpu") -> None:
        self.recipe = recipe
        self.device = torch.device(device)
        torch.manual_seed(recipe.seed)
        self.rng = np.random.default_rng(recipe.seed)
        self.generator, vocab, _ = make_generator(
            recipe.task, recipe.difficulty, recipe.n_values, recipe.seed,
        )
        sample = self.generator(1, np.random.default_rng(recipe.seed))
        self.model = MTLNNModel(model_config(recipe, vocab, sample.tokens.shape[1])).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=recipe.lr, betas=(0.9, 0.95))
        self.step = 0

    def train_until(self, total_steps: int) -> float | None:
        if total_steps < self.step:
            raise ValueError(f"Total steps {total_steps} is below restored step {self.step}")
        self.model.train()
        last_loss = None
        while self.step < total_steps:
            batch = self.generator(self.recipe.batch, self.rng)
            ids = torch.from_numpy(batch.tokens).to(self.device)
            labels = torch.full_like(ids, -100)
            labels[:, batch.ans_pos] = torch.from_numpy(batch.answer).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(ids, labels=labels)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {self.step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.step += 1
            last_loss = loss.detach().item()
        return last_loss

    @torch.no_grad()
    def evaluate(self, batches: int = 4) -> float:
        if batches < 1:
            raise ValueError("Evaluation batches must be positive")
        self.model.eval()
        rng = np.random.default_rng(self.recipe.seed + 1_000_000)
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
