from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from benchmarks.reasoning_tasks import make_generator
from mt_lnn.model import MTLNNModel

from .recipe import Recipe, model_config
from .sft_data import SFTData
from .text_data import Corpus


class TrainingRun:
    def __init__(self, recipe: Recipe, device: str = "cpu") -> None:
        self.recipe = recipe
        self.device = torch.device(device)
        torch.manual_seed(recipe.seed)
        self.rng = np.random.default_rng(recipe.seed)
        self.corpus = (Corpus.open(recipe.corpus, recipe.sequence_length)
                       if recipe.corpus and recipe.task == "text" else None)
        self.sft = (SFTData.open(recipe.corpus, recipe.sequence_length)
                    if recipe.task == "sft" else None)
        if self.sft is not None:
            vocab, length = self.sft.vocab, recipe.sequence_length + 1
        elif self.corpus is not None:
            vocab, length = self.corpus.vocab, recipe.sequence_length + 1
        else:
            self.generator, vocab, _ = make_generator(
                recipe.task, recipe.difficulty, recipe.n_values, recipe.seed,
            )
            length = self.generator(1, np.random.default_rng(recipe.seed)).tokens.shape[1]
        self.model = MTLNNModel(model_config(recipe, vocab, length)).to(self.device)
        # 2B 实配（训练机验证）：CUDA 下 bf16 + bitsandbytes AdamW8bit 让
        # 2B 装进 40G 卡；CPU/CI 保持原 fp32 + AdamW（位等价、测试口径不变）。
        if self.device.type == "cuda":
            self.model = self.model.to(torch.bfloat16)
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(), lr=recipe.lr, betas=(0.9, 0.95),
                weight_decay=0.01)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=recipe.lr, betas=(0.9, 0.95),
                weight_decay=0.01)
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

    @property
    def data_identity(self) -> str | None:
        if self.sft is not None:
            return self.sft.identity
        if self.corpus is not None:
            return self.corpus.identity
        return None

    def batch(self, rng: np.random.Generator, *, validation: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sft is not None:
            ids, labels = self.sft.sample(rng, self.recipe.batch, validation=validation)
            return (torch.from_numpy(ids).to(self.device),
                    torch.from_numpy(labels).to(self.device))
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
        if self.corpus is not None or self.sft is not None:
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
        if self.device.type == "cuda":
            # 2B 实配（训练机）：bf16 权重落盘；bnb8bit 状态不序列化
            # （2B 检查点 3.9G 已很大，优化器态随 resume 重新累积）。
            model_state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                           for k, v in self.model.state_dict().items()}
            optimizer_state: dict = {}
        else:
            # CPU/CI：完整状态，分段训练与连续训练位等价（测试口径）
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()
        state = {
            "format_version": 1, "recipe": asdict(self.recipe), "step": self.step,
            "model": model_state, "optimizer": optimizer_state,
            "data_rng": json.dumps(self.rng.bit_generator.state),
            "torch_rng": torch.get_rng_state(), "device_type": self.device.type,
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "corpus_identity": self.data_identity,
        }
        if path.exists():
            path.unlink()
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        os.close(handle)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def restore(cls, path: Path, device: str = "cpu", corpus: str | None = None) -> TrainingRun:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format_version"] != 1:
            raise ValueError("Unsupported checkpoint version")
        recipe = Recipe(**state["recipe"])
        if corpus is not None:
            recipe = replace(recipe, corpus=str(Path(corpus).resolve()))
        run = cls(recipe, device)
        if corpus is None and state.get("corpus_identity") != run.data_identity:
            raise ValueError("Checkpoint corpus identity differs from current data")
        if state["device_type"] != run.device.type:
            raise ValueError("Resume requires the same device type for RNG reproducibility")
        run.model.load_state_dict(state["model"], strict=True)
        if state.get("optimizer"):
            run.optimizer.load_state_dict(state["optimizer"])
        run.rng.bit_generator.state = json.loads(state["data_rng"])
        torch.set_rng_state(state["torch_rng"])
        if run.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        run.step = state["step"]
        return run
