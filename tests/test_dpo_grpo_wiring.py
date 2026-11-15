"""DPO/GRPO 接线端到端集成测试 — 真实执行训练步(合成数据小模型)。

队列条目 dpo-grpo-wiring 的验收: 损失数学(dpo_grpo.py)之外, rl 路径必须
真的流过 TrainingRun 的训练循环(前向/参考模型/反传/优化器/checkpoint),
且默认路径(SFT 主线)不受影响。
"""

import pytest
import torch

from m2_training.recipe import Recipe
from m2_training.runner import TrainingRun


def _recipe(rl_mode=""):
    return Recipe(task="pointer_chase", difficulty=2, n_values=8, seed=0,
                  batch=4, rl_mode=rl_mode)


@pytest.mark.parametrize("rl_mode", ["dpo", "grpo"])
def test_rl_path_executes_real_training_steps(rl_mode):
    run = TrainingRun(_recipe(rl_mode), device="cpu")
    assert run.ref_model is not None
    assert all(not p.requires_grad for p in run.ref_model.parameters())
    last = run.train_until(3)  # 三次真实 rl 梯度步(前向+反传+optimizer.step)
    assert run.step == 3
    assert last is not None and last == last and last < float("inf")


def test_default_path_unaffected_by_rl_wiring():
    run = TrainingRun(_recipe(), device="cpu")
    assert run.ref_model is None  # 默认关: 无参考模型、零额外开销
    run.train_until(2)
    assert run.step == 2


def test_rl_policy_moves_off_frozen_reference():
    run = TrainingRun(_recipe("dpo"), device="cpu")
    run.train_until(2)
    ref = torch.cat([p.detach().flatten() for p in run.ref_model.parameters()])
    pol = torch.cat([p.detach().flatten() for p in run.model.parameters()])
    assert not torch.equal(ref, pol)  # 训练只动策略, 参考冻结在初始快照


def test_rl_checkpoint_round_trip(tmp_path):
    run = TrainingRun(_recipe("grpo"), device="cpu")
    run.train_until(1)
    path = tmp_path / "ckpt.pt"
    run.save(path)
    restored = TrainingRun.restore(path, device="cpu")
    assert restored.ref_model is not None
    for key, value in run.ref_model.state_dict().items():
        assert torch.equal(restored.ref_model.state_dict()[key], value)


def test_restore_rejects_missing_ref_state(tmp_path):
    run = TrainingRun(_recipe("dpo"), device="cpu")
    run.train_until(1)
    path = tmp_path / "ckpt.pt"
    run.save(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state.pop("rl_ref_model")
    torch.save(state, path)
    with pytest.raises(ValueError, match="rl reference model"):
        TrainingRun.restore(path, device="cpu")


def test_recipe_rejects_unknown_rl_mode():
    with pytest.raises(ValueError, match="rl_mode"):
        _recipe("ppo")
    with pytest.raises(ValueError, match="synthetic"):
        Recipe(task="text", corpus="m.json", rl_mode="dpo")
