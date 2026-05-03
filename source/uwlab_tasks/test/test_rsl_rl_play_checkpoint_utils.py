from types import SimpleNamespace
import importlib.util

import pytest
import torch


REPO_ROOT = __import__("pathlib").Path(__file__).parents[3]
CHECKPOINT_UTILS = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "play_checkpoint_utils.py"


def _load_utils_module():
    spec = importlib.util.spec_from_file_location("play_checkpoint_utils", CHECKPOINT_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Normalizer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, dim))
        self.register_buffer("std", torch.ones(1, dim))


class _Policy(torch.nn.Module):
    def __init__(self, actor_dim=3, critic_dim=5):
        super().__init__()
        self.actor = torch.nn.Linear(actor_dim, 2)
        self.critic = torch.nn.Linear(critic_dim, 1)
        self.actor_obs_normalizer = _Normalizer(actor_dim)
        self.critic_obs_normalizer = _Normalizer(critic_dim)


class _StrictFailRunner:
    def __init__(self, policy):
        self.alg = SimpleNamespace(policy=policy)
        self.current_learning_iteration = 0

    def load(self, path):
        raise RuntimeError("size mismatch for critic.weight")


def test_play_checkpoint_load_skips_mismatched_critic_only_tensors(tmp_path):
    utils = _load_utils_module()
    policy = _Policy(actor_dim=3, critic_dim=5)
    runner = _StrictFailRunner(policy)

    checkpoint_state = policy.state_dict()
    checkpoint_state["actor.weight"] = torch.full_like(checkpoint_state["actor.weight"], 7.0)
    checkpoint_state["actor.bias"] = torch.full_like(checkpoint_state["actor.bias"], 8.0)
    checkpoint_state["critic.weight"] = torch.ones(1, 7)
    checkpoint_state["critic_obs_normalizer.mean"] = torch.zeros(1, 7)
    checkpoint_state["critic_obs_normalizer.std"] = torch.ones(1, 7)

    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": checkpoint_state, "iter": 42, "infos": {"loaded": True}}, checkpoint_path)

    infos = utils.load_runner_checkpoint_for_play(runner, str(checkpoint_path))

    assert infos == {"loaded": True}
    assert runner.current_learning_iteration == 42
    assert torch.all(policy.actor.weight == 7.0)
    assert torch.all(policy.actor.bias == 8.0)


def test_play_checkpoint_load_rejects_actor_shape_mismatch(tmp_path):
    utils = _load_utils_module()
    policy = _Policy(actor_dim=3, critic_dim=5)
    runner = _StrictFailRunner(policy)

    checkpoint_state = policy.state_dict()
    checkpoint_state["actor.weight"] = torch.ones(2, 4)
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": checkpoint_state}, checkpoint_path)

    with pytest.raises(RuntimeError, match="actor/inference"):
        utils.load_runner_checkpoint_for_play(runner, str(checkpoint_path))
