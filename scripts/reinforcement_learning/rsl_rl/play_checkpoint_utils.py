"""Checkpoint loading helpers for RSL-RL play/deploy runs."""

from __future__ import annotations

import torch


_ACTOR_INFERENCE_PREFIXES = (
    "actor.",
    "actor_obs_normalizer.",
    "student.",
    "student_obs_normalizer.",
)
_ACTOR_INFERENCE_KEYS = ("std", "log_std")
_SKIPPABLE_MISMATCH_PREFIXES = (
    "critic.",
    "critic_obs_normalizer.",
    "teacher.",
    "teacher_obs_normalizer.",
)


def _policy_from_runner(runner):
    if hasattr(runner.alg, "policy"):
        return runner.alg.policy
    return runner.alg.actor_critic


def _is_actor_inference_key(key: str) -> bool:
    return key in _ACTOR_INFERENCE_KEYS or key.startswith(_ACTOR_INFERENCE_PREFIXES)


def _is_skippable_play_mismatch(key: str) -> bool:
    return key.startswith(_SKIPPABLE_MISMATCH_PREFIXES)


def _load_play_compatible_state_dict(runner, checkpoint_path: str):
    loaded_dict = torch.load(checkpoint_path, weights_only=False, map_location=getattr(runner, "device", "cpu"))
    checkpoint_state = loaded_dict["model_state_dict"]
    policy = _policy_from_runner(runner)
    current_state = policy.state_dict()

    compatible_state = {}
    skipped = []
    actor_mismatches = []
    unsupported_mismatches = []
    for key, checkpoint_tensor in checkpoint_state.items():
        current_tensor = current_state.get(key)
        if current_tensor is None:
            skipped.append((key, "missing in current policy"))
            continue
        if current_tensor.shape == checkpoint_tensor.shape:
            compatible_state[key] = checkpoint_tensor
            continue
        reason = f"checkpoint {tuple(checkpoint_tensor.shape)} != current {tuple(current_tensor.shape)}"
        if _is_actor_inference_key(key):
            actor_mismatches.append((key, reason))
        elif _is_skippable_play_mismatch(key):
            skipped.append((key, reason))
        else:
            unsupported_mismatches.append((key, reason))

    if actor_mismatches:
        details = "\n".join(f"  - {key}: {reason}" for key, reason in actor_mismatches)
        raise RuntimeError(
            "Cannot load checkpoint for play: actor/inference tensors are incompatible.\n" + details
        )
    if unsupported_mismatches:
        details = "\n".join(f"  - {key}: {reason}" for key, reason in unsupported_mismatches)
        raise RuntimeError("Cannot load checkpoint for play: unsupported tensor shape mismatches.\n" + details)

    missing_actor_keys = [
        key
        for key in current_state
        if _is_actor_inference_key(key) and key not in compatible_state and key not in checkpoint_state
    ]
    if missing_actor_keys:
        details = "\n".join(f"  - {key}" for key in missing_actor_keys)
        raise RuntimeError("Cannot load checkpoint for play: actor/inference tensors are missing.\n" + details)

    policy.load_state_dict(compatible_state, strict=False)
    if "iter" in loaded_dict:
        runner.current_learning_iteration = loaded_dict["iter"]

    if skipped:
        print("[INFO] Play checkpoint fallback skipped non-inference tensors:")
        for key, reason in skipped:
            print(f"  - {key}: {reason}")
    return loaded_dict.get("infos", {})


def load_runner_checkpoint_for_play(runner, checkpoint_path: str):
    """Load a checkpoint for visualization/deploy.

    RSL-RL strictly loads actor and critic tensors together. For play runs the actor is
    the only network used for inference, so this falls back to loading compatible actor
    tensors when critic-only observation dimensions changed.
    """

    try:
        return runner.load(checkpoint_path)
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        print("[WARN] Strict RSL-RL checkpoint load failed during play; trying actor-compatible fallback.")
        print(f"[WARN] Original load error: {exc}")
        return _load_play_compatible_state_dict(runner, checkpoint_path)
