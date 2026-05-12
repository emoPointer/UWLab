"""Checkpoint loading helpers for RSL-RL play/deploy and training runs."""

from __future__ import annotations

import math
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


def _load_compatible_state_dict(
    runner,
    checkpoint_path: str,
    *,
    advance_iteration: bool,
    allow_noise_reinit: bool = False,
):
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
        if allow_noise_reinit and key in _ACTOR_INFERENCE_KEYS:
            skipped.append((key, reason))
        elif _is_actor_inference_key(key):
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
    if advance_iteration and "iter" in loaded_dict:
        runner.current_learning_iteration = loaded_dict["iter"]

    if skipped:
        print("[INFO] Checkpoint fallback skipped non-inference tensors:")
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
        return _load_compatible_state_dict(runner, checkpoint_path, advance_iteration=True)


def load_runner_checkpoint_for_training(runner, checkpoint_path: str):
    """Warm-start training from a checkpoint when only critic observation tensors changed."""

    try:
        return runner.load(checkpoint_path)
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        print("[WARN] Strict RSL-RL checkpoint load failed during training; trying actor warm-start fallback.")
        print(f"[WARN] Original load error: {exc}")
        print("[WARN] Optimizer state and incompatible tensors will be re-initialized for the current environment.")
        return _load_compatible_state_dict(runner, checkpoint_path, advance_iteration=False, allow_noise_reinit=True)


def freeze_policy_action_noise(policy, fixed_noise_std: float) -> None:
    """Set and freeze a non-state-dependent RSL-RL action noise parameter."""

    if fixed_noise_std <= 0.0:
        raise ValueError(f"fixed_noise_std must be positive, got {fixed_noise_std}.")
    if getattr(policy, "state_dependent_std", False):
        raise RuntimeError("Freezing action noise is only supported for non-state-dependent std policies.")

    if hasattr(policy, "log_std"):
        with torch.no_grad():
            policy.log_std.fill_(math.log(fixed_noise_std))
        policy.log_std.requires_grad_(False)
        print(f"[INFO] Frozen policy.log_std at std={fixed_noise_std:.6g}.")
    elif hasattr(policy, "std"):
        with torch.no_grad():
            policy.std.fill_(fixed_noise_std)
        policy.std.requires_grad_(False)
        print(f"[INFO] Frozen policy.std at std={fixed_noise_std:.6g}.")
    else:
        raise RuntimeError("Policy does not expose a supported action noise parameter: expected log_std or std.")


def freeze_runner_action_noise(runner, fixed_noise_std: float) -> None:
    """Set and freeze the action noise parameter for a runner policy."""

    freeze_policy_action_noise(_policy_from_runner(runner), fixed_noise_std)
