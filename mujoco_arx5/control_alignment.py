from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


CONFIG_PATH = Path(__file__).with_name("config") / "control_alignment.toml"


def _load_config() -> dict:
    return tomllib.loads(CONFIG_PATH.read_text())


def control_period_seconds() -> float:
    """Return the Isaac-aligned policy/control period."""

    cfg = _load_config()
    return float(cfg["timing"]["control_dt"])


def scale_arm_action(action: Iterable[float], *, mode: str = "eval") -> list[float]:
    """Scale normalized xyz+axis-angle action using the current Isaac ARX5 OSC scales."""

    values = [float(value) for value in action]
    if len(values) != 6:
        raise ValueError(f"ARX5 arm action must have 6 values, got {len(values)}.")

    cfg = _load_config()["arm"]
    if mode == "eval":
        pos_scale = float(cfg["eval_position_scale"])
        rot_scale = float(cfg["eval_orientation_scale"])
    elif mode == "train":
        pos_scale = float(cfg["train_position_scale"])
        rot_scale = float(cfg["train_orientation_scale"])
    else:
        raise ValueError(f"Unsupported control mode {mode!r}; expected 'eval' or 'train'.")

    return [
        values[0] * pos_scale,
        values[1] * pos_scale,
        values[2] * pos_scale,
        values[3] * rot_scale,
        values[4] * rot_scale,
        values[5] * rot_scale,
    ]


def binary_gripper_targets(action: float | bool) -> dict[str, float]:
    """Map IsaacLab BinaryJointPositionAction sign convention to ARX5 gripper targets."""

    cfg = _load_config()["gripper"]
    if isinstance(action, bool):
        command = cfg["close"] if action is False else cfg["open"]
    else:
        command = cfg["close"] if float(action) < 0.0 else cfg["open"]
    return {joint_name: float(target) for joint_name, target in command.items()}

