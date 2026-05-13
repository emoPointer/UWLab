from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Pose:
    """World pose in Isaac/MuJoCo convention: xyz + wxyz quaternion."""

    position: np.ndarray
    quaternion: np.ndarray

    @classmethod
    def from_sequence(cls, values) -> "Pose":
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != 7:
            raise ValueError(f"Expected pose with 7 values, got shape {array.shape}")
        return cls(position=array[..., :3].copy(), quaternion=array[..., 3:7].copy())


@dataclass(frozen=True)
class IsaacRenderState:
    """Minimal authoritative state copied from Isaac/PhysX for MuJoCo rendering."""

    robot_root: Pose | None
    joint_positions: dict[str, float]
    object_poses: dict[str, Pose] = field(default_factory=dict)
    camera_poses: dict[str, Pose] = field(default_factory=dict)
    step: int | None = None
