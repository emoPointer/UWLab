"""Small helpers for reset-state visualization scripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def resolve_gripper_joint_indices(joint_names: Sequence[str], close_command_expr: Mapping[str, float]) -> list[int]:
    """Return articulation joint indices controlled by a binary gripper action."""
    indices: list[int] = []
    for command_joint_name in close_command_expr:
        try:
            indices.append(joint_names.index(command_joint_name))
        except ValueError as exc:
            raise ValueError(
                f"Gripper command joint '{command_joint_name}' is not present in articulation joints: {joint_names}"
            ) from exc
    return indices
