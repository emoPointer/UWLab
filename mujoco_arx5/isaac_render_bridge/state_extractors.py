from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .state import IsaacRenderState, Pose


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


def _select_env(value, env_index: int) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 0:
        return array
    return array[env_index]


def _pose_from_asset_data(data, env_index: int) -> Pose:
    if hasattr(data, "root_pose_w"):
        return Pose.from_sequence(_select_env(data.root_pose_w, env_index))
    if hasattr(data, "root_state_w"):
        return Pose.from_sequence(_select_env(data.root_state_w, env_index)[:7])
    if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
        return Pose(
            position=_select_env(data.root_pos_w, env_index).copy(),
            quaternion=_select_env(data.root_quat_w, env_index).copy(),
        )
    raise AttributeError("Asset data does not expose root_pose_w, root_state_w, or root_pos_w/root_quat_w")


def _pose_from_camera_data(data, env_index: int) -> Pose:
    if hasattr(data, "pos_w") and hasattr(data, "quat_w_world"):
        return Pose(
            position=_select_env(data.pos_w, env_index).copy(),
            quaternion=_select_env(data.quat_w_world, env_index).copy(),
        )
    if hasattr(data, "pos_w") and hasattr(data, "quat_w"):
        return Pose(
            position=_select_env(data.pos_w, env_index).copy(),
            quaternion=_select_env(data.quat_w, env_index).copy(),
        )
    return _pose_from_asset_data(data, env_index)


class IsaacLabStateExtractor:
    """Extract render state from an IsaacLab environment without owning physics.

    The extractor intentionally uses the public scene/data shape common to
    IsaacLab assets and sensors, so it can be tested without launching Isaac Sim.
    """

    def __init__(
        self,
        *,
        robot_name: str = "robot",
        object_names: Sequence[str] = ("insertive_object", "receptive_object"),
        camera_names: Sequence[str] = (),
        joint_names: Sequence[str] | None = None,
        env_index: int = 0,
    ):
        self.robot_name = robot_name
        self.object_names = tuple(object_names)
        self.camera_names = tuple(camera_names)
        self.joint_names = tuple(joint_names) if joint_names is not None else None
        self.env_index = env_index

    def extract(self, env, *, step: int | None = None) -> IsaacRenderState:
        scene = env.unwrapped.scene if hasattr(env, "unwrapped") else env.scene
        robot = scene[self.robot_name]
        robot_data = robot.data

        robot_root = _pose_from_asset_data(robot_data, self.env_index)
        joint_positions = self._extract_joint_positions(robot, self.env_index)
        object_poses = {
            name: _pose_from_asset_data(scene[name].data, self.env_index)
            for name in self.object_names
            if self._scene_has(scene, name)
        }
        camera_poses = {
            name: _pose_from_camera_data(scene.sensors[name].data, self.env_index)
            for name in self.camera_names
            if hasattr(scene, "sensors") and name in scene.sensors
        }
        return IsaacRenderState(
            robot_root=robot_root,
            joint_positions=joint_positions,
            object_poses=object_poses,
            camera_poses=camera_poses,
            step=step,
        )

    def _extract_joint_positions(self, robot, env_index: int) -> dict[str, float]:
        joint_pos = _select_env(robot.data.joint_pos, env_index)
        if self.joint_names is None:
            names = tuple(robot.joint_names)
            indices = tuple(range(len(names)))
        elif hasattr(robot, "find_joints"):
            indices, names = robot.find_joints(list(self.joint_names))
        else:
            names = self.joint_names
            robot_joint_names = tuple(robot.joint_names)
            indices = tuple(robot_joint_names.index(name) for name in names)
        if len(names) != len(indices):
            raise ValueError(f"Joint name/index mismatch: {names} vs {indices}")
        return {name: float(joint_pos[index]) for name, index in zip(names, indices)}

    @staticmethod
    def _scene_has(scene, name: str) -> bool:
        if isinstance(scene, Mapping):
            return name in scene
        try:
            scene[name]
        except (KeyError, AttributeError):
            return False
        return True
