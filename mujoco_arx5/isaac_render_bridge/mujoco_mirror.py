from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .state import IsaacRenderState, Pose


class MuJoCoRenderMirror:
    """Mirror Isaac state into MuJoCo for rendering only.

    This class never calls ``mj_step``. Isaac/PhysX remains the authoritative
    simulator; MuJoCo is only forwarded kinematically before rendering.
    """

    def __init__(
        self,
        model,
        data,
        *,
        object_body_map: Mapping[str, str] | None = None,
        camera_body_map: Mapping[str, str] | None = None,
        robot_root_body: str = "arx5",
    ):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.robot_root_body = robot_root_body
        self.object_body_map = dict(
            object_body_map
            or {
                "insertive_object": "insertive_cube",
                "receptive_object": "receptive_cube",
            }
        )
        self.camera_body_map = dict(camera_body_map or {})
        self._joint_qpos_addrs = self._build_joint_qpos_addrs()
        self._body_ids = self._build_body_ids(
            [robot_root_body, *self.object_body_map.values(), *self.camera_body_map.values()]
        )

    @classmethod
    def from_xml_path(cls, xml_path: str | Path, **kwargs) -> "MuJoCoRenderMirror":
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        return cls(model, data, **kwargs)

    def apply(self, state: IsaacRenderState) -> None:
        if state.robot_root is not None:
            self._set_body_pose(self.robot_root_body, state.robot_root)

        for joint_name, joint_value in state.joint_positions.items():
            qpos_addr = self._joint_qpos_addrs.get(joint_name)
            if qpos_addr is not None:
                self.data.qpos[qpos_addr] = joint_value

        for isaac_name, pose in state.object_poses.items():
            body_name = self.object_body_map.get(isaac_name)
            if body_name is not None:
                self._set_body_pose(body_name, pose)

        if state.camera_poses and self.camera_body_map:
            # Robot FK must be current before converting world camera poses into
            # local nested-body transforms.
            self._mujoco.mj_forward(self.model, self.data)

        for isaac_name, pose in state.camera_poses.items():
            body_name = self.camera_body_map.get(isaac_name)
            if body_name is not None:
                self._set_body_world_pose(body_name, pose)

        self._mujoco.mj_forward(self.model, self.data)

    def _build_joint_qpos_addrs(self) -> dict[str, int]:
        addrs: dict[str, int] = {}
        for joint_id in range(self.model.njnt):
            name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                continue
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == self._mujoco.mjtJoint.mjJNT_FREE:
                continue
            addrs[name] = int(self.model.jnt_qposadr[joint_id])
        return addrs

    def _build_body_ids(self, body_names: list[str]) -> dict[str, int]:
        body_ids: dict[str, int] = {}
        for body_name in body_names:
            body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"MuJoCo body not found: {body_name}")
            body_ids[body_name] = body_id
        return body_ids

    def randomize_light_angles(
        self,
        *,
        rng: np.random.Generator | None = None,
        yaw_range_deg: tuple[float, float] = (0.0, 360.0),
        elevation_range_deg: tuple[float, float] = (35.0, 75.0),
        distance_range: tuple[float, float] = (2.0, 3.0),
        target: tuple[float, float, float] = (-0.3, -0.2, 0.84),
        key_light_name: str = "key_light",
        fill_light_name: str = "fill_light",
    ) -> dict[str, np.ndarray]:
        """Randomize MuJoCo light directions around a target point."""
        rng = rng or np.random.default_rng()
        yaw = np.deg2rad(float(rng.uniform(*yaw_range_deg)))
        elevation = np.deg2rad(float(rng.uniform(*elevation_range_deg)))
        distance = float(rng.uniform(*distance_range))
        target_np = np.asarray(target, dtype=np.float64)

        key_pos = target_np + distance * self._unit_vector_from_yaw_elevation(yaw, elevation)
        self._set_light_pose(key_light_name, key_pos, target_np)

        fill_yaw = yaw + np.pi + float(rng.uniform(-0.45, 0.45))
        fill_elevation = max(np.deg2rad(15.0), elevation * 0.65)
        fill_pos = target_np + 0.75 * distance * self._unit_vector_from_yaw_elevation(fill_yaw, fill_elevation)
        self._set_light_pose(fill_light_name, fill_pos, target_np)

        return {key_light_name: key_pos, fill_light_name: fill_pos}

    def _set_light_pose(self, light_name: str, position: np.ndarray, target: np.ndarray) -> None:
        light_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_LIGHT, light_name)
        if light_id < 0:
            return
        direction = target - position
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError(f"Light {light_name} cannot point at itself.")
        self.model.light_pos[light_id] = position
        self.model.light_dir[light_id] = direction / norm

    @staticmethod
    def _unit_vector_from_yaw_elevation(yaw: float, elevation: float) -> np.ndarray:
        return np.array(
            [
                np.cos(elevation) * np.cos(yaw),
                np.cos(elevation) * np.sin(yaw),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )

    def _set_body_pose(self, body_name: str, pose: Pose) -> None:
        body_id = self._body_ids[body_name]
        position = np.asarray(pose.position, dtype=np.float64)
        quaternion = self._normalized_quat(pose.quaternion)
        if position.shape != (3,):
            raise ValueError(f"Body {body_name} position must have shape (3,), got {position.shape}")
        self.model.body_pos[body_id] = position
        self.model.body_quat[body_id] = quaternion

    def _set_body_world_pose(self, body_name: str, pose: Pose) -> None:
        body_id = self._body_ids[body_name]
        position = np.asarray(pose.position, dtype=np.float64)
        quaternion = self._normalized_quat(pose.quaternion)
        if position.shape != (3,):
            raise ValueError(f"Body {body_name} position must have shape (3,), got {position.shape}")

        parent_id = int(self.model.body_parentid[body_id])
        parent_pos = np.asarray(self.data.xpos[parent_id], dtype=np.float64)
        parent_quat = self._normalized_quat(self.data.xquat[parent_id])
        inv_parent_quat = self._quat_conjugate(parent_quat)

        local_pos = self._quat_rotate(inv_parent_quat, position - parent_pos)
        local_quat = self._quat_multiply(inv_parent_quat, quaternion)
        self.model.body_pos[body_id] = local_pos
        self.model.body_quat[body_id] = self._normalized_quat(local_quat)

    @staticmethod
    def _normalized_quat(quaternion) -> np.ndarray:
        quat = np.asarray(quaternion, dtype=np.float64)
        if quat.shape != (4,):
            raise ValueError(f"Quaternion must have shape (4,), got {quat.shape}")
        norm = np.linalg.norm(quat)
        if norm == 0.0:
            raise ValueError("Quaternion has zero norm")
        return quat / norm

    @staticmethod
    def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
        return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], dtype=np.float64)

    @classmethod
    def _quat_multiply(cls, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )

    @classmethod
    def _quat_rotate(cls, quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        pure = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
        rotated = cls._quat_multiply(cls._quat_multiply(quaternion, pure), cls._quat_conjugate(quaternion))
        return rotated[1:]
