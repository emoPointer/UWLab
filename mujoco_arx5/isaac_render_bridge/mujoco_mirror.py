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

        for isaac_name, pose in state.camera_poses.items():
            body_name = self.camera_body_map.get(isaac_name)
            if body_name is not None:
                self._set_body_pose(body_name, pose)

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

    def _set_body_pose(self, body_name: str, pose: Pose) -> None:
        body_id = self._body_ids[body_name]
        position = np.asarray(pose.position, dtype=np.float64)
        quaternion = np.asarray(pose.quaternion, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"Body {body_name} position must have shape (3,), got {position.shape}")
        if quaternion.shape != (4,):
            raise ValueError(f"Body {body_name} quaternion must have shape (4,), got {quaternion.shape}")
        norm = np.linalg.norm(quaternion)
        if norm == 0.0:
            raise ValueError(f"Body {body_name} quaternion has zero norm")
        self.model.body_pos[body_id] = position
        self.model.body_quat[body_id] = quaternion / norm
