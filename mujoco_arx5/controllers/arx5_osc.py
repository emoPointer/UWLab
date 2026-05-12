from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_arx5.control_alignment import binary_gripper_targets, scale_arm_action


ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
ARM_ACTUATOR_NAMES = tuple(f"{joint_name}_torque" for joint_name in ARM_JOINT_NAMES)
GRIPPER_ACTUATOR_NAMES = ("joint7_position", "joint8_position")
EE_BODY_NAME = "link6"

TRAIN_STIFFNESS = np.array([200.0, 200.0, 200.0, 3.0, 3.0, 3.0], dtype=np.float64)
TRAIN_DAMPING_RATIO = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0], dtype=np.float64)
EVAL_STIFFNESS = np.array([1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0], dtype=np.float64)
EVAL_DAMPING_RATIO = np.ones(6, dtype=np.float64)
TORQUE_LIMIT = 50.0


@dataclass(frozen=True)
class PoseTarget:
    """End-effector target held for one MuJoCo control step."""

    position: np.ndarray
    quaternion: np.ndarray
    gripper: dict[str, float]


@dataclass(frozen=True)
class ControllerOutput:
    """Computed command values written to ``data.ctrl``."""

    torque: np.ndarray
    pose_error: np.ndarray
    ee_velocity: np.ndarray
    gripper: dict[str, float]


class Arx5OperationalSpaceController:
    """MuJoCo ARX5 OSC matching the current UWLab train/eval action contract.

    The policy action is seven dimensional: relative xyz+axis-angle target for
    ``link6`` followed by one binary gripper scalar.  The arm controller mirrors
    IsaacLab's fixed-gain ``OperationalSpaceControllerAction`` closely enough for
    control-alignment work: relative pose target, zero desired EE velocity,
    full operational-space inertia decoupling, no gravity compensation, and no
    null-space command.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        mode: str = "train",
        ee_body_name: str = EE_BODY_NAME,
        torque_limit: float = TORQUE_LIMIT,
    ) -> None:
        self.model = model
        self.mode = mode
        self.ee_body_name = ee_body_name
        self.torque_limit = float(torque_limit)

        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported ARX5 OSC mode {mode!r}; expected 'train' or 'eval'.")

        self.arm_joint_names = ARM_JOINT_NAMES
        self.arm_actuator_names = ARM_ACTUATOR_NAMES
        self.gripper_actuator_names = GRIPPER_ACTUATOR_NAMES
        self.arm_joint_ids = self._resolve_ids(mujoco.mjtObj.mjOBJ_JOINT, self.arm_joint_names)
        self.arm_actuator_ids = self._resolve_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, self.arm_actuator_names)
        self.gripper_actuator_ids = self._resolve_ids(mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_actuator_names)
        self.ee_body_id = self._resolve_id(mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

        self.arm_dof_ids = np.array([int(model.jnt_dofadr[joint_id]) for joint_id in self.arm_joint_ids], dtype=np.int32)
        self.arm_qpos_ids = np.array(
            [int(model.jnt_qposadr[joint_id]) for joint_id in self.arm_joint_ids], dtype=np.int32
        )
        self.position_scale = 0.02 if mode == "train" else 0.01
        self.orientation_scale = 0.2
        self.motion_stiffness = (TRAIN_STIFFNESS if mode == "train" else EVAL_STIFFNESS).copy()
        damping_ratio = TRAIN_DAMPING_RATIO if mode == "train" else EVAL_DAMPING_RATIO
        self.motion_damping = 2.0 * np.sqrt(self.motion_stiffness) * damping_ratio

        self._target: PoseTarget | None = None
        self.raw_action = np.zeros(7, dtype=np.float64)
        self.processed_arm_action = np.zeros(6, dtype=np.float64)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def target(self) -> PoseTarget | None:
        return self._target

    def reset(self, data: mujoco.MjData) -> None:
        """Hold the current end-effector pose and open the gripper."""

        pose = self.get_ee_pose(data)
        self.raw_action[:] = 0.0
        self.processed_arm_action[:] = 0.0
        self._target = PoseTarget(
            position=pose.position.copy(),
            quaternion=pose.quaternion.copy(),
            gripper=binary_gripper_targets(0.0),
        )

    def get_ee_pose(self, data: mujoco.MjData) -> PoseTarget:
        mujoco.mj_forward(self.model, data)
        return PoseTarget(
            position=np.asarray(data.xpos[self.ee_body_id], dtype=np.float64).copy(),
            quaternion=np.asarray(data.xquat[self.ee_body_id], dtype=np.float64).copy(),
            gripper={},
        )

    def set_target_from_action(self, data: mujoco.MjData, action: np.ndarray | list[float] | tuple[float, ...]) -> PoseTarget:
        action_array = self._as_action_array(action)
        self.raw_action[:] = action_array
        self.processed_arm_action[:] = scale_arm_action(action_array[:6], mode=self.mode)

        current_pose = self.get_ee_pose(data)
        delta_quat = _axis_angle_to_quat(self.processed_arm_action[3:6])
        target = PoseTarget(
            position=current_pose.position + self.processed_arm_action[:3],
            quaternion=_normalize_quat(_quat_mul(delta_quat, current_pose.quaternion)),
            gripper=binary_gripper_targets(float(action_array[6])),
        )
        self._target = target
        return target

    def apply_action(self, data: mujoco.MjData, action: np.ndarray | list[float] | tuple[float, ...]) -> ControllerOutput:
        """Process a new policy action, compute OSC torque, and write MuJoCo ctrl.

        Call this once per policy/control step.  For lower-level MuJoCo physics
        substeps in the same decimation window, call :meth:`apply_target` so the
        relative action target is held fixed instead of being re-applied.
        """

        target = self.set_target_from_action(data, action)
        return self.apply_target(data, target)

    def apply_target(self, data: mujoco.MjData, target: PoseTarget | None = None) -> ControllerOutput:
        """Track the current held target and write MuJoCo ctrl."""

        output = self.compute(data, target)
        data.ctrl[self.arm_actuator_ids] = output.torque
        data.ctrl[self.gripper_actuator_ids[0]] = output.gripper["joint7"]
        data.ctrl[self.gripper_actuator_ids[1]] = output.gripper["joint8"]
        return output

    def compute(self, data: mujoco.MjData, target: PoseTarget | None = None) -> ControllerOutput:
        """Compute arm torque for the current state and target without stepping."""

        if target is None:
            if self._target is None:
                self.reset(data)
            target = self._target
        assert target is not None

        mujoco.mj_forward(self.model, data)
        current_pose = self.get_ee_pose(data)
        jacobian = self._body_jacobian(data)
        ee_velocity = jacobian @ np.asarray(data.qvel, dtype=np.float64)
        pose_error = np.concatenate(
            [
                target.position - current_pose.position,
                _axis_angle_from_quat(_quat_mul(target.quaternion, _quat_inv(current_pose.quaternion))),
            ]
        )
        desired_acc = self.motion_stiffness * pose_error - self.motion_damping * ee_velocity
        wrench = self._operational_space_inertia(data, jacobian) @ desired_acc
        torque_full = jacobian.T @ wrench
        torque = np.clip(torque_full[self.arm_dof_ids], -self.torque_limit, self.torque_limit)
        return ControllerOutput(
            torque=torque,
            pose_error=pose_error,
            ee_velocity=ee_velocity,
            gripper=target.gripper.copy(),
        )

    def _body_jacobian(self, data: mujoco.MjData) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, data, jacp, jacr, self.ee_body_id)
        return np.vstack([jacp, jacr])

    def _operational_space_inertia(self, data: mujoco.MjData, jacobian: np.ndarray) -> np.ndarray:
        mass_matrix = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, mass_matrix, data.qM)
        mass_matrix_inv = np.linalg.pinv(mass_matrix)
        return np.linalg.pinv(jacobian @ mass_matrix_inv @ jacobian.T, rcond=1.0e-6)

    def _resolve_ids(self, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        return np.array([self._resolve_id(obj_type, name) for name in names], dtype=np.int32)

    def _resolve_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        item_id = mujoco.mj_name2id(self.model, obj_type, name)
        if item_id < 0:
            raise ValueError(f"MuJoCo model does not contain {obj_type.name} named {name!r}.")
        return int(item_id)

    def _as_action_array(self, action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.action_dim,):
            raise ValueError(f"ARX5 MuJoCo controller action must have shape ({self.action_dim},), got {action_array.shape}.")
        return action_array


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(quat, dtype=np.float64) / norm


def _axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle <= 1.0e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.asarray(axis_angle, dtype=np.float64) / angle
    half = 0.5 * angle
    return _normalize_quat(np.concatenate([[np.cos(half)], axis * np.sin(half)]))


def _axis_angle_from_quat(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat(quat)
    if q[0] < 0.0:
        q = -q
    mag = float(np.linalg.norm(q[1:]))
    half_angle = np.arctan2(mag, q[0])
    angle = 2.0 * half_angle
    if abs(angle) <= 1.0e-6:
        sin_half_over_angle = 0.5 - angle * angle / 48.0
    else:
        sin_half_over_angle = np.sin(half_angle) / angle
    return q[1:] / sin_half_over_angle


def _quat_inv(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    conjugate = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    norm_sq = float(np.dot(q, q))
    if norm_sq < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return conjugate / norm_sq


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
