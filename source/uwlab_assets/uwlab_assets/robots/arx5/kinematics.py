# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical kinematics for ARX5.

Unlike UR5e (all joints rotate about Z), ARX5 has mixed joint axes
(Z/Y/Y/Y/Z/X) plus pi-flips on joint3 and joint6's fixed rpy. So the
single-axis rotation trick used by ur5e_robotiq_gripper/kinematics.py
is replaced with Rodrigues rotation about an arbitrary axis stored in
``metadata.yaml -> calibrated_joints.axis``.

Exports (mirroring UR5e's kinematics.py):
- ``ARM_JOINT_NAMES``, ``EE_BODY_NAME``, ``NUM_ARM_JOINTS``
- ``compute_jacobian_analytical(joint_angles, device)`` -> (N, 6, 6)
- ``compute_mass_matrix_analytical(joint_angles, device)`` -> (N, 6, 6)

Frame convention: the chain starts at ARX5 ``base_link`` (no REP-103
base_link_inertia quirk like UR5e). The EE body is ``link6`` and the
Jacobian is computed at link6's origin.
"""

import functools
import os
import tempfile
import torch
import yaml

from isaaclab.utils.assets import retrieve_file_path

# ============================================================================
# Constants
# ============================================================================

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
EE_BODY_NAME = "link6"
NUM_ARM_JOINTS = 6


# ============================================================================
# Lazy-loaded calibration data
# ============================================================================


@functools.lru_cache(maxsize=1)
def _load_calibration() -> dict[str, torch.Tensor]:
    """Parse calibrated kinematics from the ARX5 metadata.yaml (lazy, cached)."""
    from .arx5 import ARX5_USD_PATH

    usd_dir = os.path.dirname(ARX5_USD_PATH)
    meta_path = os.path.join(usd_dir, "metadata.yaml")
    local = retrieve_file_path(meta_path, download_dir=tempfile.gettempdir())
    with open(local) as f:
        metadata = yaml.safe_load(f)
    if metadata is None:
        raise RuntimeError(f"metadata.yaml is empty or failed to load: {local}")
    if "calibrated_joints" not in metadata or "axis" not in metadata["calibrated_joints"]:
        raise RuntimeError(
            f"ARX5 metadata.yaml at {local} is missing 'calibrated_joints' block "
            "with an 'axis' field (required because ARX5 has mixed joint axes)."
        )
    joints = metadata["calibrated_joints"]
    inertials = metadata["link_inertials"]
    return {
        "joints_xyz": torch.tensor(joints["xyz"], dtype=torch.float32),
        "joints_rpy": torch.tensor(joints["rpy"], dtype=torch.float32),
        "joints_axis": torch.tensor(joints["axis"], dtype=torch.float32),
        "link_masses": torch.tensor(inertials["masses"], dtype=torch.float32),
        "link_coms": torch.tensor(inertials["coms"], dtype=torch.float32),
        "link_inertias": torch.tensor(inertials["inertias"], dtype=torch.float32),
    }


# ============================================================================
# Rotation helpers
# ============================================================================


def rpy_to_matrix_torch(rpy: torch.Tensor) -> torch.Tensor:
    """Convert single roll-pitch-yaw (XYZ intrinsic, same as URDF) to 3x3 matrix."""
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R = torch.stack([
        torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
        torch.stack([-sp, cp * sr, cp * cr]),
    ])
    return R


def rodrigues_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotation matrix about `axis` by angle `theta` (batched over theta).

    Args:
        axis: (3,) local rotation axis (will be normalized).
        theta: (N,) batch of angles in radians.
    Returns:
        R: (N, 3, 3).
    """
    N = theta.shape[0]
    device, dtype = theta.device, theta.dtype
    a = axis.to(device=device, dtype=dtype)
    a = a / (torch.norm(a) + 1e-12)
    ax, ay, az = a[0], a[1], a[2]
    ct = torch.cos(theta)
    st = torch.sin(theta)
    C = 1.0 - ct

    R = torch.empty(N, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = ct + ax * ax * C
    R[:, 0, 1] = ax * ay * C - az * st
    R[:, 0, 2] = ax * az * C + ay * st
    R[:, 1, 0] = ay * ax * C + az * st
    R[:, 1, 1] = ct + ay * ay * C
    R[:, 1, 2] = ay * az * C - ax * st
    R[:, 2, 0] = az * ax * C - ay * st
    R[:, 2, 1] = az * ay * C + ax * st
    R[:, 2, 2] = ct + az * az * C
    return R


def _build_fixed_transforms(
    xyz_all: torch.Tensor, rpy_all: torch.Tensor, N: int, device: str
) -> list[torch.Tensor]:
    """Build the per-joint (constant) fixed transform T_fixed_i expanded to (N, 4, 4)."""
    T_fixed_all = []
    for i in range(NUM_ARM_JOINTS):
        R_fixed = rpy_to_matrix_torch(rpy_all[i])
        T = torch.eye(4, device=device, dtype=torch.float32)
        T[:3, :3] = R_fixed
        T[:3, 3] = xyz_all[i]
        T_fixed_all.append(T.unsqueeze(0).expand(N, -1, -1).clone())
    return T_fixed_all


def _joint_transform(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """4x4 homogeneous transform for a revolute joint rotating by `theta` about `axis`."""
    N = theta.shape[0]
    T = torch.eye(4, device=theta.device, dtype=theta.dtype).unsqueeze(0).repeat(N, 1, 1)
    T[:, :3, :3] = rodrigues_rotation_matrix(axis, theta)
    return T


# ============================================================================
# Analytical FK  (exposed mainly for tests; callers usually want the Jacobian)
# ============================================================================


def compute_fk_link6(joint_angles: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Forward kinematics to link6 in base_link frame.

    Args:
        joint_angles: (N, 6) joint angles in radians.
    Returns:
        T: (N, 4, 4) homogeneous transform base_link -> link6.
    """
    N = joint_angles.shape[0]
    cal = _load_calibration()
    xyz_all = cal["joints_xyz"].to(device)
    rpy_all = cal["joints_rpy"].to(device)
    axis_all = cal["joints_axis"].to(device)

    T_fixed_all = _build_fixed_transforms(xyz_all, rpy_all, N, device)
    T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
    for i in range(NUM_ARM_JOINTS):
        T_joint_frame = torch.bmm(T, T_fixed_all[i])
        T_joint = _joint_transform(axis_all[i], joint_angles[:, i])
        T = torch.bmm(T_joint_frame, T_joint)
    return T


# ============================================================================
# Analytical Jacobian
# ============================================================================


def compute_jacobian_analytical(joint_angles: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Geometric Jacobian at link6 origin, expressed in base_link frame.

    Args:
        joint_angles: (N, 6) joint angles in radians.
    Returns:
        J: (N, 6, 6) stacked [linear (3); angular (3)] per joint column.
    """
    N = joint_angles.shape[0]
    cal = _load_calibration()
    xyz_all = cal["joints_xyz"].to(device)
    rpy_all = cal["joints_rpy"].to(device)
    axis_all = cal["joints_axis"].to(device)

    T_fixed_all = _build_fixed_transforms(xyz_all, rpy_all, N, device)

    # --- Pass 1: full FK to get p_ee ---
    T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
    for i in range(NUM_ARM_JOINTS):
        T_joint_frame = torch.bmm(T, T_fixed_all[i])
        T_joint = _joint_transform(axis_all[i], joint_angles[:, i])
        T = torch.bmm(T_joint_frame, T_joint)
    p_ee = T[:, :3, 3]

    # --- Pass 2: per-joint columns ---
    J = torch.zeros(N, 6, 6, device=device, dtype=torch.float32)
    T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
    for i in range(NUM_ARM_JOINTS):
        T_joint_frame = torch.bmm(T, T_fixed_all[i])
        R_frame = T_joint_frame[:, :3, :3]
        axis_world = torch.bmm(R_frame, axis_all[i].view(1, 3, 1).expand(N, -1, -1)).squeeze(-1)
        p_i = T_joint_frame[:, :3, 3]
        J[:, :3, i] = torch.cross(axis_world, p_ee - p_i, dim=1)
        J[:, 3:, i] = axis_world
        T_joint = _joint_transform(axis_all[i], joint_angles[:, i])
        T = torch.bmm(T_joint_frame, T_joint)

    return J


# ============================================================================
# Analytical Mass Matrix (CRBA)
# ============================================================================


def compute_mass_matrix_analytical(joint_angles: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """6x6 joint-space mass matrix at the current configuration.

    Uses the URDF inertial parameters stored in metadata.yaml for consistency
    with analytical Jacobian-based controllers.
    """
    N = joint_angles.shape[0]
    cal = _load_calibration()
    xyz_all = cal["joints_xyz"].to(device)
    rpy_all = cal["joints_rpy"].to(device)
    axis_all = cal["joints_axis"].to(device)
    masses = cal["link_masses"].to(device)
    coms = cal["link_coms"].to(device)
    inertias = cal["link_inertias"].to(device)

    T_fixed_all = _build_fixed_transforms(xyz_all, rpy_all, N, device)

    # Precompute link frames: transforms[i+1] = base_link -> link_(i+1).
    transforms = [torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)]
    T = transforms[0]
    for i in range(NUM_ARM_JOINTS):
        T_joint_frame = torch.bmm(T, T_fixed_all[i])
        T_joint = _joint_transform(axis_all[i], joint_angles[:, i])
        T = torch.bmm(T_joint_frame, T_joint)
        transforms.append(T.clone())

    def joint_frame_world(up_to: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return list of (axis_world, p_joint_world) for joints 0..up_to-1."""
        out = []
        T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
        for j in range(up_to):
            T_jf = torch.bmm(T, T_fixed_all[j])
            R_jf = T_jf[:, :3, :3]
            axis_w = torch.bmm(R_jf, axis_all[j].view(1, 3, 1).expand(N, -1, -1)).squeeze(-1)
            p_j = T_jf[:, :3, 3]
            out.append((axis_w, p_j))
            T_joint = _joint_transform(axis_all[j], joint_angles[:, j])
            T = torch.bmm(T_jf, T_joint)
        return out

    M = torch.zeros(N, NUM_ARM_JOINTS, NUM_ARM_JOINTS, device=device, dtype=torch.float32)

    for link_idx in range(NUM_ARM_JOINTS):
        m = masses[link_idx]
        com_local = coms[link_idx]
        I_local = inertias[link_idx]
        I_tensor = torch.zeros(3, 3, device=device, dtype=torch.float32)
        I_tensor[0, 0] = I_local[0]
        I_tensor[1, 1] = I_local[1]
        I_tensor[2, 2] = I_local[2]
        I_tensor[0, 1] = I_tensor[1, 0] = I_local[3]
        I_tensor[0, 2] = I_tensor[2, 0] = I_local[4]
        I_tensor[1, 2] = I_tensor[2, 1] = I_local[5]
        I_batch = I_tensor.unsqueeze(0).expand(N, -1, -1)

        T_link = transforms[link_idx + 1]
        R_link = T_link[:, :3, :3]
        p_link = T_link[:, :3, 3]
        p_com = p_link + torch.bmm(R_link, com_local.view(1, 3, 1).expand(N, -1, -1)).squeeze(-1)
        I_world = torch.bmm(torch.bmm(R_link, I_batch), R_link.transpose(-1, -2))

        joints_before_link = joint_frame_world(link_idx + 1)
        J_link = torch.zeros(N, 6, NUM_ARM_JOINTS, device=device, dtype=torch.float32)
        for j, (axis_w, p_j) in enumerate(joints_before_link):
            J_link[:, :3, j] = torch.cross(axis_w, p_com - p_j, dim=1)
            J_link[:, 3:, j] = axis_w

        J_v = J_link[:, :3, :]
        J_w = J_link[:, 3:, :]
        M += m * torch.bmm(J_v.transpose(-1, -2), J_v)
        M += torch.bmm(J_w.transpose(-1, -2), torch.bmm(I_world, J_w))

    M += torch.eye(NUM_ARM_JOINTS, device=device, dtype=torch.float32).unsqueeze(0) * 1e-6
    return M
