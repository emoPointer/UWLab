"""Verify ARX5 analytical FK & Jacobian against PhysX.

Samples random joint configurations within the ARX5 joint limits and
compares:
  1. analytical FK link6 pose  vs  PhysX body_pos_w/body_quat_w
  2. analytical Jacobian       vs  finite-difference of analytical FK
     (sanity check that J matches its own FK)

Gravity is disabled because IMPLICIT_ARX5 uses zero-stiffness actuators,
so otherwise the joints would drift from the written targets during
sim.step (see CLAUDE.md § IMPLICIT_ARX5 zero-stiffness actuator notes).

Usage:
    python scripts/tools/arx5_verify_analytical_fk.py --headless --num_samples 20
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    axis_angle_from_quat,
    matrix_from_quat,
)

from uwlab_assets.robots.arx5.arx5 import IMPLICIT_ARX5
from uwlab_assets.robots.arx5.kinematics import (
    ARM_JOINT_NAMES,
    EE_BODY_NAME,
    compute_fk_link6,
    compute_jacobian_analytical,
)


# Sampling ranges (rad); stay well inside URDF limits
SAMPLE_RANGES = torch.tensor([
    [-1.5, 1.5],   # joint1
    [0.0, 2.5],    # joint2
    [0.0, 2.5],    # joint3
    [-1.2, 1.2],   # joint4
    [-1.5, 1.5],   # joint5
    [-1.2, 1.2],   # joint6
])


def sample_random_q(n: int, device: str) -> torch.Tensor:
    lo = SAMPLE_RANGES[:, 0].to(device)
    hi = SAMPLE_RANGES[:, 1].to(device)
    u = torch.rand(n, 6, device=device)
    return lo + u * (hi - lo)


def physx_fk(
    robot, sim, arm_ids, ee_id, q_arm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write q_arm to PhysX, step, read link6 pose in base frame.

    Returns:
        ee_pos_b (3,), ee_quat_b (4,), q_readback (6,) -- q as PhysX actually uses it.
    """
    default_q = robot.data.default_joint_pos[0].clone()
    full_q = default_q.clone()
    for i, jid in enumerate(arm_ids):
        full_q[jid] = q_arm[i]
    robot.write_joint_state_to_sim(
        full_q.unsqueeze(0),
        torch.zeros_like(full_q).unsqueeze(0),
    )
    robot.reset()
    sim.step(render=False)
    robot.update(sim.get_physics_dt())

    # Read back what PhysX is actually using (clamped to joint limits, etc.)
    q_readback = torch.stack([robot.data.joint_pos[0, jid] for jid in arm_ids])

    ee_pos_w = robot.data.body_pos_w[0, ee_id].unsqueeze(0)
    ee_quat_w = robot.data.body_quat_w[0, ee_id].unsqueeze(0)
    root_pos = robot.data.root_pos_w[0].unsqueeze(0)
    root_quat = robot.data.root_quat_w[0].unsqueeze(0)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos, root_quat, ee_pos_w, ee_quat_w)
    return ee_pos_b[0], ee_quat_b[0], q_readback


def analytical_fk(q_arm: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pos, quat_wxyz) of link6 in base_link frame."""
    T = compute_fk_link6(q_arm.unsqueeze(0), device=device)[0]  # (4, 4)
    pos = T[:3, 3]
    R = T[:3, :3]
    # quat (w, x, y, z) from rotation matrix
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = torch.stack([w, x, y, z])
    if quat[0] < 0:
        quat = -quat
    return pos, quat


def quat_angle_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Relative rotation angle between two (w,x,y,z) quats."""
    q_rel = quat_mul(q1.unsqueeze(0), quat_inv(q2.unsqueeze(0)))
    aa = axis_angle_from_quat(q_rel)[0]
    return torch.norm(aa)


def finite_diff_jacobian(q_arm: torch.Tensor, device: str, eps: float = 1e-4) -> torch.Tensor:
    """6x6 finite-difference Jacobian from analytical FK (for sanity checking J)."""
    J = torch.zeros(6, 6, device=device)
    pos0, quat0 = analytical_fk(q_arm, device)
    for i in range(6):
        q_p = q_arm.clone()
        q_p[i] += eps
        pos_p, quat_p = analytical_fk(q_p, device)

        q_m = q_arm.clone()
        q_m[i] -= eps
        pos_m, quat_m = analytical_fk(q_m, device)

        # linear: central diff
        J[:3, i] = (pos_p - pos_m) / (2 * eps)

        # angular: (q_p * q_m^{-1}) -> axis-angle / (2*eps)
        q_rel = quat_mul(quat_p.unsqueeze(0), quat_inv(quat_m.unsqueeze(0)))
        aa = axis_angle_from_quat(q_rel)[0]
        J[3:, i] = aa / (2 * eps)
    return J


def main():
    torch.manual_seed(args_cli.seed)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    # Keep production IMPLICIT_ARX5 settings (enabled_self_collisions=True),
    # same as UR5e's IMPLICIT_UR5E_ROBOTIQ_2F85. Any contact-driven drift is
    # absorbed by comparing analytical FK against PhysX's readback joint_pos
    # (see the random-sample loop below), so the FK comparison stays valid
    # regardless of whether self-collision pushes the joints during sim.step.
    robot_cfg = IMPLICIT_ARX5.replace(prim_path="/World/Arx5")
    robot = Articulation(robot_cfg)
    sim.reset()
    device = str(sim.device)

    arm_ids, arm_resolved = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    ee_ids, _ = robot.find_bodies(EE_BODY_NAME)
    ee_id = ee_ids[0]
    print(f"Arm joints: {list(zip(arm_ids, arm_resolved))}")
    print(f"link6 body idx: {ee_id}")

    pos_errs = []
    rot_errs = []
    jac_errs = []

    qs = sample_random_q(args_cli.num_samples, device)

    # First: verify at q = 0 (matches our earlier sanity check)
    q_zero = torch.zeros(6, device=device)
    p_physx, q_physx, q_rb = physx_fk(robot, sim, arm_ids, ee_id, q_zero)
    p_ana, q_ana = analytical_fk(q_zero, device)
    print("\n=== Sanity check at q = [0]*6 ===")
    print(f"  q written          : {q_zero.cpu().numpy().round(4)}")
    print(f"  q readback (PhysX) : {q_rb.cpu().numpy().round(4)}")
    print(f"  q clamp delta      : {(q_rb - q_zero).cpu().numpy().round(4)}")
    print(f"  PhysX     pos (mm): {(p_physx.cpu().numpy() * 1000).round(3)}")
    print(f"  Analytical pos (mm): {(p_ana.cpu().numpy() * 1000).round(3)}")
    print(f"  pos diff (mm)      : {((p_physx - p_ana).cpu().numpy() * 1000).round(4)}")
    print(f"  rot diff (deg)     : {(quat_angle_diff(q_physx, q_ana).item() * 180 / np.pi):.5f}")

    # Also print PhysX joint limits once for reference
    lo = robot.data.default_joint_pos_limits[0, :, 0] if hasattr(robot.data, "default_joint_pos_limits") \
         else robot.data.joint_pos_limits[0, :, 0]
    hi = robot.data.default_joint_pos_limits[0, :, 1] if hasattr(robot.data, "default_joint_pos_limits") \
         else robot.data.joint_pos_limits[0, :, 1]
    print("\n=== Arm joint limits (from PhysX, post-soft-factor) ===")
    for i, jid in enumerate(arm_ids):
        print(f"  joint{i+1}: [{lo[jid].item():+.4f}, {hi[jid].item():+.4f}] rad")

    print("\n=== Random samples (analytical FK evaluated at PhysX readback q) ===")
    for k in range(args_cli.num_samples):
        q = qs[k]

        p_physx, quat_physx, q_rb = physx_fk(robot, sim, arm_ids, ee_id, q)

        # IMPORTANT: use PhysX's readback q (not the written q) so that both
        # sides of the comparison use exactly the same joint configuration.
        # Any self-collision / limit drift happens to q_rb, but the analytical
        # FK is then evaluated at that same q_rb, so the comparison remains
        # a pure "is my FK formula correct" check.
        p_ana, quat_ana = analytical_fk(q_rb, device)

        clamp_delta = (q_rb - q).abs().max().item()
        pos_err = (p_physx - p_ana).norm().item() * 1000  # mm
        rot_err = quat_angle_diff(quat_physx, quat_ana).item() * 180 / np.pi  # deg

        # Jacobian check: also evaluate at the same q_rb
        J_ana = compute_jacobian_analytical(q_rb.unsqueeze(0), device=device)[0]
        J_fd = finite_diff_jacobian(q_rb, device)
        jac_err = (J_ana - J_fd).abs().max().item()

        pos_errs.append(pos_err)
        rot_errs.append(rot_err)
        jac_errs.append(jac_err)

        drift_tag = "  " if clamp_delta < 1e-4 else "~~"
        if k < 5 or k == args_cli.num_samples - 1:
            print(
                f"[{k:2d}]{drift_tag} q_rb={q_rb.cpu().numpy().round(3)}  "
                f"(drift={clamp_delta:6.4f})  "
                f"pos_err={pos_err:8.5f}mm  rot_err={rot_err:8.5f}deg  J_err={jac_err:.1e}"
            )

    pos_errs = np.array(pos_errs)
    rot_errs = np.array(rot_errs)
    jac_errs = np.array(jac_errs)

    print("\n===== Summary =====")
    print(f"Samples: {args_cli.num_samples}")
    print(f"Position error (analytical FK vs PhysX):")
    print(f"  mean = {pos_errs.mean():.4f} mm    max = {pos_errs.max():.4f} mm")
    print(f"Rotation error (analytical FK vs PhysX):")
    print(f"  mean = {rot_errs.mean():.4f} deg   max = {rot_errs.max():.4f} deg")
    print(f"Jacobian vs finite-diff of analytical FK:")
    print(f"  mean max-abs-err = {jac_errs.mean():.2e}   max = {jac_errs.max():.2e}")

    simulation_app.close()


if __name__ == "__main__":
    main()
