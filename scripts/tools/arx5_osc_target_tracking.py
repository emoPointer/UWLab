"""ARX5 RelCartesianOSC controller closed-loop verification.

Spawns ARX5 in Isaac Sim (with GUI), places a few visual target markers in
the workspace, and drives the EE to each target in sequence using the same
control law as ``RelCartesianOSCAction`` (J^T * (Kp*err + Kd*(-ee_vel)))
but with absolute target poses instead of delta commands.

The goal is to see, with our own eyes and our own numbers:
  - Does the EE actually move toward the target?
  - Is the Jacobian sign correct (no runaway or stuck arm)?
  - Is the frame alignment correct (X command -> X motion)?
  - Does the arm hold against gravity at the gains we picked?
  - What is the steady-state tracking error?

Usage:
    python scripts/tools/arx5_osc_target_tracking.py              # GUI mode
    python scripts/tools/arx5_osc_target_tracking.py --headless   # CI mode
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--max_steps_per_target", type=int, default=600, help="physics steps per target (1/120 s each)")
parser.add_argument("--pos_tol_mm", type=float, default=3.0)
parser.add_argument("--rot_tol_deg", type=float, default=0.5)
parser.add_argument("--print_every", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from uwlab_assets.robots.arx5.arx5 import IMPLICIT_ARX5
from uwlab_assets.robots.arx5.kinematics import (
    ARM_JOINT_NAMES,
    EE_BODY_NAME,
    compute_jacobian_analytical,
)

# Reuse the proven UR5e settle helper (it repeatedly writes joint state +
# flushes via write_data_to_sim + steps physics, which is the only way to
# actually sync PhysX to ArticulationCfg.init_state.joint_pos).
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import settle_robot

# ------------------------- controller gains (TRAIN preset, over-damped) ----
# ARX5 is ~2.5 kg total (vs UR5e ~20 kg). With a zero-stiffness actuator and
# no inertial decoupling, the stiff EVAL gains cause first-step overshoot
# followed by torque saturation and self-collision lock-up. The TRAIN preset
# (Kp=200/3 with damping_ratio 3/1) is intentionally over-damped in
# translation and gives a clean step response, which is what we need for a
# controller smoke test. EVAL gains can be tried later once stability is
# confirmed.
KP = torch.tensor([200.0, 200.0, 200.0, 3.0, 3.0, 3.0])
DAMPING_RATIO = torch.tensor([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
TORQUE_MAX = torch.tensor([50.0] * 6)


def get_ee_pose_base_frame(robot, ee_body_idx):
    """Read EE pose from PhysX body_pos_w (used for diagnostics only)."""
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return ee_pos_b[0], ee_quat_b[0]  # (3,), (4,)




def base_to_world(robot, pos_b, quat_b):
    """Transform a pose from robot base frame to world frame for marker placement."""
    pos_w, quat_w = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        pos_b.unsqueeze(0), quat_b.unsqueeze(0),
    )
    return pos_w[0], quat_w[0]


def quat_from_rotmat(R: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to (w, x, y, z) quaternion."""
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
    q = torch.stack([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q


def compute_osc_torques(
    robot, arm_ids, ee_body_idx,
    pos_des_b: torch.Tensor, quat_des_b: torch.Tensor,
    kp: torch.Tensor, kd: torch.Tensor, torque_max: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror of RelCartesianOSCAction.apply_actions for a single env.

    Returns (joint_torques, pos_error, axis_angle_error) for logging.
    Both the target and current EE pose come from analytical FK so that
    no frame or stale-buffer inconsistency can contaminate the loop.
    """
    ee_pos_b, ee_quat_b = get_ee_pose_base_frame(robot, ee_body_idx)
    joint_pos = torch.stack([robot.data.joint_pos[0, jid] for jid in arm_ids])
    joint_vel = torch.stack([robot.data.joint_vel[0, jid] for jid in arm_ids])

    # Pose error in base frame
    pos_error = pos_des_b - ee_pos_b
    quat_err = quat_mul(quat_des_b.unsqueeze(0), quat_inv(ee_quat_b.unsqueeze(0)))
    axis_angle_error = axis_angle_from_quat(quat_err)[0]
    pose_error = torch.cat([pos_error, axis_angle_error])  # (6,)

    # Analytical Jacobian at current q
    J = compute_jacobian_analytical(joint_pos.unsqueeze(0), device=device)[0]  # (6, 6)
    ee_vel = J @ joint_vel  # (6,)

    task_force = kp * pose_error + kd * (-ee_vel)
    joint_torques = J.t() @ task_force
    joint_torques = torch.clamp(joint_torques, -torque_max, torque_max)
    return joint_torques, pos_error, axis_angle_error


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)  # gravity ON
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 0.9], [0.0, 0.0, 0.3])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg())

    robot_cfg = IMPLICIT_ARX5.replace(prim_path="/World/Arx5")
    robot = Articulation(robot_cfg)

    # Target markers: 5 small red spheres (all spawned at once, updated every frame)
    target_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Targets",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            ),
            "active": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
            ),
        },
    )
    target_markers = VisualizationMarkers(target_marker_cfg)

    sim.reset()
    device = str(sim.device)

    # Resolve joints and bodies
    arm_ids, arm_resolved = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    ee_ids, _ = robot.find_bodies(EE_BODY_NAME)
    ee_body_idx = ee_ids[0]
    print(f"Arm joints: {list(zip(arm_ids, arm_resolved))}")
    print(f"EE body '{EE_BODY_NAME}' -> body idx {ee_body_idx}")

    # Use the same proven pattern as UR5e sysid scripts: repeated
    # write_joint_state_to_sim + write_data_to_sim + sim.step, which is
    # the only reliable way to bring PhysX to the configured init_state
    # for a stiffness=0 articulation.
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)
    settle_robot(
        robot, sim, default_joint_pos, default_joint_vel,
        arm_joint_ids=arm_ids, sim_dt=sim.get_physics_dt(),
        headless=args_cli.headless, settle_steps=10,
    )
    print(f"After settle_robot:")
    print(f"  joint_pos = {robot.data.joint_pos[0].cpu().numpy().round(3)}")

    # Gain tensors on device
    kp = KP.to(device)
    damping_ratio = DAMPING_RATIO.to(device)
    kd = 2.0 * torch.sqrt(kp) * damping_ratio
    torque_max = TORQUE_MAX.to(device)
    print(f"Kp = {kp.cpu().numpy()}")
    print(f"Kd = {kd.cpu().numpy()}")

    # Read home pose from PhysX body_pos_w (fresh after settle_robot) --
    # same approach as UR5e script. settle_robot steps physics, so body
    # poses reflect the default joint positions.
    home_pos_b, home_quat_b = get_ee_pose_base_frame(robot, ee_body_idx)
    print(f"\nHome EE pose (base_link frame):")
    print(f"  pos_b  = {home_pos_b.cpu().numpy().round(4)}")
    print(f"  quat_b = {home_quat_b.cpu().numpy().round(4)}")

    # Define 5 target poses: deltas from home_pos_b, same orientation as home
    deltas_mm = torch.tensor([
        [0.0, 0.0, 0.0],       # 0: hold home
        [50.0, 0.0, 0.0],      # 1: +5cm X
        [50.0, 50.0, 0.0],     # 2: +5cm X, +5cm Y
        [0.0, 50.0, -50.0],    # 3: +5cm Y, -5cm Z
        [0.0, 0.0, 0.0],       # 4: back to home
    ], device=device) / 1000.0  # -> meters

    targets_pos_b = home_pos_b.unsqueeze(0) + deltas_mm  # (5, 3)
    targets_quat_b = home_quat_b.unsqueeze(0).expand(len(deltas_mm), -1)  # (5, 4)
    num_targets = targets_pos_b.shape[0]

    # Initial marker placement: all red, first one green (active)
    def update_markers(active_idx: int):
        all_positions_w = []
        all_indices = []
        for i in range(num_targets):
            p_w, _ = base_to_world(robot, targets_pos_b[i], targets_quat_b[i])
            all_positions_w.append(p_w)
            all_indices.append(1 if i == active_idx else 0)
        pos_tensor = torch.stack(all_positions_w)  # (N, 3)
        idx_tensor = torch.tensor(all_indices, device=device, dtype=torch.int32)
        target_markers.visualize(translations=pos_tensor, marker_indices=idx_tensor)

    print("\n=== Target sequence ===")
    for i, d in enumerate(deltas_mm.cpu().numpy()):
        p = targets_pos_b[i].cpu().numpy()
        print(f"  [{i}] delta_mm={d.round(1)}  target_pos={p.round(4)}")

    pos_tol_m = args_cli.pos_tol_mm / 1000.0
    rot_tol_rad = np.deg2rad(args_cli.rot_tol_deg)

    # Cycle through targets forever so control is always active.
    tgt_idx = 0
    cycle = 0
    steps_on_this_target = 0
    update_markers(tgt_idx)
    print(f"\n--- cycle 0, target {tgt_idx}  pos_des(mm)="
          f"{(targets_pos_b[tgt_idx].cpu().numpy() * 1000).round(1)} ---")

    while simulation_app.is_running():
        pos_des = targets_pos_b[tgt_idx]
        quat_des = targets_quat_b[tgt_idx]

        tau, pos_err, aa_err = compute_osc_torques(
            robot, arm_ids, ee_body_idx,
            pos_des, quat_des,
            kp, kd, torque_max, device,
        )
        full_tau = torch.zeros_like(robot.data.joint_pos[0])
        for i, jid in enumerate(arm_ids):
            full_tau[jid] = tau[i]
        robot.set_joint_effort_target(full_tau.unsqueeze(0))
        robot.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        robot.update(sim.get_physics_dt())

        pos_err_norm = pos_err.norm().item()
        rot_err_norm = aa_err.norm().item()

        if steps_on_this_target % args_cli.print_every == 0:
            print(
                f"  cyc{cycle} tgt{tgt_idx} step {steps_on_this_target:4d}  "
                f"pos_err=[{pos_err[0].item()*1000:+7.2f}, {pos_err[1].item()*1000:+7.2f}, {pos_err[2].item()*1000:+7.2f}] mm  "
                f"|pos|={pos_err_norm*1000:7.3f} mm  "
                f"|rot|={np.rad2deg(rot_err_norm):6.3f} deg  "
                f"|tau|max={tau.abs().max().item():6.2f} Nm"
            )

        reached = (pos_err_norm < pos_tol_m) and (rot_err_norm < rot_tol_rad)
        timeout = steps_on_this_target >= args_cli.max_steps_per_target

        if reached or timeout:
            tag = "REACHED" if reached else "TIMEOUT"
            print(f"  ✓ {tag} cyc{cycle} tgt{tgt_idx} at step {steps_on_this_target} "
                  f"(|pos|={pos_err_norm*1000:.3f} mm, |rot|={np.rad2deg(rot_err_norm):.3f} deg)")
            tgt_idx = (tgt_idx + 1) % num_targets
            if tgt_idx == 0:
                cycle += 1
            steps_on_this_target = 0
            update_markers(tgt_idx)
            print(f"\n--- cycle {cycle}, target {tgt_idx}  pos_des(mm)="
                  f"{(targets_pos_b[tgt_idx].cpu().numpy() * 1000).round(1)} ---")
        else:
            steps_on_this_target += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
