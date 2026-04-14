"""UR5e OSC target tracking (reference / sanity test).

Mirrors ``arx5_osc_target_tracking.py`` but on the production-tested
UR5e setup. Purpose: prove that the control-loop pattern (settle_robot
+ analytical-J PD tracker + set_joint_effort_target + sim.step) is
correct on a robot where we know it works, before blaming ARX5.

Both EE pose and Jacobian come from ``uwlab_assets.robots.ur5e_robotiq_gripper``
- the Jacobian via ``compute_jacobian_analytical`` (same function used by
  ``RelCartesianOSCAction`` in production)
- the pose via PhysX ``body_pos_w`` / ``root_pos_w`` (same as
  ``RelCartesianOSCAction._get_ee_pose_root_frame``)

Usage:
    python scripts/tools/ur5e_osc_target_tracking.py               # GUI
    python scripts/tools/ur5e_osc_target_tracking.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--max_steps_per_target", type=int, default=600)
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

from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import (
    ARM_JOINT_NAMES,
    EE_BODY_NAME,
    compute_jacobian_analytical,
)
from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import settle_robot

# TRAIN preset gains (same as UR5E_ROBOTIQ_2F85_RELATIVE_OSC in production)
KP = torch.tensor([200.0, 200.0, 200.0, 3.0, 3.0, 3.0])
DAMPING_RATIO = torch.tensor([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
TORQUE_MAX = torch.tensor([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])


def get_ee_pose_base_frame(robot, ee_body_idx):
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return ee_pos_b[0], ee_quat_b[0]


def base_to_world(robot, pos_b, quat_b):
    pos_w, quat_w = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        pos_b.unsqueeze(0), quat_b.unsqueeze(0),
    )
    return pos_w[0], quat_w[0]


def compute_osc_torques(
    robot, arm_ids, ee_body_idx,
    pos_des_b, quat_des_b,
    kp, kd, torque_max, device,
):
    ee_pos_b, ee_quat_b = get_ee_pose_base_frame(robot, ee_body_idx)
    joint_pos = torch.stack([robot.data.joint_pos[0, jid] for jid in arm_ids])
    joint_vel = torch.stack([robot.data.joint_vel[0, jid] for jid in arm_ids])

    pos_error = pos_des_b - ee_pos_b
    quat_err = quat_mul(quat_des_b.unsqueeze(0), quat_inv(ee_quat_b.unsqueeze(0)))
    axis_angle_error = axis_angle_from_quat(quat_err)[0]
    pose_error = torch.cat([pos_error, axis_angle_error])

    J = compute_jacobian_analytical(joint_pos.unsqueeze(0), device=device)[0]
    ee_vel = J @ joint_vel
    task_force = kp * pose_error + kd * (-ee_vel)
    joint_torques = J.t() @ task_force
    joint_torques = torch.clamp(joint_torques, -torque_max, torque_max)
    return joint_torques, pos_error, axis_angle_error


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.5])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg())

    robot_cfg = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="/World/UR5e")
    robot = Articulation(robot_cfg)

    target_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Targets",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            ),
            "active": sim_utils.SphereCfg(
                radius=0.022,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
            ),
        },
    )
    target_markers = VisualizationMarkers(target_marker_cfg)

    sim.reset()
    device = str(sim.device)

    arm_ids, arm_resolved = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    ee_ids, _ = robot.find_bodies(EE_BODY_NAME)
    ee_body_idx = ee_ids[0]
    print(f"Arm joints: {list(zip(arm_ids, arm_resolved))}")
    print(f"EE body '{EE_BODY_NAME}' -> body idx {ee_body_idx}")

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)
    settle_robot(
        robot, sim, default_joint_pos, default_joint_vel,
        arm_joint_ids=arm_ids, sim_dt=sim.get_physics_dt(),
        headless=args_cli.headless, settle_steps=10,
    )
    print(f"After settle_robot: joint_pos = {robot.data.joint_pos[0].cpu().numpy().round(3)}")

    kp = KP.to(device)
    kd = 2.0 * torch.sqrt(kp) * DAMPING_RATIO.to(device)
    torque_max = TORQUE_MAX.to(device)
    print(f"Kp = {kp.cpu().numpy()}")
    print(f"Kd = {kd.cpu().numpy()}")

    # Read home pose from PhysX (fresh after settle_robot)
    home_pos_b, home_quat_b = get_ee_pose_base_frame(robot, ee_body_idx)
    print(f"\nHome EE pose (base_link frame):")
    print(f"  pos_b  = {home_pos_b.cpu().numpy().round(4)}")
    print(f"  quat_b = {home_quat_b.cpu().numpy().round(4)}")

    # 5 targets in UR5e workspace (small offsets to stay reachable)
    deltas_mm = torch.tensor([
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
        [50.0, 50.0, 0.0],
        [0.0, 50.0, -50.0],
        [0.0, 0.0, 0.0],
    ], device=device) / 1000.0
    targets_pos_b = home_pos_b.unsqueeze(0) + deltas_mm
    targets_quat_b = home_quat_b.unsqueeze(0).expand(len(deltas_mm), -1)
    num_targets = targets_pos_b.shape[0]

    def update_markers(active_idx):
        positions_w = []
        indices = []
        for i in range(num_targets):
            p_w, _ = base_to_world(robot, targets_pos_b[i], targets_quat_b[i])
            positions_w.append(p_w)
            indices.append(1 if i == active_idx else 0)
        target_markers.visualize(
            translations=torch.stack(positions_w),
            marker_indices=torch.tensor(indices, device=device, dtype=torch.int32),
        )

    print("\n=== Target sequence ===")
    for i, d in enumerate(deltas_mm.cpu().numpy()):
        p = targets_pos_b[i].cpu().numpy()
        print(f"  [{i}] delta_mm={d.round(1) * 1000}  target_pos={p.round(4)}")

    pos_tol_m = args_cli.pos_tol_mm / 1000.0
    rot_tol_rad = np.deg2rad(args_cli.rot_tol_deg)

    # Cycle through targets forever so control is always active.
    # Previously the script exited the tracking loop after one pass and
    # fell into a no-control sim.step loop -- IMPLICIT_UR5E has stiffness=0,
    # so without torque the arm immediately collapses under gravity.
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
