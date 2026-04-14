"""Replay a recorded trajectory in Isaac Sim and compare FK vs recorded EE pose.

Usage:
    python scripts/tools/replay_arx5_fk.py --hdf5 /home/emopointer/UWLab/0.hdf5 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5", type=str, required=True)
parser.add_argument("--subsample", type=int, default=10, help="print every N frames")
parser.add_argument("--frame_delay", type=float, default=0.05, help="seconds to sleep between frames for visualization")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after AppLauncher ---
import time

import h5py
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat

from uwlab_assets.robots.arx5.arx5 import IMPLICIT_ARX5


def load_trajectory(path: str):
    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][:]  # (T, 7)
        eef = f["eef_qpos"][:]            # (T, 7)
        root_attrs = dict(f.attrs)
    return qpos, eef, root_attrs


def main():
    qpos_np, eef_np, attrs = load_trajectory(args_cli.hdf5)
    T = qpos_np.shape[0]
    print(f"Loaded {T} frames from {args_cli.hdf5}")
    print(f"  root attrs: {attrs}")
    if attrs.get("sim", False):
        print(f"  ⚠ sim=True — this file may be simulation data, not real-world.")
    print(f"  qpos range per col:")
    for i in range(qpos_np.shape[1]):
        print(f"    col{i}: [{qpos_np[:, i].min():.4f}, {qpos_np[:, i].max():.4f}]")
    print(f"  eef_qpos range per col:")
    for i in range(eef_np.shape[1]):
        print(f"    col{i}: [{eef_np[:, i].min():.4f}, {eef_np[:, i].max():.4f}]")

    # --- Set up sim ---
    # Disable gravity: IMPLICIT_ARX5 has stiffness=0/damping=0, so gravity would
    # drag joints away from the written targets during sim.step and contaminate FK.
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.3])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg())

    robot_cfg = IMPLICIT_ARX5.replace(prim_path="/World/Arx5")
    robot = Articulation(robot_cfg)
    sim.reset()
    device = sim.device

    # --- Resolve joint / body indices ---
    arm_names = [f"joint{i}" for i in range(1, 7)]
    arm_ids, arm_resolved = robot.find_joints(arm_names, preserve_order=True)
    print(f"\nResolved arm joints: {list(zip(arm_ids, arm_resolved))}")
    gripper_names = ["joint7", "joint8"]
    gripper_ids, _ = robot.find_joints(gripper_names, preserve_order=True)

    ee_body_name = "link6"
    ee_ids, _ = robot.find_bodies(ee_body_name)
    ee_id = ee_ids[0]
    print(f"EE body '{ee_body_name}' -> idx {ee_id}")

    # --- Prepare data tensors ---
    qpos_arm = torch.tensor(qpos_np[:, :6], dtype=torch.float32, device=device)
    qpos_gripper = torch.tensor(qpos_np[:, 6:7], dtype=torch.float32, device=device)
    eef_pos_real = torch.tensor(eef_np[:, :3], dtype=torch.float32, device=device)
    eef_rot_real = torch.tensor(eef_np[:, 3:6], dtype=torch.float32, device=device)

    default_q = robot.data.default_joint_pos[0].clone()

    errors_pos = []
    errors_rot = []

    print("\n=== Replay ===")
    for t in range(T):
        full_q = default_q.clone()
        full_q[arm_ids] = qpos_arm[t]
        # gripper: real data is 1 scalar; ARX5 sim has 2 gripper joints (joint7/joint8, symmetric)
        full_q[gripper_ids[0]] = qpos_gripper[t, 0]
        full_q[gripper_ids[1]] = qpos_gripper[t, 0]

        robot.write_joint_state_to_sim(
            full_q.unsqueeze(0),
            torch.zeros_like(full_q).unsqueeze(0),
        )
        robot.reset()
        # One physics step so body transforms refresh; render=True to show GUI.
        sim.step(render=not args_cli.headless)
        robot.update(sim.get_physics_dt())
        if not args_cli.headless and args_cli.frame_delay > 0:
            time.sleep(args_cli.frame_delay)

        # EE pose in world frame
        ee_pos_w = robot.data.body_pos_w[0, ee_id].unsqueeze(0)
        ee_quat_w = robot.data.body_quat_w[0, ee_id].unsqueeze(0)
        root_pos = robot.data.root_pos_w[0].unsqueeze(0)
        root_quat = robot.data.root_quat_w[0].unsqueeze(0)

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos, root_quat, ee_pos_w, ee_quat_w
        )
        ee_pos_b = ee_pos_b[0]
        ee_quat_b = ee_quat_b[0]

        # Convert sim quat -> xyz euler for comparison with eef_rot_real (assumed RPY)
        roll, pitch, yaw = euler_xyz_from_quat(ee_quat_b.unsqueeze(0))
        ee_euler_sim = torch.stack([roll[0], pitch[0], yaw[0]])

        err_pos = (ee_pos_b - eef_pos_real[t]).abs()
        # rotation error: wrap into (-pi, pi) then abs
        err_rot = (ee_euler_sim - eef_rot_real[t])
        err_rot = torch.atan2(torch.sin(err_rot), torch.cos(err_rot)).abs()

        errors_pos.append(err_pos)
        errors_rot.append(err_rot)

        if t % args_cli.subsample == 0:
            print(
                f"[{t:3d}] "
                f"real_pos={eef_pos_real[t].cpu().numpy().round(4)} "
                f"sim_pos={ee_pos_b.cpu().numpy().round(4)} "
                f"err_mm={(err_pos.cpu().numpy() * 1000).round(2)} "
                f"err_rot_deg={(err_rot.cpu().numpy() * 180 / np.pi).round(2)}"
            )

    errors_pos = torch.stack(errors_pos)
    errors_rot = torch.stack(errors_rot)

    print("\n===== Summary =====")
    print(f"Frames replayed       : {T}")
    print(f"Position error  (mm):")
    print(f"  max    per-axis    : {(errors_pos.max(dim=0).values.cpu().numpy() * 1000).round(3)}")
    print(f"  mean   per-axis    : {(errors_pos.mean(dim=0).cpu().numpy() * 1000).round(3)}")
    print(f"  max    scalar      : {errors_pos.max().item() * 1000:.3f}")
    print(f"  mean   scalar      : {errors_pos.mean().item() * 1000:.3f}")
    print(f"Rotation error (deg):")
    rot_deg = errors_rot * 180.0 / np.pi
    print(f"  max    per-axis    : {rot_deg.max(dim=0).values.cpu().numpy().round(3)}")
    print(f"  mean   per-axis    : {rot_deg.mean(dim=0).cpu().numpy().round(3)}")
    print(f"  max    scalar      : {rot_deg.max().item():.3f}")
    print(f"  mean   scalar      : {rot_deg.mean().item():.3f}")

    if not args_cli.headless:
        print("\nReplay finished — keeping window open. Close the Isaac Sim window (or Ctrl+C) to exit.")
        while simulation_app.is_running():
            sim.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
