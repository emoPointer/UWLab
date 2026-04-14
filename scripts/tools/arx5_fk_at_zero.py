"""Compute ARX5 link6 pose relative to base_link at the power-on / zero joint state.

This checks whether the real-robot's eef_qpos origin corresponds to 'link6 at boot',
i.e. whether link6(q=0) in base frame equals the (~0.0965, 0, ~0.154) constant offset
observed in replay.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat

from uwlab_assets.robots.arx5.arx5 import IMPLICIT_ARX5


def fk_link6_at(robot, sim, arm_ids, gripper_ids, ee_id, q_arm, q_gripper=0.0):
    default_q = robot.data.default_joint_pos[0].clone()
    full_q = default_q.clone()
    for i, jid in enumerate(arm_ids):
        full_q[jid] = q_arm[i]
    for jid in gripper_ids:
        full_q[jid] = q_gripper
    robot.write_joint_state_to_sim(
        full_q.unsqueeze(0),
        torch.zeros_like(full_q).unsqueeze(0),
    )
    robot.reset()
    sim.step(render=False)
    robot.update(sim.get_physics_dt())

    ee_pos_w = robot.data.body_pos_w[0, ee_id].unsqueeze(0)
    ee_quat_w = robot.data.body_quat_w[0, ee_id].unsqueeze(0)
    root_pos = robot.data.root_pos_w[0].unsqueeze(0)
    root_quat = robot.data.root_quat_w[0].unsqueeze(0)

    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos, root_quat, ee_pos_w, ee_quat_w)
    r, p, y = euler_xyz_from_quat(ee_quat_b)
    return ee_pos_b[0].cpu().numpy(), ee_quat_b[0].cpu().numpy(), (r[0].item(), p[0].item(), y[0].item())


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    robot_cfg = IMPLICIT_ARX5.replace(prim_path="/World/Arx5")
    robot = Articulation(robot_cfg)
    sim.reset()
    device = sim.device

    arm_names = [f"joint{i}" for i in range(1, 7)]
    arm_ids, arm_resolved = robot.find_joints(arm_names, preserve_order=True)
    gripper_names = ["joint7", "joint8"]
    gripper_ids, _ = robot.find_joints(gripper_names, preserve_order=True)
    ee_ids, _ = robot.find_bodies("link6")
    ee_id = ee_ids[0]

    print(f"Resolved arm joints (id, name): {list(zip(arm_ids, arm_resolved))}")
    print(f"link6 body idx: {ee_id}")

    zero = torch.zeros(6, dtype=torch.float32, device=device)
    pos, quat, rpy = fk_link6_at(robot, sim, arm_ids, gripper_ids, ee_id, zero, 0.0)

    print("\n=== link6 in base_link frame at arm q = [0,0,0,0,0,0] ===")
    print(f"  pos  (m)  : [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    print(f"  pos  (mm) : [{pos[0]*1000:.3f}, {pos[1]*1000:.3f}, {pos[2]*1000:.3f}]")
    print(f"  quat (wxyz): [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
    print(f"  rpy  (rad) : [{rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}]")

    expected = (0.0965, 0.0, 0.154)
    diff = [pos[i] - expected[i] for i in range(3)]
    print(f"\n  expected (from replay offset): [{expected[0]}, {expected[1]}, {expected[2]}] m")
    print(f"  pos - expected (mm)           : [{diff[0]*1000:.3f}, {diff[1]*1000:.3f}, {diff[2]*1000:.3f}]")

    simulation_app.close()


if __name__ == "__main__":
    main()
