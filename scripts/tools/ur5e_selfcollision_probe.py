"""Probe whether UR5e exhibits the same self-collision → joint-state drift
issue we hit with ARX5 during FK verification.

We don't need analytical FK comparison here — we just sample random
configurations and measure ``|q_readback - q_written|`` after one
sim.step under the same setup as the ARX5 verifier:

  - IMPLICIT_UR5E_ROBOTIQ_2F85  (stiffness = damping = 0)
  - enabled_self_collisions = True
  - gravity = 0

If clamp delta stays < 1e-3 everywhere, UR5e is not practically affected.
If any sample has clamp > a few mrad, UR5e has the exact same issue
and we should be aware of it when doing FK validation there too.

Usage:
    python scripts/tools/ur5e_selfcollision_probe.py --headless --num_samples 50
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--noise", type=float, default=1.5, help="+/- rad noise around default pose")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85

ARM_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
DEFAULT_Q = torch.tensor([0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708])


def main():
    torch.manual_seed(args_cli.seed)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    robot_cfg = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="/World/UR5e")
    robot = Articulation(robot_cfg)
    sim.reset()
    device = str(sim.device)

    arm_ids, arm_resolved = robot.find_joints(ARM_NAMES, preserve_order=True)
    print(f"Arm joints: {list(zip(arm_ids, arm_resolved))}")

    default_q_full = robot.data.default_joint_pos[0].clone()

    default_arm = DEFAULT_Q.to(device)
    noise = args_cli.noise
    u = torch.rand(args_cli.num_samples, 6, device=device) * 2.0 - 1.0  # (-1, 1)
    qs = default_arm.unsqueeze(0) + noise * u  # (N, 6)

    clamp_deltas = []
    worst_clamp = 0.0
    worst_idx = -1

    print(f"\n=== UR5e self-collision probe ({args_cli.num_samples} samples, ±{noise} rad around default) ===")
    for k in range(args_cli.num_samples):
        q_arm = qs[k]
        full_q = default_q_full.clone()
        for i, jid in enumerate(arm_ids):
            full_q[jid] = q_arm[i]
        robot.write_joint_state_to_sim(
            full_q.unsqueeze(0), torch.zeros_like(full_q).unsqueeze(0)
        )
        robot.reset()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())

        q_rb = torch.stack([robot.data.joint_pos[0, jid] for jid in arm_ids])
        clamp = (q_rb - q_arm).abs()
        clamp_max = clamp.max().item()
        clamp_deltas.append(clamp_max)
        if clamp_max > worst_clamp:
            worst_clamp = clamp_max
            worst_idx = k

        flag = "!!" if clamp_max > 1e-3 else "  "
        if k < 5 or clamp_max > 1e-3 or k == args_cli.num_samples - 1:
            print(
                f"[{k:2d}]{flag} q_w={q_arm.cpu().numpy().round(3)}  "
                f"clamp_max={clamp_max:.4f} rad"
            )
            if clamp_max > 1e-3:
                print(f"      q_rb={q_rb.cpu().numpy().round(3)}")

    cd = np.array(clamp_deltas)
    print("\n===== Summary =====")
    print(f"Samples                : {args_cli.num_samples}")
    print(f"Clamp delta (rad)")
    print(f"  mean max-per-sample  : {cd.mean():.6f}")
    print(f"  max  max-per-sample  : {cd.max():.6f}  (sample {worst_idx})")
    print(f"  samples with > 1e-3  : {int((cd > 1e-3).sum())} / {args_cli.num_samples}")
    print(f"  samples with > 1e-2  : {int((cd > 1e-2).sum())} / {args_cli.num_samples}")

    simulation_app.close()


if __name__ == "__main__":
    main()
