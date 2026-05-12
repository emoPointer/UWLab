"""Visualize ARX5 joint6 at its lower, neutral, and upper limits.

Usage:
    python scripts/tools/arx5_visualize_joint6_limits.py
    python scripts/tools/arx5_visualize_joint6_limits.py --headless

The GUI view spawns three ARX5 instances side by side:
    left   : joint6 = -pi/2
    center : joint6 = 0
    right  : joint6 = +pi/2

The terminal output prints the USD/PhysX joint6 limits and the PhysX readback
positions, so the visual check and numeric check are tied to the same asset.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize ARX5 joint6 limits.")
parser.add_argument(
    "--steps",
    type=int,
    default=3600,
    help="Number of simulation steps to keep the scene open. Use a large value for manual inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from uwlab_assets.robots.arx5.arx5 import ARX5_USD_PATH, IMPLICIT_ARX5


JOINT6_VALUES = {
    "lower": -math.pi / 2.0,
    "zero": 0.0,
    "upper": math.pi / 2.0,
}


def _make_robot(name: str, x_offset: float) -> Articulation:
    robot_cfg = IMPLICIT_ARX5.replace(prim_path=f"/World/Arx5_{name}")
    robot_cfg.init_state.pos = (x_offset, 0.0, 0.0)
    return Articulation(robot_cfg)


def _usd_joint6_limits() -> tuple[float, float]:
    stage = Usd.Stage.Open(ARX5_USD_PATH)
    joint6 = stage.GetPrimAtPath("/X5A/joints/joint6")
    return (
        float(joint6.GetAttribute("physics:lowerLimit").Get()),
        float(joint6.GetAttribute("physics:upperLimit").Get()),
    )


def _write_joint6(robot: Articulation, joint6_id: int, joint6_pos: float) -> None:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint6_id] = joint6_pos
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.35, 1.15, 0.55], [0.0, 0.0, 0.18])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg())

    robots = {
        "lower": _make_robot("lower_joint6_minus_pi_over_2", -0.45),
        "zero": _make_robot("zero_joint6", 0.0),
        "upper": _make_robot("upper_joint6_plus_pi_over_2", 0.45),
    }

    sim.reset()

    usd_lower, usd_upper = _usd_joint6_limits()
    print("=== ARX5 joint6 limit visualization ===")
    print(f"USD joint6 limits      : [{usd_lower:+.6f}, {usd_upper:+.6f}] deg")
    print(f"Expected +/- pi/2      : [{-math.pi / 2.0:+.6f}, {math.pi / 2.0:+.6f}] rad")

    for name, robot in robots.items():
        joint6_ids, joint6_names = robot.find_joints(["joint6"], preserve_order=True)
        joint6_id = joint6_ids[0]
        physx_limits = robot.data.default_joint_pos_limits[0, joint6_id]
        _write_joint6(robot, joint6_id, JOINT6_VALUES[name])

        print(
            f"{name:>5} robot joint6      : target={JOINT6_VALUES[name]:+.6f} rad, "
            f"PhysX limit=[{physx_limits[0].item():+.6f}, {physx_limits[1].item():+.6f}] rad, "
            f"resolved joint={joint6_names[0]}"
        )

    for _ in range(5):
        sim.step(render=not args_cli.headless)
        for robot in robots.values():
            robot.update(sim.get_physics_dt())

    print("PhysX readback joint6 positions:")
    for name, robot in robots.items():
        joint6_id = robot.find_joints(["joint6"], preserve_order=True)[0][0]
        q6 = robot.data.joint_pos[0, joint6_id].item()
        print(f"  {name:>5}: {q6:+.6f} rad")

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        for name, robot in robots.items():
            joint6_id = robot.find_joints(["joint6"], preserve_order=True)[0][0]
            _write_joint6(robot, joint6_id, JOINT6_VALUES[name])
        sim.step(render=not args_cli.headless)
        for robot in robots.values():
            robot.update(sim.get_physics_dt())
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
