"""Spawn the current ARX5 robot asset and visualize the configured gripper offset.

Usage:
    python scripts/tools/arx5_visualize_gripper_offset.py
    python scripts/tools/arx5_visualize_gripper_offset.py --headless --steps 10
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize ARX5 link6 and gripper_offset in Isaac Sim.")
parser.add_argument(
    "--steps",
    type=int,
    default=3600,
    help="Number of simulation steps to keep the scene alive for manual inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from uwlab_assets.robots.arx5.arx5 import ARX5_USD_PATH, IMPLICIT_ARX5


def _load_gripper_offset() -> tuple[str, torch.Tensor, torch.Tensor]:
    metadata_path = os.path.join(os.path.dirname(ARX5_USD_PATH), "metadata.yaml")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    offset = metadata["gripper_offset"]
    pos = torch.tensor(offset["pos"], dtype=torch.float32)
    quat = torch.tensor(offset["quat"], dtype=torch.float32)
    return metadata_path, pos, quat


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, gravity=(0.0, 0.0, -9.81))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.25, -1.1, 0.95], [0.15, 0.0, 0.25])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg())

    robot_cfg = IMPLICIT_ARX5.replace(prim_path="/World/Arx5")
    robot = Articulation(robot_cfg)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
    link6_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/link6_frame"))

    grasp_point_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/grasp_point",
        markers={
            "grasp_point": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
            ),
        },
    )
    grasp_point_marker = VisualizationMarkers(grasp_point_marker_cfg)

    sim.reset()

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    for _ in range(5):
        sim.step(render=not args_cli.headless)
        robot.update(sim.get_physics_dt())

    metadata_path, gripper_offset_pos, gripper_offset_quat = _load_gripper_offset()

    ee_body_ids, ee_body_names = robot.find_bodies(["link6"])
    ee_body_id = ee_body_ids[0]

    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    offset_pos_b = gripper_offset_pos.to(sim.device).unsqueeze(0)
    offset_quat_b = gripper_offset_quat.to(sim.device).unsqueeze(0)
    grasp_pos_w, grasp_quat_w = combine_frame_transforms(ee_pos_w, ee_quat_w, offset_pos_b, offset_quat_b)

    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    grasp_pos_b, grasp_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, grasp_pos_w, grasp_quat_w)

    print("=== ARX5 asset / offset visualization ===")
    print(f"Robot USD path        : {ARX5_USD_PATH}")
    print(f"Metadata path         : {metadata_path}")
    print(f"Resolved EE body      : {ee_body_names[0]} (body idx {ee_body_id})")
    print(f"gripper_offset pos    : {gripper_offset_pos.tolist()} m")
    print(f"gripper_offset quat   : {gripper_offset_quat.tolist()} (wxyz)")
    print(f"link6 pose in base    : pos={ee_pos_b[0].cpu().numpy().round(6).tolist()}, "
          f"quat={ee_quat_b[0].cpu().numpy().round(6).tolist()}")
    print(f"grasp pose in base    : pos={grasp_pos_b[0].cpu().numpy().round(6).tolist()}, "
          f"quat={grasp_quat_b[0].cpu().numpy().round(6).tolist()}")
    print(
        "delta(link6->grasp)  : "
        f"{(grasp_pos_b[0] - ee_pos_b[0]).cpu().numpy().round(6).tolist()} m"
    )

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        link6_marker.visualize(ee_pos_w, ee_quat_w)
        grasp_point_marker.visualize(translations=grasp_pos_w)
        sim.step(render=not args_cli.headless)
        robot.update(sim.get_physics_dt())

        ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
        ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
        grasp_pos_w, grasp_quat_w = combine_frame_transforms(ee_pos_w, ee_quat_w, offset_pos_b, offset_quat_b)
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
