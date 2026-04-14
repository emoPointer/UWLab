"""Remove the rootJoint fixed joint from a MuJoCo-converted gripper USD.

This makes the gripper a free-floating articulation so it can be teleported
during grasp sampling.

Usage:
    python scripts/tools/fix_gripper_usd.py /path/to/arx5_gripper.usd
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd_path", type=str)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd, UsdPhysics, PhysxSchema, Sdf

stage = omni.usd.get_context().get_stage()
stage_path = args.usd_path
omni.usd.get_context().open_stage(stage_path)
stage = omni.usd.get_context().get_stage()

removed = 0
gravity_fixed = 0

for prim in stage.Traverse():
    # Remove fixed joints that anchor to world
    if prim.GetTypeName() == "PhysicsFixedJoint" and "rootJoint" in prim.GetName():
        print(f"Removing: {prim.GetPath()}")
        stage.RemovePrim(prim.GetPath())
        removed += 1

    # Disable gravity on all rigid bodies
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_api:
            physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_api.GetDisableGravityAttr().Set(True)
        gravity_fixed += 1
        print(f"Disabled gravity: {prim.GetPath()}")

stage.GetRootLayer().Save()
print(f"\nDone. Removed {removed} fixed joints, disabled gravity on {gravity_fixed} bodies.")
print(f"Saved: {stage_path}")

simulation_app.close()
