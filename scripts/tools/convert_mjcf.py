# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a MJCF into USD format.

MuJoCo XML Format (MJCF) is an XML file format used in MuJoCo to describe all elements of a robot.
For more information, see: http://www.mujoco.org/book/XMLreference.html

This script uses the MJCF importer extension from Isaac Sim (``isaacsim.asset.importer.mjcf``) to convert
a MJCF asset into USD format. It is designed as a convenience script for command-line use. For more information
on the MJCF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_mjcf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --fix-base                Fix the base to where it is imported. (default: False)
  --import-sites            Import sites by parse <site> tag. (default: True)
  --make-instanceable       Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a MJCF into USD format.")
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--import-sites", action="store_true", default=False, help="Import sites by parsing the <site> tag."
)
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import xml.etree.ElementTree as ET

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
from pxr import Gf, Usd, UsdGeom

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def _parse_float_tuple(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(item) for item in value.split())


def _add_camera_only_links(mjcf_path: str, usd_path: str):
    """Preserve MJCF bodies that only serve as camera/link anchors.

    Isaac's MJCF importer may put the body transform on a repeated child body, leaving the user-facing
    anchor Xform at the origin.  For deploy camera alignment we keep the anchor itself active and put the
    MJCF body transform directly on that Xform.  The actual Camera prim can then be authored manually under it.
    """

    tree = ET.parse(mjcf_path)
    camera_link_bodies = [
        body for body in tree.getroot().findall(".//body") if _is_camera_anchor_body(body)
    ]
    if not camera_link_bodies:
        return

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD for camera-link patching: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        root_prims = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsA(UsdGeom.Xform)]
        if not root_prims:
            raise RuntimeError(f"Failed to find a root Xform in generated USD: {usd_path}")
        default_prim = root_prims[0]

    for body in camera_link_bodies:
        body_name = body.get("name")
        external_cam_path = default_prim.GetPath().AppendChild(body_name)
        xform = UsdGeom.Xform.Define(stage, external_cam_path)
        xform.GetPrim().SetActive(True)
        xformable = UsdGeom.Xformable(xform.GetPrim())
        xformable.ClearXformOpOrder()
        pos = _parse_float_tuple(body.get("pos"), (0.0, 0.0, 0.0))
        quat = _parse_float_tuple(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        xformable.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xformable.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))

        duplicate_body_path = external_cam_path.AppendChild(body_name)
        if stage.GetPrimAtPath(duplicate_body_path):
            stage.OverridePrim(duplicate_body_path).SetActive(False)

        old_world_body_anchor_path = default_prim.GetPath().AppendChild("worldBody").AppendChild(body_name)
        if stage.GetPrimAtPath(old_world_body_anchor_path):
            stage.RemovePrim(old_world_body_anchor_path)

    stage.GetRootLayer().Save()


def _is_camera_anchor_body(body: ET.Element) -> bool:
    body_name = body.get("name")
    if not body_name or body.find("geom") is not None:
        return False
    return body.find("camera") is not None or "cam" in body_name.lower()


def main():
    # check valid file path
    mjcf_path = args_cli.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # create the converter configuration
    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        import_sites=args_cli.import_sites,
        force_usd_conversion=True,
        make_instanceable=args_cli.make_instanceable,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input MJCF file: {mjcf_path}")
    print("MJCF importer config:")
    print_dict(mjcf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create mjcf converter and import the file
    mjcf_converter = MjcfConverter(mjcf_converter_cfg)
    _add_camera_only_links(mjcf_path, mjcf_converter.usd_path)
    # print output
    print("MJCF importer output:")
    print(f"Generated USD file: {mjcf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(mjcf_converter.usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
