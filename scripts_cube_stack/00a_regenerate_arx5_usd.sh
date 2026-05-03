#!/usr/bin/env bash
# After hard-revert, ARX5 USD files are LFS pointers (129 / 133 bytes) and
# cannot be loaded by IsaacLab.  Re-convert from URDF to materialise real USDs.
# config.yaml at HEAD has fix_base: true, merge_fixed_joints: false.

source "$(dirname "$0")/_common.sh"

URDF=source/uwlab_assets/uwlab_assets/robots/arx5/assets/arx5_colored.urdf
OUT=source/uwlab_assets/uwlab_assets/robots/arx5/assets/arx5.usd

python _isaaclab/IsaacLab/scripts/tools/convert_urdf.py \
    "$URDF" "$OUT" \
    --fix-base \
    --headless

ls -lh "$OUT" "$(dirname "$OUT")/configuration/arx5_base.usd"
