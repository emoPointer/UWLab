#!/usr/bin/env bash
# Step 1: Grasp sampling for ARX5 on the cube.
# Output: ./Datasets/OmniReset/Grasps/InsertiveCube/grasps.pt

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/record_grasps.py \
    --task OmniReset-Arx5-GraspSampling-v0 \
    --num_envs 8192 \
    --num_grasps 1000 \
    --gripper_body_name link6 \
    --headless \
    env.scene.object=cube
