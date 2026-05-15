#!/usr/bin/env bash
# Step 1: Grasp sampling for ARX5 on the cube.
# Output: ./Datasets/OmniReset/Grasps/InsertiveCube/grasps.pt

source "$(dirname "$0")/_common.sh"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
NUM_ENVS="${NUM_ENVS:-8192}"
NUM_GRASPS="${NUM_GRASPS:-1000}"

python scripts_v2/tools/record_grasps.py \
    --task OmniReset-Arx5-GraspSampling-v0 \
    --num_envs "$NUM_ENVS" \
    --num_grasps "$NUM_GRASPS" \
    --dataset_dir "$DATASET_DIR" \
    --gripper_body_name link6 \
    --headless \
    env.scene.object=cube
