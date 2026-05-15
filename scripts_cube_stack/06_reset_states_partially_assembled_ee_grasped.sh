#!/usr/bin/env bash
# Step 3d: Record ObjectPartiallyAssembledEEGrasped reset states.
# Requires: step 1 (grasps.pt) + step 2 (partial_assemblies.pt).
# Output: ./Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube/resets_ObjectPartiallyAssembledEEGrasped.pt

source "$(dirname "$0")/_common.sh"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
NUM_ENVS="${NUM_ENVS:-8192}"
NUM_RESET_STATES="${NUM_RESET_STATES:-10000}"

python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-Arx5-ObjectPartiallyAssembledEEGrasped-v0 \
    --num_envs "$NUM_ENVS" \
    --num_reset_states "$NUM_RESET_STATES" \
    --dataset_dir "$DATASET_DIR" \
    --headless \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_receptive_object_pose.params.pose_range.x="[$ROBOSUITE_WORKSPACE_X_MIN,$ROBOSUITE_WORKSPACE_X_MAX]" \
    env.events.reset_receptive_object_pose.params.pose_range.y="[$ROBOSUITE_WORKSPACE_Y_MIN,$ROBOSUITE_WORKSPACE_Y_MAX]" \
    env.events.reset_receptive_object_pose.params.pose_range.z="[$ROBOSUITE_TABLE_OBJECT_Z,$ROBOSUITE_TABLE_OBJECT_Z]" \
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir="$DATASET_DIR" \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="$DATASET_DIR"
