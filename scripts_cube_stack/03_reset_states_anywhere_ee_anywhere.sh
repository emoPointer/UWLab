#!/usr/bin/env bash
# Step 3a: Record ObjectAnywhereEEAnywhere reset states (no upstream deps).
# Output: ./Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube/resets_ObjectAnywhereEEAnywhere.pt

source "$(dirname "$0")/_common.sh"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
NUM_ENVS="${NUM_ENVS:-8192}"
NUM_RESET_STATES="${NUM_RESET_STATES:-10000}"

python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-Arx5-ObjectAnywhereEEAnywhere-v0 \
    --num_envs "$NUM_ENVS" \
    --num_reset_states "$NUM_RESET_STATES" \
    --dataset_dir "$DATASET_DIR" \
    --headless \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_receptive_object_pose.params.pose_range.x="[$ROBOSUITE_WORKSPACE_X_MIN,$ROBOSUITE_WORKSPACE_X_MAX]" \
    env.events.reset_receptive_object_pose.params.pose_range.y="[$ROBOSUITE_WORKSPACE_Y_MIN,$ROBOSUITE_WORKSPACE_Y_MAX]" \
    env.events.reset_receptive_object_pose.params.pose_range.z="[$ROBOSUITE_TABLE_OBJECT_Z,$ROBOSUITE_TABLE_OBJECT_Z]" \
    env.events.reset_insertive_object_pose.params.pose_range.x="[$ROBOSUITE_WORKSPACE_X_MIN,$ROBOSUITE_WORKSPACE_X_MAX]" \
    env.events.reset_insertive_object_pose.params.pose_range.y="[$ROBOSUITE_WORKSPACE_Y_MIN,$ROBOSUITE_WORKSPACE_Y_MAX]" \
    env.events.reset_insertive_object_pose.params.pose_range.z="[$ROBOSUITE_TABLE_OBJECT_Z,$ROBOSUITE_OBJECT_AIR_Z_MAX]"
