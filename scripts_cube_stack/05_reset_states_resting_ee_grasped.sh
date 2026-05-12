#!/usr/bin/env bash
# Step 3c: Record ObjectRestingEEGrasped reset states.
# Requires: step 1 (grasps.pt) + step 3a (resets_ObjectAnywhereEEAnywhere.pt).
# Output: ./Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube/resets_ObjectRestingEEGrasped.pt

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-Arx5-ObjectRestingEEGrasped-v0 \
    --num_envs 8192 \
    --num_reset_states 10000 \
    --headless \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_receptive_object_pose.params.pose_range.x="[$ROBOSUITE_WORKSPACE_X_MIN,$ROBOSUITE_WORKSPACE_X_MAX]" \
    env.events.reset_receptive_object_pose.params.pose_range.y="[$ROBOSUITE_WORKSPACE_Y_MIN,$ROBOSUITE_WORKSPACE_Y_MAX]" \
    env.events.reset_receptive_object_pose.params.pose_range.z="[$ROBOSUITE_TABLE_OBJECT_Z,$ROBOSUITE_TABLE_OBJECT_Z]" \
    env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=./Datasets/OmniReset \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
