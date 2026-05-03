#!/usr/bin/env bash
# Step 3d: Record ObjectPartiallyAssembledEEGrasped reset states.
# Requires: step 1 (grasps.pt) + step 2 (partial_assemblies.pt).
# Output: ./Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube/resets_ObjectPartiallyAssembledEEGrasped.pt

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-Arx5-ObjectPartiallyAssembledEEGrasped-v0 \
    --num_envs 8192 \
    --num_reset_states 10000 \
    --headless \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=./Datasets/OmniReset \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
