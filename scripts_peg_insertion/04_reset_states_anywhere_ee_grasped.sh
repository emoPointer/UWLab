#!/usr/bin/env bash
# Step 3b: Record ObjectAnywhereEEGrasped reset states.
# Requires: step 1 (grasps.pt).
# Output: ./Datasets/OmniReset/Resets/Peg__PegHole/resets_ObjectAnywhereEEGrasped.pt

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-Arx5-ObjectAnywhereEEGrasped-v0 \
    --num_envs 8192 \
    --num_reset_states 10000 \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
