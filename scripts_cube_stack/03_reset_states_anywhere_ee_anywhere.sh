#!/usr/bin/env bash
# Step 3a: Record ObjectAnywhereEEAnywhere reset states (no upstream deps).
# Output: ./Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube/resets_ObjectAnywhereEEAnywhere.pt

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-Arx5-ObjectAnywhereEEAnywhere-v0 \
    --num_envs 8192 \
    --num_reset_states 10000 \
    --headless \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube
