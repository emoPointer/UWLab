#!/usr/bin/env bash
# Helper: visualize collected reset states with the GUI.
# Run after any of steps 3a-3d to sanity-check the data.

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/visualize_reset_states.py \
    --task OmniReset-Arx5-OSC-State-Play-v0 \
    --num_envs 4 \
    --dataset_dir ./Datasets/OmniReset \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole
