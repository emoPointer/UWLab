#!/usr/bin/env bash
# Step 4: Stage-1 RL training (IMPLICIT_ARX5, no sysid) on cube data.
# Defaults from doc: --num_envs 16384 (4-GPU distributed). Single-GPU here -> 8192.
# Drop to 4096 if 8192 OOMs.

source "$(dirname "$0")/_common.sh"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Arx5-OSC-State-v0 \
    --num_envs 8192 \
    --logger tensorboard \
    --headless \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
