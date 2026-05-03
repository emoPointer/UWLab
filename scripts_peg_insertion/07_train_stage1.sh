#!/usr/bin/env bash
# Step 4: Stage-1 RL training (IMPLICIT_ARX5, no sysid) on local datasets.
# Doc reference defaults: --num_envs 16384 --logger wandb --headless --distributed --nproc_per_node 4
# Single 4090 here -> drop torch.distributed.run wrapper. If 16384 envs OOM, edit num_envs.
# Cube training previously used this same task with peg/peghole defaults swapped to cube.

source "$(dirname "$0")/_common.sh"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Arx5-OSC-State-v0 \
    --num_envs 8192 \
    --logger tensorboard \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
