#!/usr/bin/env bash
# Validation: re-train ARX5 cube stack to confirm the rollback to ce81319
# still reproduces the 2026-04-09 success (task_2 jumps to 80% by iter 1000,
# all four tasks ≥95% by iter 1500). If this works, the controller is clean
# and we can move on to re-collecting peg insertion data.
#
# Differences vs 07_train_stage1.sh: insertive/receptive switched to cube,
# pair dir resolves to ./Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube
# (already populated from the 4-09 run, no re-collection needed).

source "$(dirname "$0")/_common.sh"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Arx5-OSC-State-v0 \
    --num_envs 8192 \
    --logger tensorboard \
    --headless \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
