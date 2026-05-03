#!/usr/bin/env bash
# Step 4 (UR5e variant): Stage-1 RL training on official UR5e + Robotiq 2F85.
# Uses the official UWLab task `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0`
# and the official UR5e Peg__PegHole reset states downloaded from HuggingFace
# into `~/.cache/uwlab/assets_official/Datasets/OmniReset/`.
#
# This is for *baseline comparison* — to see how the official robot+pipeline
# performs on the same peg-insertion task, isolating ARX5-specific issues.

source "$(dirname "$0")/_common.sh"

# Override the dataset path: official UR5e Peg__PegHole resets (12-joint).
# Our UWLAB_ASSETS_DIR_OVERRIDE (= ~/.cache/uwlab/assets) keeps USDs (Peg, Mounts,
# UR5e robot) resolving to the local cache; only the dataset_dir is redirected
# to the official UR5e cache.
OFFICIAL_DATASETS="$HOME/.cache/uwlab/assets_official/Datasets/OmniReset"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
    --num_envs 4096 \
    --logger tensorboard \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_from_reset_states.params.dataset_dir="$OFFICIAL_DATASETS"
