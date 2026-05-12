#!/usr/bin/env bash
# Replay collected cube-stack state-policy HDF5 rollouts.

source "$(dirname "$0")/_common.sh"

DATASET_FILE="${1:-/home/emopointer/UWLab/datasets/cube_stack_state_policy_demo_*.hdf5}"
if [[ "${1:-}" != "" ]]; then
    shift
fi

ENV_SPACING="${ENV_SPACING:-3.0}"
VIDEO_PATH="${VIDEO_PATH:-./videos/cube_stack_replays}"

python scripts_cube_stack/09_replay_state_policy_dataset.py \
    --dataset_file "$DATASET_FILE" \
    --num_envs 1 \
    --env_spacing "$ENV_SPACING" \
    --video_path "$VIDEO_PATH" \
    --headless \
    "$@"
