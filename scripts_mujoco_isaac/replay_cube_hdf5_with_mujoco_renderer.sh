#!/usr/bin/env bash
# Replay cube HDF5 actions with Isaac/PhysX physics and MuJoCo rendering.

source "$(dirname "$0")/../scripts_cube_stack/_common.sh"

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "usage: $(basename "$0") <dataset.hdf5>" >&2
    exit 2
fi
shift

MUJOCO_VIDEO_PATH="${MUJOCO_VIDEO_PATH:-videos/mujoco_isaac_replays/cube_000000.mp4}"
METRICS_PATH="${METRICS_PATH:-logs/mujoco_isaac_replay_metrics/cube_000000.json}"

python scripts_mujoco_isaac/replay_hdf5_actions_with_mujoco_renderer.py \
    --dataset "$DATASET" \
    --mujoco_video_path "$MUJOCO_VIDEO_PATH" \
    --metrics_path "$METRICS_PATH" \
    --headless \
    "$@"
