#!/usr/bin/env bash
# Collect cube-stack state-policy rollouts into HDF5.

source "$(dirname "$0")/_common.sh"

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
    echo "usage: $(basename "$0") <checkpoint.pt>" >&2
    exit 2
fi
shift

NUM_ENVS="${NUM_ENVS:-4}"
ENV_SPACING="${ENV_SPACING:-3.0}"
NUM_DEMOS="${NUM_DEMOS:-50}"
OUTPUT_FILE="${OUTPUT_FILE:-./datasets/cube_stack_state_policy.hdf5}"
MAX_STEPS_PER_DEMO="${MAX_STEPS_PER_DEMO:-160}"
SEED="${SEED:--1}"

python scripts_cube_stack/08_collect_state_policy_dataset.py \
    --checkpoint "$CKPT" \
    --output_file "$OUTPUT_FILE" \
    --num_envs "$NUM_ENVS" \
    --env_spacing "$ENV_SPACING" \
    --num_demos "$NUM_DEMOS" \
    --max_steps_per_demo "$MAX_STEPS_PER_DEMO" \
    --headless \
    "$@"
