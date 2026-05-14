#!/usr/bin/env bash
# Step 4: Stage-1 RL training (IMPLICIT_ARX5, no sysid) on cube data.
# Defaults from doc: --num_envs 16384 (4-GPU distributed). Single-GPU here -> 8192.
# Drop to 4096 if 8192 OOMs.

source "$(dirname "$0")/_common.sh"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40000}"
SUCCESS_THRESHOLD_SCALE="${SUCCESS_THRESHOLD_SCALE:-2.0}"
LOGGER="${LOGGER:-wandb}"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Arx5-OSC-CubeStack-State-v0 \
    --num_envs "$NUM_ENVS" \
    --logger "$LOGGER" \
    --headless \
    agent.max_iterations="$MAX_ITERATIONS" \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.commands.task_command.success_mode=stack_center \
    env.commands.task_command.success_orientation_required=false \
    env.commands.task_command.success_threshold_scale="$SUCCESS_THRESHOLD_SCALE" \
    env.events.reset_from_reset_states.params.dataset_dir="$DATASET_DIR"
