#!/usr/bin/env bash
# Stage-1 ARX5 peg curriculum training.
#
# Usage:
#   bash scripts_peg_insertion/07_train_curriculum.sh
#   bash scripts_peg_insertion/07_train_curriculum.sh task3
#   bash scripts_peg_insertion/07_train_curriculum.sh task23 /path/to/model_XXXX.pt
#   SUCCESS_THRESHOLD_SCALE=1.0 bash scripts_peg_insertion/07_train_curriculum.sh all-real /path/to/model_XXXX.pt
#   MAX_ITERATIONS=3000 bash scripts_peg_insertion/07_train_curriculum.sh task3
#   NUM_ENVS=4096 bash scripts_peg_insertion/07_train_curriculum.sh all-easy /path/to/model_XXXX.pt
#
# Stages:
#   task3    : train only near/partial assembly states
#   task23   : add object-anywhere + grasped states
#   task123  : add resting + grasped states
#   all-easy : all four reset types, relaxed success threshold
#   all-real : all four reset types, original success threshold

source "$(dirname "$0")/_common.sh"

DEFAULT_RESUME_PATH="$HOME/UWLab/logs/rsl_rl/arx5_omnireset_agent/2026-05-02_02-18-13/model_9600.pt"

STAGE="${1:-task23}"
RESUME_PATH="${2:-$DEFAULT_RESUME_PATH}"

case "$STAGE" in
    task3)
        RESET_TYPES="[ObjectPartiallyAssembledEEGrasped]"
        PROBS="[1.0]"
        DEFAULT_SUCCESS_THRESHOLD_SCALE="4.0"
        ;;
    task23)
        RESET_TYPES="[ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped]"
        PROBS="[0.5,0.5]"
        DEFAULT_SUCCESS_THRESHOLD_SCALE="1.0"
        ;;
    task123)
        RESET_TYPES="[ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped]"
        PROBS="[0.30,0.35,0.35]"
        DEFAULT_SUCCESS_THRESHOLD_SCALE="3.0"
        ;;
    all-easy)
        RESET_TYPES="[ObjectAnywhereEEAnywhere,ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped]"
        PROBS="[0.25,0.25,0.25,0.25]"
        DEFAULT_SUCCESS_THRESHOLD_SCALE="4.0"
        ;;
    all-real)
        RESET_TYPES="[ObjectAnywhereEEAnywhere,ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped]"
        PROBS="[0.25,0.25,0.25,0.25]"
        DEFAULT_SUCCESS_THRESHOLD_SCALE="1.0"
        ;;
    *)
        echo "[error] unknown curriculum stage: $STAGE" >&2
        echo "        expected one of: task3, task23, task123, all-easy, all-real" >&2
        exit 2
        ;;
esac

SUCCESS_THRESHOLD_SCALE="${SUCCESS_THRESHOLD_SCALE:-$DEFAULT_SUCCESS_THRESHOLD_SCALE}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
NUM_ENVS="${NUM_ENVS:-8192}"

echo "[curriculum] stage=$STAGE"
echo "[curriculum] reset_types=$RESET_TYPES"
echo "[curriculum] probs=$PROBS"
echo "[curriculum] success_threshold_scale=$SUCCESS_THRESHOLD_SCALE"
echo "[curriculum] num_envs=$NUM_ENVS"
if [[ -n "$MAX_ITERATIONS" ]]; then
    echo "[curriculum] max_iterations=$MAX_ITERATIONS"
fi
if [[ -n "$RESUME_PATH" ]]; then
    if [[ ! -f "$RESUME_PATH" ]]; then
        echo "[error] resume checkpoint not found: $RESUME_PATH" >&2
        exit 3
    fi
    echo "[curriculum] resume_path=$RESUME_PATH"
fi

TRAIN_ARGS=(
    scripts/reinforcement_learning/rsl_rl/train.py
    --task OmniReset-Arx5-OSC-State-v0
    --num_envs "$NUM_ENVS"
    --logger tensorboard
    --headless
)

if [[ -n "$RESUME_PATH" ]]; then
    TRAIN_ARGS+=(--resume_path "$RESUME_PATH")
fi

if [[ -n "$MAX_ITERATIONS" ]]; then
    TRAIN_ARGS+=(agent.max_iterations="$MAX_ITERATIONS")
fi

python "${TRAIN_ARGS[@]}" \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset \
    env.events.reset_from_reset_states.params.reset_types="$RESET_TYPES" \
    env.events.reset_from_reset_states.params.probs="$PROBS" \
    env.commands.task_command.success_threshold_scale="$SUCCESS_THRESHOLD_SCALE"
