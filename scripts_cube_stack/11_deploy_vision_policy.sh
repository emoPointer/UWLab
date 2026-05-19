#!/usr/bin/env bash
# Deploy the RSL-RL vision-distilled cube-stack policy.

source "$(dirname "$0")/_common.sh"

CKPT="${1:-/home/emopointer/UWLab/logs/rsl_rl/arx5_omnireset_vision_distill/2026-05-16_01-56-20/model_8100.pt}"
if [[ ! -f "$CKPT" ]]; then
    echo "checkpoint not found: $CKPT" >&2
    echo "usage: $(basename "$0") [checkpoint.pt]" >&2
    exit 2
fi
shift || true

NUM_ENVS="${NUM_ENVS:-1}"
DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
CAMERA_OUTPUT_DIR="${CAMERA_OUTPUT_DIR:-./videos/vision_deploy/model_8100}"
PRINT_ACTOR_OUTPUT="${PRINT_ACTOR_OUTPUT:-true}"
PRINT_ACTOR_OUTPUT_INTERVAL="${PRINT_ACTOR_OUTPUT_INTERVAL:-20}"
RECORD_DEPLOY_CAMERAS="${RECORD_DEPLOY_CAMERAS:-true}"
REAL_TIME="${REAL_TIME:-false}"

EXTRA_ARGS=()
case "${PRINT_ACTOR_OUTPUT,,}" in
    1|true|yes|on)
        EXTRA_ARGS+=(--print_actor_output --print_actor_output_interval "$PRINT_ACTOR_OUTPUT_INTERVAL")
        ;;
    0|false|no|off)
        ;;
    *)
        echo "PRINT_ACTOR_OUTPUT must be one of true/false/1/0/yes/no/on/off, got: $PRINT_ACTOR_OUTPUT" >&2
        exit 2
        ;;
esac

case "${RECORD_DEPLOY_CAMERAS,,}" in
    1|true|yes|on)
        EXTRA_ARGS+=(--record_deploy_cameras_until_reset --deploy_camera_output_dir "$CAMERA_OUTPUT_DIR")
        ;;
    0|false|no|off)
        ;;
    *)
        echo "RECORD_DEPLOY_CAMERAS must be one of true/false/1/0/yes/no/on/off, got: $RECORD_DEPLOY_CAMERAS" >&2
        exit 2
        ;;
esac

case "${REAL_TIME,,}" in
    1|true|yes|on)
        EXTRA_ARGS+=(--real-time)
        ;;
    0|false|no|off)
        ;;
    *)
        echo "REAL_TIME must be one of true/false/1/0/yes/no/on/off, got: $REAL_TIME" >&2
        exit 2
        ;;
esac

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Arx5-OSC-Vision-Deploy-Play-v0 \
    --checkpoint "$CKPT" \
    --num_envs "$NUM_ENVS" \
    --video_length "$VIDEO_LENGTH" \
    --enable_cameras \
    "${EXTRA_ARGS[@]}" \
    agent.algorithm.teacher_checkpoint="" \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_from_reset_states.params.dataset_dir="$DATASET_DIR" \
    env.commands.task_command.success_mode=stack_center \
    env.commands.task_command.success_orientation_required=false \
    env.commands.task_command.success_threshold_scale=2.0 \
    "$@"
