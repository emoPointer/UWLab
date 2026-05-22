#!/usr/bin/env bash
# Deploy the vision-distilled cube-stack policy with MuJoCo-rendered policy images and Isaac/PhysX physics.

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
MAX_STEPS="${MAX_STEPS:-0}"
SUCCESS_THRESHOLD_SCALE="${SUCCESS_THRESHOLD_SCALE:-2.0}"
MUJOCO_VIDEO_PATH="${MUJOCO_VIDEO_PATH:-./videos/mujoco_isaac_bridge/model_8100_external_camera.mp4}"
MUJOCO_VIDEO_WIDTH="${MUJOCO_VIDEO_WIDTH:-640}"
MUJOCO_VIDEO_HEIGHT="${MUJOCO_VIDEO_HEIGHT:-480}"
RANDOMIZE_MUJOCO_LIGHT_ANGLES="${RANDOMIZE_MUJOCO_LIGHT_ANGLES:-true}"
MUJOCO_LIGHT_YAW_MIN="${MUJOCO_LIGHT_YAW_MIN:-0.0}"
MUJOCO_LIGHT_YAW_MAX="${MUJOCO_LIGHT_YAW_MAX:-360.0}"
MUJOCO_LIGHT_ELEVATION_MIN="${MUJOCO_LIGHT_ELEVATION_MIN:-35.0}"
MUJOCO_LIGHT_ELEVATION_MAX="${MUJOCO_LIGHT_ELEVATION_MAX:-75.0}"
RECORD_MUJOCO_VIDEO="${RECORD_MUJOCO_VIDEO:-true}"
PRINT_ACTOR_OUTPUT="${PRINT_ACTOR_OUTPUT:-true}"
PRINT_ACTOR_OUTPUT_INTERVAL="${PRINT_ACTOR_OUTPUT_INTERVAL:-20}"
STOP_ON_DONE="${STOP_ON_DONE:-true}"
HEADLESS="${HEADLESS:-true}"
REAL_TIME="${REAL_TIME:-false}"

EXTRA_ARGS=()
case "${RECORD_MUJOCO_VIDEO,,}" in
    1|true|yes|on)
        EXTRA_ARGS+=(--record_mujoco_video --mujoco_video_path "$MUJOCO_VIDEO_PATH")
        ;;
    0|false|no|off)
        ;;
    *)
        echo "RECORD_MUJOCO_VIDEO must be one of true/false/1/0/yes/no/on/off, got: $RECORD_MUJOCO_VIDEO" >&2
        exit 2
        ;;
esac

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

case "${STOP_ON_DONE,,}" in
    1|true|yes|on)
        EXTRA_ARGS+=(--stop_on_done)
        ;;
    0|false|no|off)
        ;;
    *)
        echo "STOP_ON_DONE must be one of true/false/1/0/yes/no/on/off, got: $STOP_ON_DONE" >&2
        exit 2
        ;;
esac

case "${HEADLESS,,}" in
    1|true|yes|on)
        EXTRA_ARGS+=(--headless)
        ;;
    0|false|no|off)
        ;;
    *)
        echo "HEADLESS must be one of true/false/1/0/yes/no/on/off, got: $HEADLESS" >&2
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

case "${RANDOMIZE_MUJOCO_LIGHT_ANGLES,,}" in
    1|true|yes|on)
        EXTRA_ARGS+=(
            --randomize_mujoco_light_angles
            --mujoco_light_yaw_range "$MUJOCO_LIGHT_YAW_MIN" "$MUJOCO_LIGHT_YAW_MAX"
            --mujoco_light_elevation_range "$MUJOCO_LIGHT_ELEVATION_MIN" "$MUJOCO_LIGHT_ELEVATION_MAX"
        )
        ;;
    0|false|no|off)
        ;;
    *)
        echo "RANDOMIZE_MUJOCO_LIGHT_ANGLES must be one of true/false/1/0/yes/no/on/off, got: $RANDOMIZE_MUJOCO_LIGHT_ANGLES" >&2
        exit 2
        ;;
esac

python scripts_mujoco_isaac/play_with_mujoco_renderer.py \
    "$CKPT" \
    --task OmniReset-Arx5-OSC-Vision-Deploy-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --max_steps "$MAX_STEPS" \
    --mujoco_policy_images \
    --mujoco_external_camera external_camera \
    --mujoco_wrist_camera wrist_camera \
    --mujoco_camera external_camera \
    --mujoco_video_width "$MUJOCO_VIDEO_WIDTH" \
    --mujoco_video_height "$MUJOCO_VIDEO_HEIGHT" \
    "${EXTRA_ARGS[@]}" \
    agent.algorithm.teacher_checkpoint="" \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_from_reset_states.params.dataset_dir="$DATASET_DIR" \
    env.commands.task_command.success_mode=stack_center \
    env.commands.task_command.success_orientation_required=false \
    env.commands.task_command.success_threshold_scale="$SUCCESS_THRESHOLD_SCALE" \
    env.events.align_deploy_scene_to_robosuite_table.params.insertive_object_color='[0.0,1.0,0.0]' \
    env.events.align_deploy_scene_to_robosuite_table.params.receptive_object_color='[1.0,0.0,0.0]' \
    "$@"
