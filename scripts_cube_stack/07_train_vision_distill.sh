#!/usr/bin/env bash
# Train a cube-stack vision student with online PPO plus state-policy action distillation.

source "$(dirname "$0")/_common.sh"

: "${TEACHER_CKPT:?Set TEACHER_CKPT=/path/to/state_policy/model_xxxx.pt}"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
NUM_ENVS="${NUM_ENVS:-128}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40000}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-arx5_vision_distill}"
WANDB_CAMERA_VIDEO_INTERVAL="${WANDB_CAMERA_VIDEO_INTERVAL:-100}"
WANDB_CAMERA_VIDEO_LENGTH="${WANDB_CAMERA_VIDEO_LENGTH:-16}"
WANDB_CAMERA_VIDEO_FPS="${WANDB_CAMERA_VIDEO_FPS:-10}"
DISTILL_LAMBDA_INITIAL="${DISTILL_LAMBDA_INITIAL:-1.0}"
DISTILL_LAMBDA_FINAL="${DISTILL_LAMBDA_FINAL:-0.05}"
DISTILL_DECAY_ITERATIONS="${DISTILL_DECAY_ITERATIONS:-8000}"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Arx5-OSC-Vision-v0 \
    --num_envs "$NUM_ENVS" \
    --logger "$LOGGER" \
    --log_project_name "$WANDB_PROJECT" \
    --headless \
    --enable_cameras \
    agent.max_iterations="$MAX_ITERATIONS" \
    agent.wandb_camera_video_interval="$WANDB_CAMERA_VIDEO_INTERVAL" \
    agent.wandb_camera_video_length="$WANDB_CAMERA_VIDEO_LENGTH" \
    agent.wandb_camera_video_camera_names='[external_camera]' \
    agent.wandb_camera_video_env_index=0 \
    agent.wandb_camera_video_fps="$WANDB_CAMERA_VIDEO_FPS" \
    agent.algorithm.teacher_checkpoint="$TEACHER_CKPT" \
    agent.algorithm.distillation.lambda_initial="$DISTILL_LAMBDA_INITIAL" \
    agent.algorithm.distillation.lambda_final="$DISTILL_LAMBDA_FINAL" \
    agent.algorithm.distillation.decay_iterations="$DISTILL_DECAY_ITERATIONS" \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.set_cube_stack_colors.params.insertive_object_color='[0.0,1.0,0.0]' \
    env.events.set_cube_stack_colors.params.receptive_object_color='[1.0,0.0,0.0]' \
    env.events.reset_from_reset_states.params.dataset_dir="$DATASET_DIR" \
    "$@"
