#!/usr/bin/env bash
# Train a cube-stack modal vision student with online PPO plus state-policy action distillation.

source "$(dirname "$0")/_common.sh"

: "${TEACHER_CKPT:?Set TEACHER_CKPT=/path/to/state_policy/model_xxxx.pt}"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40000}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-arx5_modal_vision_distill}"
WANDB_CAMERA_VIDEO_INTERVAL="${WANDB_CAMERA_VIDEO_INTERVAL:-100}"
WANDB_CAMERA_VIDEO_LENGTH="${WANDB_CAMERA_VIDEO_LENGTH:-16}"
WANDB_CAMERA_VIDEO_FPS="${WANDB_CAMERA_VIDEO_FPS:-10}"
DISTILL_LAMBDA_INITIAL="${DISTILL_LAMBDA_INITIAL:-1.0}"
DISTILL_LAMBDA_FINAL="${DISTILL_LAMBDA_FINAL:-0.05}"
DISTILL_DECAY_ITERATIONS="${DISTILL_DECAY_ITERATIONS:-8000}"
SUCCESS_THRESHOLD_SCALE="${SUCCESS_THRESHOLD_SCALE:-2.0}"

SSI_ROOT="${SSI_ROOT:-/home/emopointer/SSI-SimToReal}"
SSI_CONFIG="${SSI_CONFIG:-/home/emopointer/SSI-SimToReal/results/policy/0417_UWLab_delat_OSC_control_1447_seed42/config.yaml}"
TRAJECTORY_CONFIG="${TRAJECTORY_CONFIG:-/home/emopointer/UWLab/logs/trajectory_predict/config.yaml}"
TRAJECTORY_CKPT="${TRAJECTORY_CKPT:-/home/emopointer/UWLab/logs/trajectory_predict/model_final.ckpt}"
TABLE_PROMPT="${TABLE_PROMPT:-robot, red cube, green cube}"
WRIST_PROMPT="${WRIST_PROMPT:-red cube, green cube}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Put the red block on the green block.}"
DEPTH_ENCODER="${DEPTH_ENCODER:-vitb}"
MODAL_BATCH_SIZE="${MODAL_BATCH_SIZE:-16}"
DEPTH_CHUNK_SIZE="${DEPTH_CHUNK_SIZE:-16}"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Arx5-OSC-Modal-Vision-v0 \
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
    env.commands.task_command.success_mode=stack_center \
    env.commands.task_command.success_orientation_required=false \
    env.commands.task_command.success_threshold_scale="$SUCCESS_THRESHOLD_SCALE" \
    env.events.set_cube_stack_colors.params.insertive_object_color='[0.0,1.0,0.0]' \
    env.events.set_cube_stack_colors.params.receptive_object_color='[1.0,0.0,0.0]' \
    env.events.reset_from_reset_states.params.dataset_dir="$DATASET_DIR" \
    env.observations.policy.depth_map.params.ssi_root="$SSI_ROOT" \
    env.observations.policy.bboxes.params.ssi_root="$SSI_ROOT" \
    env.observations.policy.trajectory.params.ssi_root="$SSI_ROOT" \
    env.observations.policy.depth_map.params.ssi_config="$SSI_CONFIG" \
    env.observations.policy.bboxes.params.ssi_config="$SSI_CONFIG" \
    env.observations.policy.trajectory.params.ssi_config="$SSI_CONFIG" \
    env.observations.policy.depth_map.params.trajectory_config="$TRAJECTORY_CONFIG" \
    env.observations.policy.bboxes.params.trajectory_config="$TRAJECTORY_CONFIG" \
    env.observations.policy.trajectory.params.trajectory_config="$TRAJECTORY_CONFIG" \
    env.observations.policy.depth_map.params.trajectory_ckpt="$TRAJECTORY_CKPT" \
    env.observations.policy.bboxes.params.trajectory_ckpt="$TRAJECTORY_CKPT" \
    env.observations.policy.trajectory.params.trajectory_ckpt="$TRAJECTORY_CKPT" \
    env.observations.policy.depth_map.params.table_prompt="'$TABLE_PROMPT'" \
    env.observations.policy.bboxes.params.table_prompt="'$TABLE_PROMPT'" \
    env.observations.policy.trajectory.params.table_prompt="'$TABLE_PROMPT'" \
    env.observations.policy.depth_map.params.wrist_prompt="'$WRIST_PROMPT'" \
    env.observations.policy.bboxes.params.wrist_prompt="'$WRIST_PROMPT'" \
    env.observations.policy.trajectory.params.wrist_prompt="'$WRIST_PROMPT'" \
    env.observations.policy.depth_map.params.task_description="$TASK_DESCRIPTION" \
    env.observations.policy.bboxes.params.task_description="$TASK_DESCRIPTION" \
    env.observations.policy.trajectory.params.task_description="$TASK_DESCRIPTION" \
    env.observations.policy.depth_map.params.depth_encoder="$DEPTH_ENCODER" \
    env.observations.policy.bboxes.params.depth_encoder="$DEPTH_ENCODER" \
    env.observations.policy.trajectory.params.depth_encoder="$DEPTH_ENCODER" \
    env.observations.policy.depth_map.params.batch_size="$MODAL_BATCH_SIZE" \
    env.observations.policy.bboxes.params.batch_size="$MODAL_BATCH_SIZE" \
    env.observations.policy.trajectory.params.batch_size="$MODAL_BATCH_SIZE" \
    env.observations.policy.depth_map.params.depth_chunk_size="$DEPTH_CHUNK_SIZE" \
    env.observations.policy.bboxes.params.depth_chunk_size="$DEPTH_CHUNK_SIZE" \
    env.observations.policy.trajectory.params.depth_chunk_size="$DEPTH_CHUNK_SIZE" \
    "$@"
