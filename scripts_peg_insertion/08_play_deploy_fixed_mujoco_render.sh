#!/usr/bin/env bash
# Deployment-style play with Isaac/PhysX physics and MuJoCo render-only video.
#
# Usage:
#   MUJOCO_RECORD_VIDEO=true bash scripts_peg_insertion/08_play_deploy_fixed_mujoco_render.sh <checkpoint.pt>

source "$(dirname "$0")/_common.sh"

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
    echo "usage: $(basename "$0") <checkpoint.pt>" >&2
    exit 2
fi
shift
EXTRA_ARGS=("$@")

NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:--1}"
TRAINING_ROBOT_X="${TRAINING_ROBOT_X:-"-0.535"}"
TRAINING_ROBOT_Y="${TRAINING_ROBOT_Y:-"-0.21"}"
TRAINING_ROBOT_Z="${TRAINING_ROBOT_Z:-0.8}"
ROBOT_X="${ROBOT_X:-"-0.535"}"
ROBOT_Y="${ROBOT_Y:-"-0.21"}"
ROBOT_Z="${ROBOT_Z:-0.8}"
TABLE_X="${TABLE_X:-0.0}"
TABLE_Y="${TABLE_Y:-0.0}"
TABLE_Z="${TABLE_Z:-0.799375}"
PEGHOLE_X="${PEGHOLE_X:-"-0.30"}"
PEGHOLE_Y="${PEGHOLE_Y:-"-0.20"}"
PEGHOLE_Z="${PEGHOLE_Z:-0.84}"
PEGHOLE_QW="${PEGHOLE_QW:-1.0}"
PEGHOLE_QX="${PEGHOLE_QX:-0.0}"
PEGHOLE_QY="${PEGHOLE_QY:-0.0}"
PEGHOLE_QZ="${PEGHOLE_QZ:-0.0}"
WORKSPACE_X_MIN="${WORKSPACE_X_MIN:-"-0.4"}"
WORKSPACE_X_MAX="${WORKSPACE_X_MAX:-"-0.2"}"
WORKSPACE_Y_MIN="${WORKSPACE_Y_MIN:-"-0.3"}"
WORKSPACE_Y_MAX="${WORKSPACE_Y_MAX:-"-0.1"}"
DEPLOY_LOG_EVERY_RESET="${DEPLOY_LOG_EVERY_RESET:-true}"
RESET_TYPES="${RESET_TYPES:-[ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped]}"
PROBS="${PROBS:-[0.5,0.5]}"

MUJOCO_XML="${MUJOCO_XML:-mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml}"
MUJOCO_CAMERA="${MUJOCO_CAMERA:-external_camera}"
MUJOCO_VIDEO_PATH="${MUJOCO_VIDEO_PATH:-logs/mujoco_isaac_bridge/deploy_fixed.mp4}"
MUJOCO_VIDEO_WIDTH="${MUJOCO_VIDEO_WIDTH:-640}"
MUJOCO_VIDEO_HEIGHT="${MUJOCO_VIDEO_HEIGHT:-480}"
MUJOCO_INSERTIVE_BODY="${MUJOCO_INSERTIVE_BODY:-insertive_cube}"
MUJOCO_RECEPTIVE_BODY="${MUJOCO_RECEPTIVE_BODY:-receptive_cube}"
MUJOCO_RECORD_VIDEO="${MUJOCO_RECORD_VIDEO:-true}"
MUJOCO_MAX_STEPS="${MUJOCO_MAX_STEPS:-0}"
MUJOCO_STOP_ON_DONE="${MUJOCO_STOP_ON_DONE:-true}"

echo "[deploy-play-mujoco-render] checkpoint=$CKPT"
echo "[deploy-play-mujoco-render] num_envs=$NUM_ENVS"
echo "[deploy-play-mujoco-render] seed=$SEED"
echo "[deploy-play-mujoco-render] mujoco_xml=$MUJOCO_XML"
echo "[deploy-play-mujoco-render] mujoco_camera=$MUJOCO_CAMERA"
echo "[deploy-play-mujoco-render] mujoco_video_path=$MUJOCO_VIDEO_PATH"
echo "[deploy-play-mujoco-render] mujoco_object_bodies=insertive:$MUJOCO_INSERTIVE_BODY receptive:$MUJOCO_RECEPTIVE_BODY"

mujoco_record_args=()
if [[ "$MUJOCO_RECORD_VIDEO" == "true" ]]; then
    mujoco_record_args+=(--record_mujoco_video)
fi
if [[ "$MUJOCO_STOP_ON_DONE" == "true" ]]; then
    mujoco_record_args+=(--stop_on_done)
fi

python scripts_mujoco_isaac/play_with_mujoco_renderer.py "$CKPT" \
    --task OmniReset-Arx5-OSC-State-Deploy-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --mujoco_xml "$MUJOCO_XML" \
    --mujoco_camera "$MUJOCO_CAMERA" \
    --mujoco_video_path "$MUJOCO_VIDEO_PATH" \
    --mujoco_video_width "$MUJOCO_VIDEO_WIDTH" \
    --mujoco_video_height "$MUJOCO_VIDEO_HEIGHT" \
    --mujoco_insertive_body "$MUJOCO_INSERTIVE_BODY" \
    --mujoco_receptive_body "$MUJOCO_RECEPTIVE_BODY" \
    --max_steps "$MUJOCO_MAX_STEPS" \
    "${mujoco_record_args[@]}" \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset \
    env.events.reset_from_reset_states.params.reset_types="$RESET_TYPES" \
    env.events.reset_from_reset_states.params.probs="$PROBS" \
    env.events.align_deploy_scene_to_robosuite_table.params.training_robot_base_pose="[$TRAINING_ROBOT_X,$TRAINING_ROBOT_Y,$TRAINING_ROBOT_Z,1.0,0.0,0.0,0.0]" \
    env.events.align_deploy_scene_to_robosuite_table.params.robosuite_robot_base_pose="[$ROBOT_X,$ROBOT_Y,$ROBOT_Z,1.0,0.0,0.0,0.0]" \
    env.events.align_deploy_scene_to_robosuite_table.params.table_pose="[$TABLE_X,$TABLE_Y,$TABLE_Z,1.0,0.0,0.0,0.0]" \
    env.events.align_deploy_scene_to_robosuite_table.params.receptive_object_pose="[$PEGHOLE_X,$PEGHOLE_Y,$PEGHOLE_Z,$PEGHOLE_QW,$PEGHOLE_QX,$PEGHOLE_QY,$PEGHOLE_QZ]" \
    env.events.align_deploy_scene_to_robosuite_table.params.workspace_x_range="[$WORKSPACE_X_MIN,$WORKSPACE_X_MAX]" \
    env.events.align_deploy_scene_to_robosuite_table.params.workspace_y_range="[$WORKSPACE_Y_MIN,$WORKSPACE_Y_MAX]" \
    env.events.align_deploy_scene_to_robosuite_table.params.log_every_reset="$DEPLOY_LOG_EVERY_RESET" \
    "${EXTRA_ARGS[@]}"
