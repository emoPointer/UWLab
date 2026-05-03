#!/usr/bin/env bash
# Deployment-style play: align the reset-state robot and peg to the robosuite
# Lift table frame, and keep the peghole visible on top of the table.
#
# Defaults:
#   robot base = (-0.535, -0.21, 0.8) from SSI-SimToReal robosuite Lift
#   table top ~= z 0.8
#   peghole is sampled in the robosuite Lift workspace each reset
#
# Override examples:
#   WORKSPACE_X_MIN=-0.35 WORKSPACE_X_MAX=-0.25 bash scripts_peg_insertion/08_play_deploy_fixed.sh <checkpoint.pt>
#   SEED=42 bash scripts_peg_insertion/08_play_deploy_fixed.sh <checkpoint.pt>
#   NUM_ENVS=1 bash scripts_peg_insertion/08_play_deploy_fixed.sh <checkpoint.pt>

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

echo "[deploy-play] checkpoint=$CKPT"
echo "[deploy-play] num_envs=$NUM_ENVS"
echo "[deploy-play] seed=$SEED"
echo "[deploy-play] robot_pose=[$ROBOT_X, $ROBOT_Y, $ROBOT_Z, 1.0, 0.0, 0.0, 0.0]"
echo "[deploy-play] table_pose=[$TABLE_X, $TABLE_Y, $TABLE_Z, 1.0, 0.0, 0.0, 0.0]"
echo "[deploy-play] peghole_pose_fallback=[$PEGHOLE_X, $PEGHOLE_Y, $PEGHOLE_Z, $PEGHOLE_QW, $PEGHOLE_QX, $PEGHOLE_QY, $PEGHOLE_QZ]"
echo "[deploy-play] workspace_x_range=[$WORKSPACE_X_MIN, $WORKSPACE_X_MAX]"
echo "[deploy-play] workspace_y_range=[$WORKSPACE_Y_MIN, $WORKSPACE_Y_MAX]"
echo "[deploy-play] log_every_reset=$DEPLOY_LOG_EVERY_RESET"
echo "[deploy-play] reset_types=$RESET_TYPES"
echo "[deploy-play] probs=$PROBS"
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    echo "[deploy-play] extra_args=${EXTRA_ARGS[*]}"
fi

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Arx5-OSC-State-Deploy-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$CKPT" \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset \
    env.events.reset_from_reset_states.params.reset_types="$RESET_TYPES" \
    env.events.reset_from_reset_states.params.probs="$PROBS" \
    env.events.align_deploy_scene_to_robosuite_table.params.robosuite_robot_base_pose="[$ROBOT_X,$ROBOT_Y,$ROBOT_Z,1.0,0.0,0.0,0.0]" \
    env.events.align_deploy_scene_to_robosuite_table.params.table_pose="[$TABLE_X,$TABLE_Y,$TABLE_Z,1.0,0.0,0.0,0.0]" \
    env.events.align_deploy_scene_to_robosuite_table.params.receptive_object_pose="[$PEGHOLE_X,$PEGHOLE_Y,$PEGHOLE_Z,$PEGHOLE_QW,$PEGHOLE_QX,$PEGHOLE_QY,$PEGHOLE_QZ]" \
    env.events.align_deploy_scene_to_robosuite_table.params.workspace_x_range="[$WORKSPACE_X_MIN,$WORKSPACE_X_MAX]" \
    env.events.align_deploy_scene_to_robosuite_table.params.workspace_y_range="[$WORKSPACE_Y_MIN,$WORKSPACE_Y_MAX]" \
    env.events.align_deploy_scene_to_robosuite_table.params.log_every_reset="$DEPLOY_LOG_EVERY_RESET" \
    "${EXTRA_ARGS[@]}"
