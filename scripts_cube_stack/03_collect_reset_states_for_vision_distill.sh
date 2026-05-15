#!/usr/bin/env bash
# Re-collect cube-stack reset states in the same robosuite workspace used by vision distillation.

source "$(dirname "$0")/_common.sh"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
NUM_ENVS="${NUM_ENVS:-8192}"
NUM_RESET_STATES="${NUM_RESET_STATES:-10000}"

echo "[collect_reset_states_for_vision_distill] DATASET_DIR=$DATASET_DIR"
echo "[collect_reset_states_for_vision_distill] NUM_ENVS=$NUM_ENVS NUM_RESET_STATES=$NUM_RESET_STATES"
echo "[collect_reset_states_for_vision_distill] workspace x=[$ROBOSUITE_WORKSPACE_X_MIN,$ROBOSUITE_WORKSPACE_X_MAX] y=[$ROBOSUITE_WORKSPACE_Y_MIN,$ROBOSUITE_WORKSPACE_Y_MAX] z=$ROBOSUITE_TABLE_OBJECT_Z"

DATASET_DIR="$DATASET_DIR" NUM_ENVS="$NUM_ENVS" NUM_RESET_STATES="$NUM_RESET_STATES" \
    bash scripts_cube_stack/03_reset_states_anywhere_ee_anywhere.sh

DATASET_DIR="$DATASET_DIR" NUM_ENVS="$NUM_ENVS" NUM_RESET_STATES="$NUM_RESET_STATES" \
    bash scripts_cube_stack/04_reset_states_anywhere_ee_grasped.sh

DATASET_DIR="$DATASET_DIR" NUM_ENVS="$NUM_ENVS" NUM_RESET_STATES="$NUM_RESET_STATES" \
    bash scripts_cube_stack/05_reset_states_resting_ee_grasped.sh

DATASET_DIR="$DATASET_DIR" NUM_ENVS="$NUM_ENVS" NUM_RESET_STATES="$NUM_RESET_STATES" \
    bash scripts_cube_stack/06_reset_states_partially_assembled_ee_grasped.sh
