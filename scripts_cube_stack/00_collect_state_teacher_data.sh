#!/usr/bin/env bash
# Collect all cube-stack data needed to train the state teacher policy.

source "$(dirname "$0")/_common.sh"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
NUM_ENVS="${NUM_ENVS:-8192}"
NUM_GRASPS="${NUM_GRASPS:-1000}"
PARTIAL_NUM_ENVS="${PARTIAL_NUM_ENVS:-10}"
NUM_PARTIAL_TRAJECTORIES="${NUM_PARTIAL_TRAJECTORIES:-10}"
NUM_RESET_STATES="${NUM_RESET_STATES:-10000}"

echo "[collect_state_teacher_data] DATASET_DIR=$DATASET_DIR"
echo "[collect_state_teacher_data] NUM_ENVS=$NUM_ENVS NUM_GRASPS=$NUM_GRASPS NUM_RESET_STATES=$NUM_RESET_STATES"
echo "[collect_state_teacher_data] PARTIAL_NUM_ENVS=$PARTIAL_NUM_ENVS NUM_PARTIAL_TRAJECTORIES=$NUM_PARTIAL_TRAJECTORIES"
echo "[collect_state_teacher_data] workspace x=[$ROBOSUITE_WORKSPACE_X_MIN,$ROBOSUITE_WORKSPACE_X_MAX] y=[$ROBOSUITE_WORKSPACE_Y_MIN,$ROBOSUITE_WORKSPACE_Y_MAX] z=$ROBOSUITE_TABLE_OBJECT_Z"

DATASET_DIR="$DATASET_DIR" NUM_ENVS="$NUM_ENVS" NUM_GRASPS="$NUM_GRASPS" \
    bash scripts_cube_stack/01_record_grasps.sh

DATASET_DIR="$DATASET_DIR" PARTIAL_NUM_ENVS="$PARTIAL_NUM_ENVS" NUM_PARTIAL_TRAJECTORIES="$NUM_PARTIAL_TRAJECTORIES" \
    bash scripts_cube_stack/02_record_partial_assemblies.sh

DATASET_DIR="$DATASET_DIR" NUM_ENVS="$NUM_ENVS" NUM_RESET_STATES="$NUM_RESET_STATES" \
    bash scripts_cube_stack/03_collect_reset_states_for_vision_distill.sh
