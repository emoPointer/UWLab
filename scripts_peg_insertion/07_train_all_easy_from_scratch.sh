#!/usr/bin/env bash
# Train ARX5 peg insertion on task0-3 from scratch with relaxed success thresholds.
#
# Direct run:
#   bash scripts_peg_insertion/07_train_all_easy_from_scratch.sh
#
# Optional overrides:
#   MAX_ITERATIONS=10000 NUM_ENVS=4096 SUCCESS_THRESHOLD_SCALE=3.0 bash scripts_peg_insertion/07_train_all_easy_from_scratch.sh

source "$(dirname "$0")/_common.sh"

DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
NUM_ENVS="${NUM_ENVS:-8192}"
SUCCESS_THRESHOLD_SCALE="${SUCCESS_THRESHOLD_SCALE:-4.0}"

RESET_TYPES="[ObjectAnywhereEEAnywhere,ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped]"
PROBS="[0.25,0.25,0.25,0.25]"

required_files=(
    "$DATASET_DIR/Grasps/Peg/grasps.pt"
    "$DATASET_DIR/Resets/Peg__PegHole/partial_assemblies.pt"
    "$DATASET_DIR/Resets/Peg__PegHole/resets_ObjectAnywhereEEAnywhere.pt"
    "$DATASET_DIR/Resets/Peg__PegHole/resets_ObjectRestingEEGrasped.pt"
    "$DATASET_DIR/Resets/Peg__PegHole/resets_ObjectAnywhereEEGrasped.pt"
    "$DATASET_DIR/Resets/Peg__PegHole/resets_ObjectPartiallyAssembledEEGrasped.pt"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if (( ${#missing_files[@]} > 0 )); then
    echo "[error] missing required peg datasets:" >&2
    printf '  %s\n' "${missing_files[@]}" >&2
    echo "[hint] rerun scripts_peg_insertion/01-06 data collection before training." >&2
    exit 3
fi

echo "[train-all-easy] from_scratch=true"
echo "[train-all-easy] dataset_dir=$DATASET_DIR"
echo "[train-all-easy] reset_types=$RESET_TYPES"
echo "[train-all-easy] probs=$PROBS"
echo "[train-all-easy] success_threshold_scale=$SUCCESS_THRESHOLD_SCALE"
echo "[train-all-easy] num_envs=$NUM_ENVS"
echo "[train-all-easy] max_iterations=$MAX_ITERATIONS"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Arx5-OSC-State-v0 \
    --num_envs "$NUM_ENVS" \
    --logger tensorboard \
    --headless \
    agent.max_iterations="$MAX_ITERATIONS" \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_from_reset_states.params.dataset_dir="$DATASET_DIR" \
    env.events.reset_from_reset_states.params.reset_types="$RESET_TYPES" \
    env.events.reset_from_reset_states.params.probs="$PROBS" \
    env.commands.task_command.success_threshold_scale="$SUCCESS_THRESHOLD_SCALE" \
    "$@"
