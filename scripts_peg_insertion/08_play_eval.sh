#!/usr/bin/env bash
# Step 5: Play / eval the trained Stage-1 policy.
# Pass the checkpoint path as $1. Example:
#   bash 08_play_eval.sh logs/rsl_rl/arx5_omnireset_agent/<run_dir>/model_<iter>.pt
#
# By default, play on the same four reset types used by all-real/all-easy
# curriculum training. Override RESET_TYPES/PROBS for a narrower check.

source "$(dirname "$0")/_common.sh"

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
    echo "usage: $(basename "$0") <checkpoint.pt>" >&2
    exit 2
fi

NUM_ENVS="${NUM_ENVS:-4}"
RESET_TYPES="${RESET_TYPES:-[ObjectAnywhereEEAnywhere,ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped]}"
PROBS="${PROBS:-[0.25,0.25,0.25,0.25]}"

echo "[play] checkpoint=$CKPT"
echo "[play] num_envs=$NUM_ENVS"
echo "[play] reset_types=$RESET_TYPES"
echo "[play] probs=$PROBS"

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Arx5-OSC-State-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --checkpoint "$CKPT" \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset \
    env.events.reset_from_reset_states.params.reset_types="$RESET_TYPES" \
    env.events.reset_from_reset_states.params.probs="$PROBS"
