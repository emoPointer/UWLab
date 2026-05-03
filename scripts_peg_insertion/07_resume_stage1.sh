#!/usr/bin/env bash
# Step 4 (resume): continue Stage-1 RL training from the most recent ARX5 ckpt.
# Mirrors 07_train_stage1.sh but adds --resume_path pointing at the latest
# `model_<iter>.pt` under logs/rsl_rl/arx5_omnireset_agent/<latest run>/.
#
# Override the ckpt explicitly by passing a path as $1, e.g.:
#   bash scripts_peg_insertion/07_resume_stage1.sh \
#       /home/emopointer/UWLab/logs/rsl_rl/arx5_omnireset_agent/2026-04-28_18-40-50/model_300.pt

source "$(dirname "$0")/_common.sh"

LOG_ROOT="$HOME/UWLab/logs/rsl_rl/arx5_omnireset_agent"

if [[ -n "${1:-}" ]]; then
    RESUME_PATH="$1"
else
    LATEST_RUN=$(ls -1t "$LOG_ROOT" | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "[error] no run dir found under $LOG_ROOT — start with 07_train_stage1.sh first." >&2
        exit 1
    fi
    LATEST_CKPT=$(ls -1 "$LOG_ROOT/$LATEST_RUN"/model_*.pt 2>/dev/null \
                  | awk -F'model_|\\.pt' '{print $2"\t"$0}' \
                  | sort -n -k1,1 \
                  | tail -1 \
                  | cut -f2)
    if [[ -z "$LATEST_CKPT" ]]; then
        echo "[error] no model_*.pt found in $LOG_ROOT/$LATEST_RUN" >&2
        exit 1
    fi
    RESUME_PATH="$LATEST_CKPT"
fi

echo "[resume] loading checkpoint: $RESUME_PATH"

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Arx5-OSC-State-v0 \
    --num_envs 8192 \
    --logger tensorboard \
    --headless \
    --resume_path "$RESUME_PATH" \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
