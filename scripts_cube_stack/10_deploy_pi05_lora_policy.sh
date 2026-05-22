#!/usr/bin/env bash
# Deploy the pi0.5 LoRA cube-stack policy through an openpi websocket server.

source "$(dirname "$0")/_common.sh"

OPENPI_ROOT="${OPENPI_ROOT:-/home/emopointer/openpi}"
POLICY_CONFIG="${POLICY_CONFIG:-uwlab_cube_stack_lora}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/emopointer/openpi/checkpoints/uwlab_cube_stack_lora/only_action_expert_lora/29999}"
POLICY_HOST="${POLICY_HOST:-localhost}"
POLICY_PORT="${POLICY_PORT:-8000}"
PROMPT="${PROMPT:-Place the green block on top of the red block.}"
START_SERVER="${START_SERVER:-0}"
SERVER_CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
CLIENT_CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
SERVER_XLA_PYTHON_CLIENT_PREALLOCATE="${SERVER_XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
SERVER_XLA_PYTHON_CLIENT_MEM_FRACTION="${SERVER_XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

export PYTHONPATH="${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "$START_SERVER" == "1" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is required to start the openpi server automatically." >&2
        exit 1
    fi
    (
        cd "$OPENPI_ROOT"
        if [[ -n "$SERVER_CUDA_VISIBLE_DEVICES" ]]; then
            export CUDA_VISIBLE_DEVICES="$SERVER_CUDA_VISIBLE_DEVICES"
        fi
        export XLA_PYTHON_CLIENT_PREALLOCATE="$SERVER_XLA_PYTHON_CLIENT_PREALLOCATE"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="$SERVER_XLA_PYTHON_CLIENT_MEM_FRACTION"
        uv run scripts/serve_policy.py \
            --port "$POLICY_PORT" \
            --default-prompt "$PROMPT" \
            policy:checkpoint \
            --policy.config="$POLICY_CONFIG" \
            --policy.dir="$CHECKPOINT_DIR"
    ) &
    server_pid="$!"
    echo "[10_deploy_pi05_lora_policy.sh] started openpi server pid=$server_pid port=$POLICY_PORT xla_preallocate=$SERVER_XLA_PYTHON_CLIENT_PREALLOCATE xla_mem_fraction=$SERVER_XLA_PYTHON_CLIENT_MEM_FRACTION"
else
    echo "[10_deploy_pi05_lora_policy.sh] using existing openpi server at ${POLICY_HOST}:${POLICY_PORT}"
fi

if [[ -n "$CLIENT_CUDA_VISIBLE_DEVICES" ]]; then
    CUDA_VISIBLE_DEVICES="$CLIENT_CUDA_VISIBLE_DEVICES" python scripts_cube_stack/10_deploy_pi05_lora_policy.py \
        --policy-host "$POLICY_HOST" \
        --policy-port "$POLICY_PORT" \
        --prompt "$PROMPT" \
        "$@"
else
    python scripts_cube_stack/10_deploy_pi05_lora_policy.py \
        --policy-host "$POLICY_HOST" \
        --policy-port "$POLICY_PORT" \
        --prompt "$PROMPT" \
        "$@"
fi
