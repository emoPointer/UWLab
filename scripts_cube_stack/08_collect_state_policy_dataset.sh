#!/usr/bin/env bash
# Collect cube-stack state-policy rollouts into HDF5.

source "$(dirname "$0")/_common.sh"

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
    echo "usage: $(basename "$0") <checkpoint.pt>" >&2
    exit 2
fi
shift

NUM_ENVS="${NUM_ENVS:-4}"
ENV_SPACING="${ENV_SPACING:-3.0}"
NUM_DEMOS="${NUM_DEMOS:-50}"
DATASET_DIR="${DATASET_DIR:-./Datasets/OmniReset}"
OUTPUT_FILE="${OUTPUT_FILE:-./datasets/cube_stack_state_policy.hdf5}"
MAX_STEPS_PER_DEMO="${MAX_STEPS_PER_DEMO:-160}"
SEED="${SEED:--1}"
FIX_PHYSICS_DR="${FIX_PHYSICS_DR:-true}"
FIX_CONTROL_DR="${FIX_CONTROL_DR:-true}"
LIGHTWEIGHT_RENDER="${LIGHTWEIGHT_RENDER:-false}"
RANDOMIZE_LIGHT="${RANDOMIZE_LIGHT:-true}"
LIGHT_INTENSITY_MIN="${LIGHT_INTENSITY_MIN:-800.0}"
LIGHT_INTENSITY_MAX="${LIGHT_INTENSITY_MAX:-3500.0}"
LIGHT_YAW_MIN="${LIGHT_YAW_MIN:-0.0}"
LIGHT_YAW_MAX="${LIGHT_YAW_MAX:-360.0}"
LIGHT_PITCH_MIN="${LIGHT_PITCH_MIN:--10.0}"
LIGHT_PITCH_MAX="${LIGHT_PITCH_MAX:-10.0}"
LIGHT_ROLL_MIN="${LIGHT_ROLL_MIN:--5.0}"
LIGHT_ROLL_MAX="${LIGHT_ROLL_MAX:-5.0}"

LIGHT_ARGS=()
case "${RANDOMIZE_LIGHT,,}" in
    1|true|yes|on)
        LIGHT_ARGS+=(
            --randomize_light
            --light_intensity_range "$LIGHT_INTENSITY_MIN" "$LIGHT_INTENSITY_MAX"
            --light_yaw_range "$LIGHT_YAW_MIN" "$LIGHT_YAW_MAX"
            --light_pitch_range "$LIGHT_PITCH_MIN" "$LIGHT_PITCH_MAX"
            --light_roll_range "$LIGHT_ROLL_MIN" "$LIGHT_ROLL_MAX"
        )
        ;;
    0|false|no|off)
        ;;
    *)
        echo "RANDOMIZE_LIGHT must be one of true/false/1/0/yes/no/on/off, got: $RANDOMIZE_LIGHT" >&2
        exit 2
        ;;
esac

DR_ARGS=()
case "${FIX_PHYSICS_DR,,}" in
    1|true|yes|on)
        DR_ARGS+=(--fix_physics_dr_to_mean)
        ;;
    0|false|no|off)
        ;;
    *)
        echo "FIX_PHYSICS_DR must be one of true/false/1/0/yes/no/on/off, got: $FIX_PHYSICS_DR" >&2
        exit 2
        ;;
esac
case "${FIX_CONTROL_DR,,}" in
    1|true|yes|on)
        DR_ARGS+=(--fix_control_dr_to_nominal)
        ;;
    0|false|no|off)
        ;;
    *)
        echo "FIX_CONTROL_DR must be one of true/false/1/0/yes/no/on/off, got: $FIX_CONTROL_DR" >&2
        exit 2
        ;;
esac

RENDER_ARGS=()
case "${LIGHTWEIGHT_RENDER,,}" in
    1|true|yes|on)
        RENDER_ARGS+=(--lightweight_render)
        ;;
    0|false|no|off)
        ;;
    *)
        echo "LIGHTWEIGHT_RENDER must be one of true/false/1/0/yes/no/on/off, got: $LIGHTWEIGHT_RENDER" >&2
        exit 2
        ;;
esac

python scripts_cube_stack/08_collect_state_policy_dataset.py \
    --checkpoint "$CKPT" \
    --output_file "$OUTPUT_FILE" \
    --num_envs "$NUM_ENVS" \
    --env_spacing "$ENV_SPACING" \
    --num_demos "$NUM_DEMOS" \
    --dataset_dir "$DATASET_DIR" \
    --max_steps_per_demo "$MAX_STEPS_PER_DEMO" \
    --headless \
    "${RENDER_ARGS[@]}" \
    "${DR_ARGS[@]}" \
    "${LIGHT_ARGS[@]}" \
    "$@"
