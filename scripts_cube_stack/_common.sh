#!/usr/bin/env bash
# Sourced by every step. Activates conda env_isaaclab and cd's to UWLab root.
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

cd "$HOME/UWLab"

ROBOSUITE_WORKSPACE_X_MIN="${ROBOSUITE_WORKSPACE_X_MIN:-"-0.4"}"
ROBOSUITE_WORKSPACE_X_MAX="${ROBOSUITE_WORKSPACE_X_MAX:-"-0.2"}"
ROBOSUITE_WORKSPACE_Y_MIN="${ROBOSUITE_WORKSPACE_Y_MIN:-"-0.3"}"
ROBOSUITE_WORKSPACE_Y_MAX="${ROBOSUITE_WORKSPACE_Y_MAX:-"-0.1"}"
ROBOSUITE_TABLE_OBJECT_Z="${ROBOSUITE_TABLE_OBJECT_Z:-"0.84"}"
ROBOSUITE_OBJECT_AIR_Z_MAX="${ROBOSUITE_OBJECT_AIR_Z_MAX:-"1.14"}"

echo "[$(basename "${BASH_SOURCE[1]:-script}")] env=$(conda info --envs | awk '/\*/{print $1}') cwd=$(pwd)"
