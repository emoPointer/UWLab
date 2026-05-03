#!/usr/bin/env bash
# Sourced by every step. Activates conda env_isaaclab and cd's to UWLab root.
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

cd "$HOME/UWLab"

echo "[$(basename "${BASH_SOURCE[1]:-script}")] env=$(conda info --envs | awk '/\*/{print $1}') cwd=$(pwd)"
