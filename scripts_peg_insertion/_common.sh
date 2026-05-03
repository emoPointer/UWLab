#!/usr/bin/env bash
# Sourced by every step. Activates conda env_isaaclab and cd's to UWLab root.
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab

cd "$HOME/UWLab"

# Force UWLab cfg cloud-asset URLs to resolve to the local cache so IsaacLab's
# spawn-time HEAD request bypasses HuggingFace (avoids HTTP 429 rate limits).
# Pre-populate the cache via huggingface_hub when adding new assets.
export UWLAB_ASSETS_DIR_OVERRIDE="$HOME/.cache/uwlab/assets"

echo "[$(basename "${BASH_SOURCE[1]:-script}")] env=$(conda info --envs | awk '/\*/{print $1}') cwd=$(pwd)"
echo "[uwlab] UWLAB_ASSETS_DIR_OVERRIDE=$UWLAB_ASSETS_DIR_OVERRIDE"
