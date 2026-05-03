#!/usr/bin/env bash
# Step 2: Record partial-assembly poses (peg partially inserted into peghole).
# Output: ./Datasets/OmniReset/Resets/Peg__PegHole/partial_assemblies.pt
#
# Note: OmniReset-PartialAssemblies-v0 is robot-agnostic — registered under ur5e_robotiq_2f85
# but reused for ARX5 (it only records object poses). Doc defaults verbatim.

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/record_partial_assemblies.py \
    --task OmniReset-PartialAssemblies-v0 \
    --num_envs 10 \
    --num_trajectories 10 \
    --headless \
    env.scene.insertive_object=peg \
    env.scene.receptive_object=peghole
