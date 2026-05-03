#!/usr/bin/env bash
# Step 2: Record partial-assembly poses (cube partially inserted into receptive cube hole).
# Output: ./Datasets/OmniReset/Resets/InsertiveCube__ReceptiveCube/partial_assemblies.pt
#
# OmniReset-PartialAssemblies-v0 is robot-agnostic — registered under ur5e_robotiq_2f85
# but reused for ARX5 (it only records object poses).

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/record_partial_assemblies.py \
    --task OmniReset-PartialAssemblies-v0 \
    --num_envs 10 \
    --num_trajectories 10 \
    --headless \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube
