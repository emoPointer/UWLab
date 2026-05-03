#!/usr/bin/env bash
# Step 1: Grasp sampling for ARX5 on the peg.
# Output: ./Datasets/OmniReset/Grasps/Peg/grasps.pt
#
# Doc reference: https://uw-lab.github.io/UWLab/main/source/publications/omnireset/rl_training.html
# UR5e doc default: --num_envs 8192 --num_grasps 1000
# ARX5-specific: --gripper_body_name link6 (UR5e default 'robotiq_base_link' will not exist on ARX5).

source "$(dirname "$0")/_common.sh"

python scripts_v2/tools/record_grasps.py \
    --task OmniReset-Arx5-GraspSampling-v0 \
    --num_envs 8192 \
    --num_grasps 1000 \
    --gripper_body_name link6 \
    --headless \
    env.scene.object=peg
