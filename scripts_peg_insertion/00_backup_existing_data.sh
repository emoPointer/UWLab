#!/usr/bin/env bash
# Step 0: snapshot the current Datasets/OmniReset/Grasps/Peg + Resets/Peg__PegHole
# into a timestamped sibling dir so the next collection run can overwrite without
# losing the previous version. Cube and SquareLeg datasets are not touched.

source "$(dirname "$0")/_common.sh"

ts=$(date '+%Y%m%d_%H%M%S')
src_root="$HOME/UWLab/Datasets/OmniReset"
backup_root="$HOME/UWLab/Datasets/OmniReset_backups/$ts"

mkdir -p "$backup_root/Grasps" "$backup_root/Resets"

if [[ -d "$src_root/Grasps/Peg" ]]; then
    cp -r "$src_root/Grasps/Peg" "$backup_root/Grasps/Peg"
    echo "[backup] Grasps/Peg -> $backup_root/Grasps/Peg"
else
    echo "[skip] Grasps/Peg not present"
fi

if [[ -d "$src_root/Resets/Peg__PegHole" ]]; then
    cp -r "$src_root/Resets/Peg__PegHole" "$backup_root/Resets/Peg__PegHole"
    echo "[backup] Resets/Peg__PegHole -> $backup_root/Resets/Peg__PegHole"
else
    echo "[skip] Resets/Peg__PegHole not present"
fi

echo
echo "Backup complete at: $backup_root"
du -sh "$backup_root"/* 2>/dev/null
