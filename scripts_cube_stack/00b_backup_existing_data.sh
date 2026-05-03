#!/usr/bin/env bash
# Snapshot the existing cube datasets before any re-collection overwrites them.
# Output: ~/UWLab/Datasets/OmniReset_backups/<timestamp>/{Grasps/InsertiveCube,Resets/InsertiveCube__ReceptiveCube}

source "$(dirname "$0")/_common.sh"

ts=$(date '+%Y%m%d_%H%M%S')
src_root="$HOME/UWLab/Datasets/OmniReset"
backup_root="$HOME/UWLab/Datasets/OmniReset_backups/$ts"

mkdir -p "$backup_root/Grasps" "$backup_root/Resets"

if [[ -d "$src_root/Grasps/InsertiveCube" ]]; then
    cp -r "$src_root/Grasps/InsertiveCube" "$backup_root/Grasps/InsertiveCube"
    echo "[backup] Grasps/InsertiveCube -> $backup_root/Grasps/InsertiveCube"
else
    echo "[skip] Grasps/InsertiveCube not present"
fi

if [[ -d "$src_root/Resets/InsertiveCube__ReceptiveCube" ]]; then
    cp -r "$src_root/Resets/InsertiveCube__ReceptiveCube" "$backup_root/Resets/InsertiveCube__ReceptiveCube"
    echo "[backup] Resets/InsertiveCube__ReceptiveCube -> $backup_root/Resets/InsertiveCube__ReceptiveCube"
else
    echo "[skip] Resets/InsertiveCube__ReceptiveCube not present"
fi

echo
echo "Backup at: $backup_root"
du -sh "$backup_root"/* 2>/dev/null || true
