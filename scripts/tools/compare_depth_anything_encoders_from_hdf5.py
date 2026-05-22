#!/usr/bin/env python3
"""Compare Depth-Anything-V2 encoders on UWLab HDF5 camera frames.

The image path is intentionally explicit:

- table/external view: top-right 400x400 crop
- wrist view: full image resized to 400x400
- DepthAnything runs from those 400x400 RGB images
- depth maps are resized to 128x128 for policy resolution
- videos show the 400x400 RGB input and the 128x128 depth maps upscaled for viewing
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


DEFAULT_DATASET_DIR = "/home/emopointer/UWLab/datasets_test"
DEFAULT_OUTPUT_DIR = "/home/emopointer/UWLab/videos/depth_anything_encoder_compare"
DEFAULT_SSI_ROOT = "/home/emopointer/SSI-SimToReal"
DEFAULT_DEPTH_ROOT = "/home/emopointer/SSI-SimToReal/Depth_Anything_V2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate vitl/vitb DepthAnything comparison videos from HDF5 demos.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--datasets", nargs="*", default=None, help="Explicit HDF5 paths. Defaults to dataset-dir files.")
    parser.add_argument("--max-datasets", type=int, default=3)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ssi-root", default=DEFAULT_SSI_ROOT)
    parser.add_argument("--depth-root", default=DEFAULT_DEPTH_ROOT)
    parser.add_argument("--encoders", nargs="+", default=["vitl", "vitb"])
    parser.add_argument("--table-key", default="obs/table_cam")
    parser.add_argument("--wrist-key", default="obs/wrist_cam")
    parser.add_argument("--process-size", type=int, default=400)
    parser.add_argument("--model-size", type=int, default=128)
    parser.add_argument("--depth-input-size", type=int, default=400)
    parser.add_argument("--display-size", type=int, default=256)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc used for the temporary MP4.")
    parser.add_argument("--no-ffmpeg-h264", action="store_true", help="Skip ffmpeg H.264/yuv420p compatibility conversion.")
    parser.add_argument("--print-stats", action="store_true", help="Print vitl/vitb shared-normalized difference statistics.")
    return parser.parse_args()


def setup_imports(ssi_root: Path, depth_root: Path) -> None:
    sys.path.insert(0, str(ssi_root))
    sys.path.insert(0, str(depth_root))


def resolve_datasets(args: argparse.Namespace) -> list[Path]:
    if args.datasets:
        paths = [Path(p) for p in args.datasets]
    else:
        root = Path(args.dataset_dir)
        paths = sorted(list(root.glob("*.hdf5")) + list(root.glob("*.h5")))
    if args.max_datasets is not None:
        paths = paths[: args.max_datasets]
    if not paths:
        raise FileNotFoundError(f"No HDF5 files found under {args.dataset_dir}")
    return paths


def check_checkpoints(depth_root: Path, encoders: list[str]) -> None:
    missing = []
    for encoder in encoders:
        ckpt = depth_root / "checkpoints" / f"depth_anything_v2_{encoder}.pth"
        if not ckpt.exists():
            missing.append(str(ckpt))
    if missing:
        raise FileNotFoundError("Missing DepthAnything checkpoint(s):\n" + "\n".join(missing))


def load_images(path: Path, table_key: str, wrist_key: str, max_frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        table = h5_file[table_key][:]
        wrist = h5_file[wrist_key][:]
    if max_frames is not None:
        table = table[:max_frames]
        wrist = wrist[:max_frames]
    if table.shape[0] != wrist.shape[0]:
        raise ValueError(f"{path.name}: table/wrist frame count mismatch: {table.shape[0]} vs {wrist.shape[0]}")
    return table[..., :3].astype(np.uint8, copy=False), wrist[..., :3].astype(np.uint8, copy=False)


def preprocess_table(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if h < size or w < size:
        raise ValueError(f"table image {frame_rgb.shape} is smaller than {size}x{size}")
    return frame_rgb[:size, w - size : w].copy()


def preprocess_wrist(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)


def normalize_depth(depth: np.ndarray, d_min: float | None = None, d_max: float | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if d_min is None:
        d_min = float(np.nanmin(depth))
    if d_max is None:
        d_max = float(np.nanmax(depth))
    return np.clip((depth - d_min) / max(d_max - d_min, 1e-8), 0.0, 1.0)


def depth_to_rgb(depth: np.ndarray, d_min: float | None = None, d_max: float | None = None) -> np.ndarray:
    norm = normalize_depth(depth, d_min=d_min, d_max=d_max)
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def diff_to_rgb(diff: np.ndarray, max_value: float | None = None) -> np.ndarray:
    diff = np.asarray(diff, dtype=np.float32)
    if max_value is None:
        max_value = float(np.nanmax(diff))
    norm = np.clip(diff / max(max_value, 1e-8), 0.0, 1.0)
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def render_tile(image_rgb: np.ndarray, label: str, size: int) -> np.ndarray:
    tile = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_NEAREST)
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (size, 24), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_frame(
    table_rgb_400: np.ndarray,
    wrist_rgb_400: np.ndarray,
    table_depths: dict[str, np.ndarray],
    wrist_depths: dict[str, np.ndarray],
    encoders: list[str],
    display_size: int,
) -> np.ndarray:
    table_tiles = [render_tile(table_rgb_400, "external RGB 400", display_size)]
    wrist_tiles = [render_tile(wrist_rgb_400, "wrist RGB 400", display_size)]
    for encoder in encoders:
        table_tiles.append(render_tile(depth_to_rgb(table_depths[encoder]), f"external {encoder} own", display_size))
        wrist_tiles.append(render_tile(depth_to_rgb(wrist_depths[encoder]), f"wrist {encoder} own", display_size))

    if len(encoders) >= 2:
        left, right = encoders[:2]
        for depths, tiles, prefix in (
            (table_depths, table_tiles, "external"),
            (wrist_depths, wrist_tiles, "wrist"),
        ):
            d_min = min(float(np.nanmin(depths[left])), float(np.nanmin(depths[right])))
            d_max = max(float(np.nanmax(depths[left])), float(np.nanmax(depths[right])))
            left_norm = normalize_depth(depths[left], d_min=d_min, d_max=d_max)
            right_norm = normalize_depth(depths[right], d_min=d_min, d_max=d_max)
            tiles.append(render_tile(depth_to_rgb(depths[left], d_min=d_min, d_max=d_max), f"{prefix} {left} shared", display_size))
            tiles.append(render_tile(depth_to_rgb(depths[right], d_min=d_min, d_max=d_max), f"{prefix} {right} shared", display_size))
            tiles.append(render_tile(diff_to_rgb(np.abs(left_norm - right_norm), max_value=1.0), f"{prefix} abs diff", display_size))

    top = np.concatenate(table_tiles, axis=1)
    bottom = np.concatenate(wrist_tiles, axis=1)
    return np.concatenate([top, bottom], axis=0)


def pairwise_depth_stats(left_depths: np.ndarray, right_depths: np.ndarray) -> dict[str, float]:
    if left_depths.shape != right_depths.shape:
        raise ValueError(f"depth shapes differ: {left_depths.shape} vs {right_depths.shape}")
    left = left_depths.astype(np.float32, copy=False)
    right = right_depths.astype(np.float32, copy=False)
    d_min = np.minimum(
        left.reshape(left.shape[0], -1).min(axis=1),
        right.reshape(right.shape[0], -1).min(axis=1),
    )[:, None, None]
    d_max = np.maximum(
        left.reshape(left.shape[0], -1).max(axis=1),
        right.reshape(right.shape[0], -1).max(axis=1),
    )[:, None, None]
    denom = np.maximum(d_max - d_min, 1e-8)
    left_norm = np.clip((left - d_min) / denom, 0.0, 1.0)
    right_norm = np.clip((right - d_min) / denom, 0.0, 1.0)
    diff = np.abs(left_norm - right_norm)
    signed = left_norm - right_norm
    left_flat = left_norm.reshape(left_norm.shape[0], -1)
    right_flat = right_norm.reshape(right_norm.shape[0], -1)
    corrs = []
    for l_frame, r_frame in zip(left_flat, right_flat):
        l_std = float(l_frame.std())
        r_std = float(r_frame.std())
        if l_std < 1e-8 or r_std < 1e-8:
            continue
        corrs.append(float(np.corrcoef(l_frame, r_frame)[0, 1]))
    return {
        "frames": float(left.shape[0]),
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(signed * signed))),
        "p50": float(np.percentile(diff, 50)),
        "p90": float(np.percentile(diff, 90)),
        "p95": float(np.percentile(diff, 95)),
        "p99": float(np.percentile(diff, 99)),
        "max": float(diff.max()),
        "bias": float(signed.mean()),
        "corr": float(np.mean(corrs)) if corrs else float("nan"),
    }


def print_stats_line(dataset_name: str, view_name: str, left_name: str, right_name: str, stats: dict[str, float]) -> None:
    print(
        "[stats] "
        f"demo={dataset_name} view={view_name} pair={left_name}/{right_name} "
        f"frames={int(stats['frames'])} "
        f"mae={stats['mae']:.6f} rmse={stats['rmse']:.6f} "
        f"p50={stats['p50']:.6f} p90={stats['p90']:.6f} "
        f"p95={stats['p95']:.6f} p99={stats['p99']:.6f} "
        f"max={stats['max']:.6f} bias={stats['bias']:.6f} corr={stats['corr']:.6f}"
    )


@torch.no_grad()
def compute_depths(
    images_400: np.ndarray,
    encoders: list[str],
    depth_generators: dict[str, torch.nn.Module],
    depth_preprocessor,
    get_depth_emb_a800,
    model_size: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    images = torch.from_numpy(images_400).float().permute(0, 3, 1, 2)
    images = images[:, None, None]  # N, V=1, T=1, C, H, W
    out: dict[str, np.ndarray] = {}
    for encoder in encoders:
        generator = depth_generators[encoder]
        device = next(generator.parameters()).device
        depth = get_depth_emb_a800(
            generator,
            depth_preprocessor,
            images.to(device),
            n_obs_steps=1,
            chunk_size=batch_size,
            target_size=(model_size, model_size),
        )
        out[encoder] = depth[:, 0, 0, 0].detach().cpu().numpy()
    return out


def process_dataset(
    path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    encoders: list[str],
    depth_generators: dict[str, torch.nn.Module],
    depth_preprocessor,
    get_depth_emb_a800,
) -> Path:
    table, wrist = load_images(path, args.table_key, args.wrist_key, args.max_frames)
    table_400 = np.stack([preprocess_table(frame, args.process_size) for frame in table], axis=0)
    wrist_400 = np.stack([preprocess_wrist(frame, args.process_size) for frame in wrist], axis=0)

    stacked = np.concatenate([table_400, wrist_400], axis=0)
    depth_by_encoder = compute_depths(
        stacked,
        encoders,
        depth_generators,
        depth_preprocessor,
        get_depth_emb_a800,
        args.model_size,
        args.batch_size,
    )
    n_frames = table_400.shape[0]
    if args.print_stats and len(encoders) >= 2:
        left, right = encoders[:2]
        print_stats_line(
            path.stem,
            "external",
            left,
            right,
            pairwise_depth_stats(depth_by_encoder[left][:n_frames], depth_by_encoder[right][:n_frames]),
        )
        print_stats_line(
            path.stem,
            "wrist",
            left,
            right,
            pairwise_depth_stats(depth_by_encoder[left][n_frames:], depth_by_encoder[right][n_frames:]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{path.stem}_{'_'.join(encoders)}_depth_shared_diff_compare.mp4"
    temp_path = out_path.with_suffix(".tmp.mp4")
    display_size = args.display_size
    extra_compare_columns = 3 if len(encoders) >= 2 else 0
    frame_size = ((1 + len(encoders) + extra_compare_columns) * display_size, 2 * display_size)
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*args.codec), args.fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {temp_path}")
    try:
        for frame_idx in range(n_frames):
            table_depths = {encoder: depth_by_encoder[encoder][frame_idx] for encoder in encoders}
            wrist_depths = {encoder: depth_by_encoder[encoder][n_frames + frame_idx] for encoder in encoders}
            frame_rgb = render_frame(table_400[frame_idx], wrist_400[frame_idx], table_depths, wrist_depths, encoders, display_size)
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if args.no_ffmpeg_h264 or shutil.which("ffmpeg") is None:
        temp_path.replace(out_path)
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(temp_path),
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
        )
        temp_path.unlink(missing_ok=True)
    return out_path


def main() -> None:
    args = parse_args()
    ssi_root = Path(args.ssi_root).resolve()
    depth_root = Path(args.depth_root).resolve()
    encoders = list(args.encoders)
    check_checkpoints(depth_root, encoders)
    setup_imports(ssi_root, depth_root)

    from atm.utils.depth_utils import DepthPreprocessor, get_depth_emb_a800, init_depth_generator

    depth_preprocessor = DepthPreprocessor(input_size=args.depth_input_size)
    depth_generators = {
        encoder: init_depth_generator(str(depth_root), encoder=encoder, return_feat_only=False)
        for encoder in encoders
    }

    output_dir = Path(args.output_dir)
    for dataset_path in resolve_datasets(args):
        print(f"[process] {dataset_path}")
        out_path = process_dataset(
            dataset_path,
            output_dir,
            args,
            encoders,
            depth_generators,
            depth_preprocessor,
            get_depth_emb_a800,
        )
        print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
