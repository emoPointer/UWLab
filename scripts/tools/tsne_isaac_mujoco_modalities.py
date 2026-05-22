#!/usr/bin/env python3
"""Compare Isaac HDF5 and MuJoCo replay visual modalities with t-SNE.

This script reuses the same environment-view image processing used by
``visual_track_modalities_from_hdf5.py``:

- Isaac HDF5 env camera: top-right 400x400 crop from ``obs/table_cam``.
- MuJoCo replay video: top-right 400x400 crop from each RGB video frame.
- Grounded-SAM2/GroundingDINO prompt: ``robot, red cube, green cube``.
- Bbox and trajectory coordinates live in the 128x128 model space.
- DepthAnything runs from the 400x400 image and is resized to 128x128.

Outputs:

- ``modalities_features.npz``: bbox, trajectory, and depth features.
- ``isaac_mujoco_modalities_tsne.png``: one figure with bbox, trajectory,
  and depth t-SNE plots.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools import visual_track_modalities_from_hdf5 as vt  # noqa: E402


DEFAULT_HDF5_DIR = "/home/emopointer/UWLab/datasets_test"
DEFAULT_MUJOCO_VIDEO_DIR = "/home/emopointer/UWLab/videos/mujoco_isaac_replays"
DEFAULT_OUTPUT_DIR = "/home/emopointer/UWLab/videos/modality_tsne"
MUJOCO_SUFFIX = "_isaac_physics_mujoco_render"
TARGET_SLOTS = ("robot", "red cube", "green cube")


@dataclass(frozen=True)
class PairedDemo:
    demo_id: str
    hdf5_path: Path
    mujoco_video_path: Path


@dataclass(frozen=True)
class SourceDemo:
    domain: str
    demo_id: str
    path: Path


@dataclass(frozen=True)
class SampleMeta:
    domain: str
    demo_id: str
    frame_index: int
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bbox/trajectory/depth t-SNE for Isaac vs MuJoCo env views.")
    parser.add_argument("--hdf5-dir", default=DEFAULT_HDF5_DIR)
    parser.add_argument("--mujoco-video-dir", default=DEFAULT_MUJOCO_VIDEO_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-key", default="obs/table_cam")
    parser.add_argument("--prompt", default="robot, red cube, green cube")
    parser.add_argument("--task-description", default="Put the red block on the green block.")
    parser.add_argument("--trajectory-config", default=vt.DEFAULT_TRAJ_CONFIG)
    parser.add_argument("--trajectory-ckpt", default=vt.DEFAULT_TRAJ_CKPT)
    parser.add_argument("--ssi-root", default=vt.DEFAULT_SSI_ROOT)
    parser.add_argument("--ssi-config", default=vt.DEFAULT_SSI_CONFIG)
    parser.add_argument("--process-size", type=int, default=400)
    parser.add_argument("--model-size", type=int, default=128)
    parser.add_argument(
        "--depth-feature-size",
        type=int,
        default=128,
        help="Depth feature resolution. Default 128 keeps the raw 128x128 DepthAnything output.",
    )
    parser.add_argument("--viz-points", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit demos per domain, or paired demos with --paired-only.")
    parser.add_argument("--max-frames-per-pair", type=int, default=None, help="Backward-compatible alias.")
    parser.add_argument("--max-frames-per-source", type=int, default=None)
    parser.add_argument(
        "--paired-only",
        action="store_true",
        default=False,
        help="Use only paired HDF5/video demos and trim each pair to the common frame count.",
    )
    parser.add_argument("--allow-bert-download", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def demo_id_from_hdf5(path: Path) -> str:
    return path.stem


def demo_id_from_mujoco_video(path: Path) -> str:
    stem = path.stem
    if stem.endswith(MUJOCO_SUFFIX):
        return stem[: -len(MUJOCO_SUFFIX)]
    return stem


def resolve_pairs(hdf5_dir: Path, mujoco_video_dir: Path, max_pairs: int | None) -> list[PairedDemo]:
    hdf5_by_id = {
        demo_id_from_hdf5(path): path
        for path in sorted(list(hdf5_dir.glob("*.hdf5")) + list(hdf5_dir.glob("*.h5")))
    }
    video_by_id = {
        demo_id_from_mujoco_video(path): path
        for path in sorted(mujoco_video_dir.glob("*.mp4"))
    }
    paired_ids = sorted(set(hdf5_by_id).intersection(video_by_id), key=_natural_key)
    if max_pairs is not None:
        paired_ids = paired_ids[:max_pairs]
    if not paired_ids:
        raise FileNotFoundError(
            f"no paired demos found between HDF5 dir {hdf5_dir} and MuJoCo video dir {mujoco_video_dir}"
        )
    missing_videos = sorted(set(hdf5_by_id) - set(video_by_id), key=_natural_key)
    if missing_videos:
        print(f"[WARN] {len(missing_videos)} HDF5 file(s) have no matching MuJoCo video; first={missing_videos[0]}")
    return [PairedDemo(demo_id=demo_id, hdf5_path=hdf5_by_id[demo_id], mujoco_video_path=video_by_id[demo_id]) for demo_id in paired_ids]


def resolve_sources(hdf5_dir: Path, mujoco_video_dir: Path, max_sources: int | None) -> list[SourceDemo]:
    hdf5_paths = sorted(list(hdf5_dir.glob("*.hdf5")) + list(hdf5_dir.glob("*.h5")), key=lambda p: _natural_key(p.stem))
    mujoco_paths = sorted(mujoco_video_dir.glob("*.mp4"), key=lambda p: _natural_key(p.stem))
    if max_sources is not None:
        hdf5_paths = hdf5_paths[:max_sources]
        mujoco_paths = mujoco_paths[:max_sources]
    if not hdf5_paths:
        raise FileNotFoundError(f"no HDF5 files found in {hdf5_dir}")
    if not mujoco_paths:
        raise FileNotFoundError(f"no MuJoCo videos found in {mujoco_video_dir}")
    sources = [SourceDemo("isaac", demo_id_from_hdf5(path), path) for path in hdf5_paths]
    sources.extend(SourceDemo("mujoco", demo_id_from_mujoco_video(path), path) for path in mujoco_paths)
    return sources


def _natural_key(text: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def read_isaac_env_frames(path: Path, key: str, frame_stride: int, max_frames: int | None) -> list[tuple[int, np.ndarray]]:
    with h5py.File(path, "r") as h5_file:
        frames = h5_file[key]
        frame_count = int(frames.shape[0])
        selected = _selected_frame_indices(frame_count, frame_stride, max_frames)
        return [(frame_idx, frames[frame_idx][..., :3].astype(np.uint8, copy=False)) for frame_idx in selected]


def read_mujoco_video_frames(path: Path, frame_stride: int, max_frames: int | None) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = set(_selected_frame_indices(frame_count, frame_stride, max_frames))
    out: list[tuple[int, np.ndarray]] = []
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx in selected:
                out.append((frame_idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
            frame_idx += 1
    finally:
        cap.release()
    return out


def _selected_frame_indices(frame_count: int, frame_stride: int, max_frames: int | None) -> list[int]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    indices = list(range(0, frame_count, frame_stride))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def preprocess_env_view(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    return vt.preprocess_table_view(frame_rgb, size)


def compute_depth_env_batch(models: dict, images_400: np.ndarray, model_size: int) -> np.ndarray:
    images = images_400[:, None]  # B, V, H, W, C
    tensor = torch.from_numpy(images).float().permute(0, 1, 4, 2, 3).unsqueeze(2)
    tensor = tensor.to(models["depth_device"])
    depth = models["get_depth_emb_a800"](
        models["depth_generator"],
        models["depth_preprocessor"],
        tensor,
        n_obs_steps=1,
        target_size=(model_size, model_size),
    )
    return depth[:, 0, 0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def bbox_feature(boxes_128: np.ndarray, labels: list[str], confidences: np.ndarray, image_size: int) -> np.ndarray:
    features = np.zeros((len(TARGET_SLOTS), 5), dtype=np.float32)
    best_conf = np.full((len(TARGET_SLOTS),), -np.inf, dtype=np.float32)
    for box, label, conf in zip(boxes_128, labels, confidences):
        slot = slot_index(label)
        if slot is None or conf <= best_conf[slot]:
            continue
        x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
        x1 = float(np.clip(x1, 0, image_size))
        y1 = float(np.clip(y1, 0, image_size))
        x2 = float(np.clip(x2, 0, image_size))
        y2 = float(np.clip(y2, 0, image_size))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = 0.5 * (x1 + x2) / image_size
        cy = 0.5 * (y1 + y2) / image_size
        width = (x2 - x1) / image_size
        height = (y2 - y1) / image_size
        features[slot] = np.asarray([cx, cy, width, height, float(conf)], dtype=np.float32)
        best_conf[slot] = float(conf)
    return features.reshape(-1)


def slot_index(label: str) -> int | None:
    text = label.lower()
    if "red" in text and ("cube" in text or "block" in text):
        return TARGET_SLOTS.index("red cube")
    if "green" in text and ("cube" in text or "block" in text):
        return TARGET_SLOTS.index("green cube")
    if "robot" in text or "arm" in text or "gripper" in text:
        return TARGET_SLOTS.index("robot")
    return None


def depth_feature(depth_128: np.ndarray, feature_size: int) -> np.ndarray:
    depth = np.asarray(depth_128, dtype=np.float32)
    if feature_size != depth.shape[0]:
        depth = cv2.resize(depth, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
    d_min = float(np.nanmin(depth))
    d_max = float(np.nanmax(depth))
    if d_max - d_min > 1.0e-8:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)
    return depth.reshape(-1).astype(np.float32, copy=False)


def trajectory_feature(tracks: np.ndarray, image_size: int) -> np.ndarray:
    tracks = np.asarray(tracks, dtype=np.float32).copy()
    tracks[..., 0] = np.clip(tracks[..., 0], 0, image_size)
    tracks[..., 1] = np.clip(tracks[..., 1], 0, image_size)
    return (tracks / float(image_size)).reshape(-1).astype(np.float32, copy=False)


def process_batch(
    models: dict,
    images_400: list[np.ndarray],
    metas: list[SampleMeta],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[SampleMeta]]:
    if not images_400:
        return [], [], [], []
    images_np = np.stack(images_400, axis=0)
    detector_device = str(models["detector_device"])

    detections = vt.batch_detect(models["detector"], images_np, args.prompt, detector_device)
    depths = compute_depth_env_batch(models, images_np, args.model_size)

    bbox_features: list[np.ndarray] = []
    trajectory_features: list[np.ndarray] = []
    depth_features: list[np.ndarray] = []
    out_metas: list[SampleMeta] = []

    for local_idx, image_400 in enumerate(images_np):
        image_128 = cv2.resize(image_400, (args.model_size, args.model_size), interpolation=cv2.INTER_AREA)
        det = detections[local_idx]
        boxes_128 = vt.scale_boxes(det["boxes"], args.process_size, args.model_size)
        points = vt.sample_points_in_boxes(boxes_128, args.viz_points, args.model_size)
        tracks = vt.predict_tracks(models["track_model"], image_128, points, models["task_emb"], args.device)

        bbox_features.append(bbox_feature(boxes_128, det["labels"], det["confidences"], args.model_size))
        trajectory_features.append(trajectory_feature(tracks, args.model_size))
        depth_features.append(depth_feature(depths[local_idx], args.depth_feature_size))
        out_metas.append(metas[local_idx])

    return bbox_features, trajectory_features, depth_features, out_metas


def extract_features(pairs: list[PairedDemo], models: dict, args: argparse.Namespace) -> dict[str, np.ndarray]:
    bbox_features: list[np.ndarray] = []
    trajectory_features: list[np.ndarray] = []
    depth_features: list[np.ndarray] = []
    metas: list[SampleMeta] = []
    pending_images: list[np.ndarray] = []
    pending_metas: list[SampleMeta] = []
    start_t = time.perf_counter()

    def flush() -> None:
        nonlocal pending_images, pending_metas
        b, tr, d, m = process_batch(models, pending_images, pending_metas, args)
        bbox_features.extend(b)
        trajectory_features.extend(tr)
        depth_features.extend(d)
        metas.extend(m)
        pending_images = []
        pending_metas = []

    for pair_idx, pair in enumerate(pairs, start=1):
        print(f"[data] pair {pair_idx}/{len(pairs)} {pair.demo_id}")
        isaac_frames = read_isaac_env_frames(pair.hdf5_path, args.table_key, args.frame_stride, args.max_frames_per_pair)
        mujoco_frames = read_mujoco_video_frames(pair.mujoco_video_path, args.frame_stride, args.max_frames_per_pair)
        target_count = min(len(isaac_frames), len(mujoco_frames))
        if len(isaac_frames) != len(mujoco_frames):
            print(
                f"[WARN] frame count mismatch for {pair.demo_id}: "
                f"isaac={len(isaac_frames)} mujoco={len(mujoco_frames)}; using {target_count}"
            )

        for domain, frames, source_path in (
            ("isaac", isaac_frames[:target_count], pair.hdf5_path),
            ("mujoco", mujoco_frames[:target_count], pair.mujoco_video_path),
        ):
            for frame_idx, frame_rgb in frames:
                pending_images.append(preprocess_env_view(frame_rgb, args.process_size))
                pending_metas.append(
                    SampleMeta(
                        domain=domain,
                        demo_id=pair.demo_id,
                        frame_index=frame_idx,
                        source_path=str(source_path),
                    )
                )
                if len(pending_images) >= args.batch_size:
                    flush()

    flush()
    print(f"[timing] feature extraction {time.perf_counter() - start_t:.3f}s for {len(metas)} samples")
    return {
        "bbox": np.stack(bbox_features, axis=0),
        "trajectory": np.stack(trajectory_features, axis=0),
        "depth": np.stack(depth_features, axis=0),
        "domain": np.asarray([meta.domain for meta in metas]),
        "demo_id": np.asarray([meta.demo_id for meta in metas]),
        "frame_index": np.asarray([meta.frame_index for meta in metas], dtype=np.int32),
        "source_path": np.asarray([meta.source_path for meta in metas]),
    }


def extract_all_features(sources: list[SourceDemo], models: dict, args: argparse.Namespace) -> dict[str, np.ndarray]:
    bbox_features: list[np.ndarray] = []
    trajectory_features: list[np.ndarray] = []
    depth_features: list[np.ndarray] = []
    metas: list[SampleMeta] = []
    pending_images: list[np.ndarray] = []
    pending_metas: list[SampleMeta] = []
    start_t = time.perf_counter()
    max_frames = args.max_frames_per_source
    if max_frames is None:
        max_frames = args.max_frames_per_pair

    def flush() -> None:
        nonlocal pending_images, pending_metas
        b, tr, d, m = process_batch(models, pending_images, pending_metas, args)
        bbox_features.extend(b)
        trajectory_features.extend(tr)
        depth_features.extend(d)
        metas.extend(m)
        pending_images = []
        pending_metas = []

    for source_idx, source in enumerate(sources, start=1):
        print(f"[data] source {source_idx}/{len(sources)} {source.domain} {source.demo_id}")
        if source.domain == "isaac":
            frames = read_isaac_env_frames(source.path, args.table_key, args.frame_stride, max_frames)
        elif source.domain == "mujoco":
            frames = read_mujoco_video_frames(source.path, args.frame_stride, max_frames)
        else:
            raise ValueError(f"unknown domain: {source.domain}")

        for frame_idx, frame_rgb in frames:
            pending_images.append(preprocess_env_view(frame_rgb, args.process_size))
            pending_metas.append(
                SampleMeta(
                    domain=source.domain,
                    demo_id=source.demo_id,
                    frame_index=frame_idx,
                    source_path=str(source.path),
                )
            )
            if len(pending_images) >= args.batch_size:
                flush()

    flush()
    print(f"[timing] feature extraction {time.perf_counter() - start_t:.3f}s for {len(metas)} samples")
    return {
        "bbox": np.stack(bbox_features, axis=0),
        "trajectory": np.stack(trajectory_features, axis=0),
        "depth": np.stack(depth_features, axis=0),
        "domain": np.asarray([meta.domain for meta in metas]),
        "demo_id": np.asarray([meta.demo_id for meta in metas]),
        "frame_index": np.asarray([meta.frame_index for meta in metas], dtype=np.int32),
        "source_path": np.asarray([meta.source_path for meta in metas]),
    }


def run_tsne(features: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    if features.shape[0] < 3:
        raise ValueError("need at least 3 samples for t-SNE")
    x = StandardScaler().fit_transform(features)
    n_components = min(50, x.shape[1], x.shape[0] - 1)
    if n_components >= 2:
        x = PCA(n_components=n_components, random_state=seed).fit_transform(x)
    effective_perplexity = min(float(perplexity), max(1.0, (features.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x)


def save_tsne_figure(feature_data: dict[str, np.ndarray], output_path: Path, seed: int, perplexity: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    domains = feature_data["domain"]
    modalities = (
        ("bbox", "BBox"),
        ("trajectory", "Trajectory"),
        ("depth", "Depth"),
    )
    colors = {"isaac": "#2563eb", "mujoco": "#dc2626"}
    markers = {"isaac": "o", "mujoco": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (key, title) in zip(axes, modalities):
        emb = run_tsne(feature_data[key], seed=seed, perplexity=perplexity)
        for domain in ("isaac", "mujoco"):
            mask = domains == domain
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=12,
                c=colors[domain],
                marker=markers[domain],
                alpha=0.68,
                linewidths=0,
                label=f"{domain} ({int(mask.sum())})",
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.18)
        ax.legend(loc="best", frameon=True, fontsize=8)

    fig.suptitle("Isaac HDF5 vs MuJoCo Replay: BBox / Trajectory / Depth t-SNE", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.model_size != 128:
        raise ValueError("TrackTransformer path assumes --model-size 128.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_dir = Path(args.hdf5_dir).expanduser()
    mujoco_video_dir = Path(args.mujoco_video_dir).expanduser()
    if args.paired_only:
        pairs = resolve_pairs(hdf5_dir, mujoco_video_dir, args.max_pairs)
        print(f"[data] paired demos={len(pairs)}")
    else:
        sources = resolve_sources(hdf5_dir, mujoco_video_dir, args.max_pairs)
        isaac_count = sum(source.domain == "isaac" for source in sources)
        mujoco_count = sum(source.domain == "mujoco" for source in sources)
        print(f"[data] mixed sources: isaac={isaac_count} mujoco={mujoco_count}")

    model_args = argparse.Namespace(
        ssi_root=args.ssi_root,
        ssi_config=args.ssi_config,
        trajectory_config=args.trajectory_config,
        trajectory_ckpt=args.trajectory_ckpt,
        task_description=args.task_description,
        allow_bert_download=args.allow_bert_download,
        device=args.device,
    )
    models = vt.load_models(model_args)

    if args.paired_only:
        feature_data = extract_features(pairs, models, args)
    else:
        feature_data = extract_all_features(sources, models, args)
    feature_path = output_dir / "modalities_features.npz"
    np.savez_compressed(feature_path, **feature_data)
    print(f"[done] wrote features: {feature_path}")

    figure_path = output_dir / "isaac_mujoco_modalities_tsne.png"
    save_tsne_figure(feature_data, figure_path, seed=args.seed, perplexity=args.tsne_perplexity)
    print(f"[done] wrote t-SNE figure: {figure_path}")


if __name__ == "__main__":
    main()
