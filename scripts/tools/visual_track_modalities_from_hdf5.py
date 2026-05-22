#!/usr/bin/env python3
"""Build a visual-track modality diagnostic video from a UWLab HDF5 demo.

Recommended interpreter:
    /home/emopointer/miniconda3/envs/SimToReal/bin/python \
        scripts/tools/visual_track_modalities_from_hdf5.py

The script intentionally keeps all model-facing image operations explicit:
table view uses the top-right 400x400 crop, wrist view is resized to 400x400,
detection runs on those 400x400 images, bbox coordinates are scaled to 128x128,
DepthAnything is run from the 400x400 images and resized to 128x128, and the
TrackTransformer predicts trajectories on 128x128 images.

The trajectory model uses the BERT task embedding expected by the original
TrackTransformer training setup.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from glob import glob
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf


DEFAULT_DATASET = "/home/emopointer/UWLab/datasets_test/cube_stack_state_policy_demo_000000.hdf5"
DEFAULT_OUTPUT_DIR = "/home/emopointer/UWLab/videos/visual_track_modalities"
DEFAULT_TRAJ_CONFIG = "/home/emopointer/UWLab/logs/trajectory_predict/config.yaml"
DEFAULT_TRAJ_CKPT = "/home/emopointer/UWLab/logs/trajectory_predict/model_final.ckpt"
DEFAULT_SSI_ROOT = "/home/emopointer/SSI-SimToReal"
DEFAULT_SSI_CONFIG = (
    "/home/emopointer/SSI-SimToReal/results/policy/"
    "0417_UWLab_delat_OSC_control_1447_seed42/config.yaml"
)


RGB_COLORS = {
    "robot": (80, 190, 255),
    "red": (255, 40, 40),
    "green": (40, 220, 80),
    "cube": (255, 220, 60),
    "track": (255, 230, 50),
    "point": (0, 255, 255),
    "fallback": (255, 180, 40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize bbox, depth, and TrackTransformer trajectories from a UWLab HDF5 demo."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-glob", default=None, help="Optional glob for batch processing multiple HDF5 files.")
    parser.add_argument("--output", default=None, help="Explicit output mp4 path. Only valid with a single dataset.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-config", default=DEFAULT_TRAJ_CONFIG)
    parser.add_argument("--trajectory-ckpt", default=DEFAULT_TRAJ_CKPT)
    parser.add_argument("--ssi-root", default=DEFAULT_SSI_ROOT)
    parser.add_argument("--ssi-config", default=DEFAULT_SSI_CONFIG)
    parser.add_argument("--table-key", default="obs/table_cam")
    parser.add_argument("--wrist-key", default="obs/wrist_cam")
    parser.add_argument("--table-prompt", default="robot, red cube, green cube")
    parser.add_argument("--wrist-prompt", default="red cube, green cube")
    parser.add_argument("--task-description", default="Put the red block on the green block.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--process-size", type=int, default=400)
    parser.add_argument("--model-size", type=int, default=128)
    parser.add_argument("--viz-points", type=int, default=16)
    parser.add_argument("--display-scale", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-bert-download", action="store_true")
    return parser.parse_args()


def load_task_embedding(task_description: str, cache_dir: Path, device: str, allow_download: bool) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased",
        cache_dir=str(cache_dir),
        local_files_only=not allow_download,
    )
    bert = AutoModel.from_pretrained(
        "bert-base-cased",
        cache_dir=str(cache_dir),
        local_files_only=not allow_download,
    ).eval().to(device)
    tokens = tokenizer(
        text=[task_description],
        add_special_tokens=True,
        max_length=25,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        task_emb = bert(tokens["input_ids"], tokens["attention_mask"])["pooler_output"]
    return task_emb


def setup_ssi_imports(ssi_root: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(ssi_root))
    sys.path.insert(0, str(ssi_root / "Grounded_SAM_2"))
    sys.path.insert(0, str(ssi_root / "Depth_Anything_V2"))


def rgb_to_tensor(images: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device)


def load_hdf5_images(path: Path, table_key: str, wrist_key: str, max_frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        table = f[table_key][:]
        wrist = f[wrist_key][:]
    if max_frames is not None:
        table = table[:max_frames]
        wrist = wrist[:max_frames]
    if table.shape[0] != wrist.shape[0]:
        raise ValueError(f"table/wrist frame count mismatch: {table.shape[0]} vs {wrist.shape[0]}")
    return table, wrist


def preprocess_table_view(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if h < size or w < size:
        raise ValueError(f"table image {frame_rgb.shape} is smaller than requested crop {size}")
    return frame_rgb[:size, w - size : w].copy()


def preprocess_wrist_view(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)


def load_models(args: argparse.Namespace):
    ssi_root = Path(args.ssi_root).resolve()
    setup_ssi_imports(ssi_root)
    os.chdir(ssi_root)

    from atm.model import TrackTransformer
    from atm.utils.depth_utils import DepthPreprocessor, get_depth_emb_a800, init_depth_generator
    from atm.utils.ground_dino_utils import Grounded_SAM2

    ssi_cfg = OmegaConf.load(args.ssi_config)

    print("[load] Grounded-SAM2 / GroundingDINO")
    detector = Grounded_SAM2(ssi_cfg.mask_cfg, init_grounding_dino=True)
    detector_device = next(detector.detector.parameters()).device if hasattr(detector, "detector") else torch.device(args.device)

    print("[load] DepthAnything")
    depth_root = Path(ssi_cfg.depth_cfg.depth_generator_root)
    if not depth_root.is_absolute():
        depth_root = ssi_root / depth_root
    depth_generator = init_depth_generator(
        str(depth_root),
        encoder=ssi_cfg.depth_cfg.encoder,
        return_feat_only=False,
    )
    depth_device = next(depth_generator.parameters()).device
    depth_preprocessor = DepthPreprocessor()

    print("[load] TrackTransformer")
    traj_cfg = OmegaConf.load(args.trajectory_config)
    track_model = TrackTransformer(**traj_cfg.model_cfg).to(args.device).eval()
    state = torch.load(args.trajectory_ckpt, map_location="cpu")
    track_model.load_state_dict(state)
    for param in track_model.parameters():
        param.requires_grad = False

    print(f"[load] BERT task embedding: {args.task_description!r}")
    task_emb = load_task_embedding(
        args.task_description,
        cache_dir=ssi_root / "data" / "bert_cache",
        device=args.device,
        allow_download=args.allow_bert_download,
    )

    return {
        "detector": detector,
        "detector_device": detector_device,
        "depth_generator": depth_generator,
        "depth_device": depth_device,
        "depth_preprocessor": depth_preprocessor,
        "get_depth_emb_a800": get_depth_emb_a800,
        "track_model": track_model,
        "task_emb": task_emb,
        "traj_cfg": traj_cfg,
    }


def batch_detect(detector, images: np.ndarray, prompt: str, device: str) -> list[dict]:
    if len(images) == 0:
        return []
    tensor = rgb_to_tensor(images, device)
    prompts = [prompt] * len(images)
    boxes, confidences, labels = detector.predict_batch(
        tensor,
        text_prompts=prompts,
        only_generate_bbox=True,
    )
    results = []
    for box_i, conf_i, label_i in zip(boxes, confidences, labels):
        box_arr = np.asarray(box_i, dtype=np.float32).reshape(-1, 4)
        if isinstance(conf_i, torch.Tensor):
            conf_arr = conf_i.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            conf_arr = np.asarray(conf_i, dtype=np.float32).reshape(-1)
        label_list = list(label_i) if isinstance(label_i, Iterable) and not isinstance(label_i, str) else [str(label_i)]
        if len(label_list) < len(box_arr):
            label_list += [""] * (len(box_arr) - len(label_list))
        results.append({"boxes": box_arr, "confidences": conf_arr, "labels": label_list[: len(box_arr)]})
    return results


def scale_boxes(boxes_xyxy: np.ndarray, src_size: int, dst_size: int) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy.reshape(0, 4).astype(np.float32)
    boxes = boxes_xyxy.astype(np.float32).copy()
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, src_size)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, src_size)
    boxes *= float(dst_size) / float(src_size)
    return boxes


def box_color(label: str) -> tuple[int, int, int]:
    text = label.lower()
    if "red" in text:
        return RGB_COLORS["red"]
    if "green" in text:
        return RGB_COLORS["green"]
    if "robot" in text or "arm" in text:
        return RGB_COLORS["robot"]
    if "cube" in text or "block" in text:
        return RGB_COLORS["cube"]
    return RGB_COLORS["fallback"]


def draw_boxes(image_rgb: np.ndarray, boxes: np.ndarray, labels: list[str], confidences: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = np.round(box).astype(int)
        if x2 <= x1 or y2 <= y1:
            continue
        label = labels[i] if i < len(labels) else ""
        conf = confidences[i] if i < len(confidences) else 0.0
        color = box_color(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
        if label:
            text = f"{label}:{conf:.2f}"
            cv2.putText(out, text, (x1, max(9, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
    return out


def sample_points_in_boxes(boxes: np.ndarray, num_points: int, image_size: int) -> np.ndarray:
    valid = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(np.float32)
        if x2 > x1 + 1 and y2 > y1 + 1:
            valid.append((x1, y1, x2, y2))

    if not valid:
        grid = np.linspace(0.2, 0.8, int(np.sqrt(num_points)), dtype=np.float32)
        points = np.array([(x * image_size, y * image_size) for y in grid for x in grid], dtype=np.float32)
        return points[:num_points]

    points = []
    for point_idx in range(num_points):
        x1, y1, x2, y2 = valid[point_idx % len(valid)]
        local = point_idx // len(valid)
        gx = (local % 4 + 0.5) / 4.0
        gy = ((local // 4) % 4 + 0.5) / 4.0
        x = x1 + gx * (x2 - x1)
        y = y1 + gy * (y2 - y1)
        points.append((x, y))
    return np.asarray(points, dtype=np.float32)


@torch.no_grad()
def predict_tracks(track_model, image128_rgb: np.ndarray, points128: np.ndarray, task_emb: torch.Tensor, device: str) -> np.ndarray:
    num_model_points = int(track_model.num_track_ids)
    if len(points128) < num_model_points:
        repeats = int(np.ceil(num_model_points / max(len(points128), 1)))
        model_points = np.tile(points128, (repeats, 1))[:num_model_points]
    else:
        model_points = points128[:num_model_points]

    query = torch.from_numpy(model_points / 128.0).float().to(device)
    query = query.unsqueeze(0).unsqueeze(0).repeat(1, int(track_model.num_track_ts), 1, 1)
    vid = torch.from_numpy(image128_rgb).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
    pred, _ = track_model.reconstruct(vid, query, task_emb, p_img=0)
    pred = pred[0, :, : len(points128)].detach().cpu().numpy() * 128.0
    pred[0, :, :] = points128
    return pred


def draw_points_and_tracks(image_rgb: np.ndarray, points: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    n = len(points)
    for point_idx in range(n):
        hue = int(180 * point_idx / max(n, 1))
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        pts = np.round(tracks[:, point_idx]).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, image_rgb.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, image_rgb.shape[0] - 1)
        if len(pts) >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        x, y = np.round(points[point_idx]).astype(int)
        cv2.circle(out, (int(np.clip(x, 0, 127)), int(np.clip(y, 0, 127))), 2, RGB_COLORS["point"], -1, cv2.LINE_AA)
    return out


def depth_to_rgb(depth_128: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_128, dtype=np.float32)
    d_min = float(np.nanmin(depth))
    d_max = float(np.nanmax(depth))
    if d_max - d_min < 1e-8:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalized = ((depth - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def compute_depth_batch(models: dict, table_400: np.ndarray, wrist_400: np.ndarray, model_size: int) -> np.ndarray:
    images = np.stack([table_400, wrist_400], axis=1)  # B, V, H, W, C
    tensor = torch.from_numpy(images).float().permute(0, 1, 4, 2, 3).unsqueeze(2)
    tensor = tensor.to(models["depth_device"])
    depth = models["get_depth_emb_a800"](
        models["depth_generator"],
        models["depth_preprocessor"],
        tensor,
        n_obs_steps=1,
        target_size=(model_size, model_size),
    )
    return depth[:, :, 0, 0].detach().cpu().numpy()


def make_composite(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray, scale: int) -> np.ndarray:
    top = np.concatenate([tl, tr], axis=1)
    bottom = np.concatenate([bl, br], axis=1)
    frame = np.concatenate([top, bottom], axis=0)
    if scale > 1:
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return frame


def resolve_datasets(args: argparse.Namespace) -> list[Path]:
    if args.dataset_glob:
        datasets = [Path(path) for path in sorted(glob(args.dataset_glob))]
        if not datasets:
            raise FileNotFoundError(f"no HDF5 files matched --dataset-glob {args.dataset_glob!r}")
        return datasets
    return [Path(args.dataset)]


def resolve_output_path(args: argparse.Namespace, dataset_path: Path, multiple: bool) -> Path:
    if args.output is not None:
        if multiple:
            raise ValueError("--output cannot be used with --dataset-glob; use --output-dir instead.")
        return Path(args.output)

    dataset_stem = dataset_path.stem
    return Path(args.output_dir) / f"{dataset_stem}_visual_track_modalities.mp4"


def process_dataset(args: argparse.Namespace, models: dict, dataset_path: Path, output_path: Path) -> None:
    dataset_t0 = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.model_size != 128:
        raise ValueError("This script currently assumes the trajectory model uses 128x128 inputs.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    table_raw, wrist_raw = load_hdf5_images(dataset_path, args.table_key, args.wrist_key, args.max_frames)
    num_frames = table_raw.shape[0]
    print(f"[data] loaded {num_frames} frames from {dataset_path}")
    read_s = time.perf_counter() - dataset_t0
    timings = {
        "read": read_s,
        "preprocess": 0.0,
        "detect": 0.0,
        "depth": 0.0,
        "render": 0.0,
        "write": 0.0,
    }

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        for start in range(0, num_frames, args.batch_size):
            end = min(start + args.batch_size, num_frames)
            preprocess_t0 = time.perf_counter()
            table_400 = np.stack([preprocess_table_view(frame, args.process_size) for frame in table_raw[start:end]])
            wrist_400 = np.stack([preprocess_wrist_view(frame, args.process_size) for frame in wrist_raw[start:end]])
            preprocess_s = time.perf_counter() - preprocess_t0
            timings["preprocess"] += preprocess_s

            print(f"[batch] frames {start}-{end - 1}: detection")
            detect_t0 = time.perf_counter()
            detector_device = str(models["detector_device"])
            table_det = batch_detect(models["detector"], table_400, args.table_prompt, detector_device)
            wrist_det = batch_detect(models["detector"], wrist_400, args.wrist_prompt, detector_device)
            detect_s = time.perf_counter() - detect_t0
            timings["detect"] += detect_s

            print(f"[batch] frames {start}-{end - 1}: depth")
            depth_t0 = time.perf_counter()
            depth = compute_depth_batch(models, table_400, wrist_400, args.model_size)
            depth_s = time.perf_counter() - depth_t0
            timings["depth"] += depth_s

            render_t0 = time.perf_counter()
            batch_write_s = 0.0

            for local_idx in range(end - start):
                table_128 = cv2.resize(table_400[local_idx], (args.model_size, args.model_size), interpolation=cv2.INTER_AREA)
                wrist_128 = cv2.resize(wrist_400[local_idx], (args.model_size, args.model_size), interpolation=cv2.INTER_AREA)

                table_boxes = scale_boxes(table_det[local_idx]["boxes"], args.process_size, args.model_size)
                wrist_boxes = scale_boxes(wrist_det[local_idx]["boxes"], args.process_size, args.model_size)

                table_points = sample_points_in_boxes(table_boxes, args.viz_points, args.model_size)
                wrist_points = sample_points_in_boxes(wrist_boxes, args.viz_points, args.model_size)

                table_tracks = predict_tracks(models["track_model"], table_128, table_points, models["task_emb"], args.device)
                wrist_tracks = predict_tracks(models["track_model"], wrist_128, wrist_points, models["task_emb"], args.device)

                table_vis = draw_boxes(
                    table_128,
                    table_boxes,
                    table_det[local_idx]["labels"],
                    table_det[local_idx]["confidences"],
                )
                table_vis = draw_points_and_tracks(table_vis, table_points, table_tracks)

                wrist_vis = draw_boxes(
                    wrist_128,
                    wrist_boxes,
                    wrist_det[local_idx]["labels"],
                    wrist_det[local_idx]["confidences"],
                )
                wrist_vis = draw_points_and_tracks(wrist_vis, wrist_points, wrist_tracks)

                table_depth = depth_to_rgb(depth[local_idx, 0])
                wrist_depth = depth_to_rgb(depth[local_idx, 1])

                composite_rgb = make_composite(table_vis, table_depth, wrist_vis, wrist_depth, args.display_scale)
                if writer is None:
                    h, w = composite_rgb.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open video writer for {output_path}")
                write_t0 = time.perf_counter()
                writer.write(cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR))
                batch_write_s += time.perf_counter() - write_t0
            render_s = time.perf_counter() - render_t0 - batch_write_s
            timings["render"] += render_s
            timings["write"] += batch_write_s
            batch_frames = end - start
            batch_images = max(batch_frames * 2, 1)
            batch_total_s = preprocess_s + detect_s + depth_s + render_s + batch_write_s
            print(
                "[timing] frames "
                f"{start}-{end - 1}: "
                f"total={batch_total_s / batch_frames:.3f}s/frame "
                f"({batch_total_s / batch_images:.3f}s/image), "
                f"preprocess={preprocess_s / batch_frames:.3f}s/frame, "
                f"detect={detect_s / batch_frames:.3f}s/frame, "
                f"depth={depth_s / batch_frames:.3f}s/frame, "
                f"render={render_s / batch_frames:.3f}s/frame, "
                f"write={batch_write_s / batch_frames:.3f}s/frame"
            )
    finally:
        if writer is not None:
            writer.release()

    total_s = time.perf_counter() - dataset_t0
    denom = max(num_frames, 1)
    image_denom = max(num_frames * 2, 1)
    print(f"[done] wrote {output_path}")
    print(
        "[timing] summary: "
        f"frames={num_frames}, images={num_frames * 2}, total={total_s:.3f}s, "
        f"avg={total_s / denom:.3f}s/frame, avg_image={total_s / image_denom:.3f}s/image, "
        f"read={timings['read']:.3f}s ({timings['read'] / denom:.3f}s/frame), "
        f"preprocess={timings['preprocess']:.3f}s ({timings['preprocess'] / denom:.3f}s/frame), "
        f"detect={timings['detect']:.3f}s ({timings['detect'] / denom:.3f}s/frame), "
        f"depth={timings['depth']:.3f}s ({timings['depth'] / denom:.3f}s/frame), "
        f"render={timings['render']:.3f}s ({timings['render'] / denom:.3f}s/frame), "
        f"write={timings['write']:.3f}s ({timings['write'] / denom:.3f}s/frame)"
    )


def main() -> None:
    args = parse_args()
    run_t0 = time.perf_counter()
    dataset_paths = resolve_datasets(args)
    multiple = len(dataset_paths) > 1

    model_t0 = time.perf_counter()
    models = load_models(args)
    model_load_s = time.perf_counter() - model_t0
    print(f"[timing] model_load={model_load_s:.3f}s")
    for dataset_path in dataset_paths:
        output_path = resolve_output_path(args, dataset_path, multiple)
        process_dataset(args, models, dataset_path, output_path)
    print(f"[timing] all_done total_wall={time.perf_counter() - run_t0:.3f}s")


if __name__ == "__main__":
    main()
