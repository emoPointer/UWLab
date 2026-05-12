#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from mujoco_arx5.controllers import Arx5OperationalSpaceController
from mujoco_arx5.control_alignment import control_period_seconds


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "mujoco_arx5" / "models" / "arx5_robosuite_tabletop_dynamic.xml"
DEFAULT_DATASET = REPO_ROOT / "datasets_test" / "cube_stack_state_policy_demo_000000.hdf5"
DEFAULT_VIDEO_DIR = REPO_ROOT / "videos" / "mujoco_replays"
DEFAULT_BASE_POSE = np.array([-0.535, -0.21, 0.8], dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay UWLab HDF5 7D policy actions on the MuJoCo ARX5 robot.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--keyframe", type=str, default="isaac_default")
    parser.add_argument("--decimation", type=int, default=12)
    parser.add_argument("--viewer", action="store_true", help="Show an interactive MuJoCo viewer while replaying.")
    parser.add_argument("--real-time", action="store_true", default=True)
    parser.add_argument("--no-real-time", action="store_false", dest="real_time")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--camera", type=str, default="external_camera")
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true", help="Disable environment-camera video recording.")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument(
        "--align-initial-eef",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shift the robot root xyz so the initial MuJoCo link6 position matches the HDF5 first-frame EEF pose.",
    )
    args = parser.parse_args()
    if args.viewer and not args.no_video:
        parser.error(
            "--viewer cannot be combined with video recording in one process. "
            "Use no --viewer to record video, or pass --no-video when opening the viewer."
        )
    return args


def _default_video_path(dataset_path: Path) -> Path:
    return DEFAULT_VIDEO_DIR / f"{dataset_path.stem}_mujoco_external_camera.mp4"


def _read_actions(dataset_path: Path) -> np.ndarray:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"HDF5 dataset does not exist: {dataset_path}")
    with h5py.File(dataset_path, "r") as h5_file:
        if "actions" not in h5_file:
            raise KeyError(f"{dataset_path} does not contain an 'actions' dataset.")
        actions = h5_file["actions"][:].astype(np.float64, copy=False)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions with shape (T, 7), got {actions.shape}.")
    return actions


def _read_recorded_initial_eef(dataset_path: Path) -> np.ndarray | None:
    with h5py.File(dataset_path, "r") as h5_file:
        if "obs/eef_pos" not in h5_file:
            return None
        eef_world = h5_file["obs/eef_pos"][0].astype(np.float64, copy=False)
        source_origin = np.asarray(h5_file.attrs.get("source_env_origin", np.zeros(3)), dtype=np.float64)
        return eef_world - source_origin


def _read_recorded_eef_trajectory(dataset_path: Path) -> np.ndarray | None:
    with h5py.File(dataset_path, "r") as h5_file:
        if "obs/eef_pos" not in h5_file:
            return None
        eef_world = h5_file["obs/eef_pos"][:].astype(np.float64, copy=False)
        source_origin = np.asarray(h5_file.attrs.get("source_env_origin", np.zeros(3)), dtype=np.float64)
        return eef_world - source_origin.reshape(1, 3)


def _reset_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData, keyframe: str) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        raise ValueError(f"MuJoCo model does not contain keyframe {keyframe!r}.")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def _align_robot_root_to_recorded_eef(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Arx5OperationalSpaceController,
    recorded_eef_pos: np.ndarray | None,
) -> None:
    if recorded_eef_pos is None:
        return
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arx5")
    if root_body_id < 0:
        return

    current_eef_pos = controller.get_ee_pose(data).position
    root_shift = recorded_eef_pos - current_eef_pos
    model.body_pos[root_body_id] = DEFAULT_BASE_POSE + root_shift
    mujoco.mj_forward(model, data)
    print(
        f"[INFO] aligned robot root by ({root_shift[0]:+.4f}, {root_shift[1]:+.4f}, {root_shift[2]:+.4f}) "
        f"to match recorded initial eef=({recorded_eef_pos[0]:+.4f}, {recorded_eef_pos[1]:+.4f}, {recorded_eef_pos[2]:+.4f})",
        flush=True,
    )


def _print_step(step_id: int, total_steps: int, controller: Arx5OperationalSpaceController, data: mujoco.MjData) -> None:
    eef_pos = controller.get_ee_pose(data).position
    print(
        f"[{step_id:04d}/{total_steps}] "
        f"eef=({eef_pos[0]:+.3f}, {eef_pos[1]:+.3f}, {eef_pos[2]:+.3f}) "
        f"q=({data.qpos[0]:+.2f}, {data.qpos[1]:+.2f}, {data.qpos[2]:+.2f}, "
        f"{data.qpos[3]:+.2f}, {data.qpos[4]:+.2f}, {data.qpos[5]:+.2f}) "
        f"grip=({data.qpos[6]:.3f}, {data.qpos[7]:.3f})",
        flush=True,
    )


def _summarize_eef_error(
    replay_eef: list[np.ndarray],
    recorded_eef: np.ndarray | None,
    *,
    label: str,
    metrics_path: Path | None,
) -> dict[str, object] | None:
    if recorded_eef is None:
        return None
    replay = np.asarray(replay_eef, dtype=np.float64)
    count = min(len(replay), len(recorded_eef))
    if count == 0:
        return None
    replay = replay[:count]
    reference = recorded_eef[:count]
    error = np.linalg.norm(replay - reference, axis=1)
    summary: dict[str, object] = {
        "label": label,
        "count": int(count),
        "mean_m": float(np.mean(error)),
        "max_m": float(np.max(error)),
        "final_m": float(error[-1]),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "first_replay_eef": replay[0].tolist(),
        "first_hdf5_eef": reference[0].tolist(),
        "final_replay_eef": replay[-1].tolist(),
        "final_hdf5_eef": reference[-1].tolist(),
    }
    print(
        f"[INFO] {label} eef position error vs hdf5: "
        f"count={count} mean={summary['mean_m']:.6f}m "
        f"max={summary['max_m']:.6f}m final={summary['final_m']:.6f}m "
        f"rmse={summary['rmse_m']:.6f}m",
        flush=True,
    )
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[INFO] wrote eef metrics: {metrics_path}", flush=True)
    return summary


def _camera_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id) or f"camera_{camera_id}"
        for camera_id in range(model.ncam)
    ]


class VideoRecorder:
    def __init__(self, model: mujoco.MjModel, camera_name: str, video_path: Path, width: int, height: int, fps: float):
        self.model = model
        self.camera_name = camera_name
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._renderer: mujoco.Renderer | None = None
        self._writer = None
        self._frame_count = 0

        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo model does not contain camera {camera_name!r}. Available: {_camera_names(model)}")

    def __enter__(self) -> "VideoRecorder":
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._writer = imageio.get_writer(self.video_path, fps=self.fps, codec="libx264", format="FFMPEG")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._renderer is not None:
            self._renderer.close()
        if exc_type is None:
            print(f"[INFO] saved {self._frame_count} {self.camera_name} frames to: {self.video_path}", flush=True)

    def capture(self, data: mujoco.MjData) -> None:
        if self._renderer is None or self._writer is None:
            return
        self._renderer.update_scene(data, camera=self.camera_name)
        self._writer.append_data(self._renderer.render())
        self._frame_count += 1


def _print_scene_alignment(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_forward(model, data)
    print(f"[INFO] cameras: {', '.join(_camera_names(model))}", flush=True)
    for body_name in ("arx5", "external_cam", "camera"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        pos = data.xpos[body_id]
        quat = data.xquat[body_id]
        print(
            f"[INFO] body {body_name}: pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) "
            f"quat=({quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f})",
            flush=True,
        )


def _run_replay(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Arx5OperationalSpaceController,
    actions: np.ndarray,
    *,
    decimation: int,
    real_time: bool,
    eef_trace: list[np.ndarray] | None = None,
    video_recorder: VideoRecorder | None = None,
    viewer=None,
) -> None:
    step_dt = float(model.opt.timestep) * decimation
    for action_id, action in enumerate(actions, start=1):
        start_time = time.time()
        controller.set_target_from_action(data, action)
        for _ in range(decimation):
            controller.apply_target(data)
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    return
        if video_recorder is not None:
            video_recorder.capture(data)
        if eef_trace is not None:
            eef_trace.append(controller.get_ee_pose(data).position.copy())
        if action_id == 1 or action_id == len(actions) or action_id % 5 == 0:
            _print_step(action_id, len(actions), controller, data)
        if real_time:
            sleep_time = step_dt - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def main() -> None:
    args = _parse_args()
    actions = _read_actions(args.dataset)
    recorded_initial_eef = _read_recorded_initial_eef(args.dataset) if args.align_initial_eef else None
    recorded_eef = _read_recorded_eef_trajectory(args.dataset)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    controller = Arx5OperationalSpaceController(model, mode=args.mode)

    print(f"[INFO] model: {args.model}")
    print(f"[INFO] dataset: {args.dataset}")
    print(f"[INFO] actions: {actions.shape[0]} x {actions.shape[1]}")
    print(f"[INFO] dt={model.opt.timestep:.6f}, decimation={args.decimation}, control_dt={model.opt.timestep * args.decimation:.3f}s")
    print(f"[INFO] UWLab recorded control_dt={control_period_seconds():.3f}s")
    video_path = args.video_path if args.video_path is not None else _default_video_path(args.dataset)
    video_fps = 1.0 / (float(model.opt.timestep) * args.decimation)

    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                _reset_to_keyframe(model, data, args.keyframe)
                _align_robot_root_to_recorded_eef(model, data, controller, recorded_initial_eef)
                controller.reset(data)
                _print_scene_alignment(model, data)
                _print_step(0, len(actions), controller, data)
                eef_trace = [controller.get_ee_pose(data).position.copy()]
                if args.no_video:
                    _run_replay(
                        model,
                        data,
                        controller,
                        actions,
                        decimation=args.decimation,
                        real_time=args.real_time,
                        eef_trace=eef_trace,
                        viewer=viewer,
                    )
                else:
                    with VideoRecorder(
                        model, args.camera, video_path, args.video_width, args.video_height, video_fps
                    ) as recorder:
                        recorder.capture(data)
                        _run_replay(
                            model,
                            data,
                            controller,
                            actions,
                            decimation=args.decimation,
                            real_time=args.real_time,
                            eef_trace=eef_trace,
                            video_recorder=recorder,
                            viewer=viewer,
                        )
                _summarize_eef_error(eef_trace, recorded_eef, label="mujoco", metrics_path=args.metrics_path)
                if not args.loop:
                    break
    else:
        _reset_to_keyframe(model, data, args.keyframe)
        _align_robot_root_to_recorded_eef(model, data, controller, recorded_initial_eef)
        controller.reset(data)
        _print_scene_alignment(model, data)
        _print_step(0, len(actions), controller, data)
        eef_trace = [controller.get_ee_pose(data).position.copy()]
        if args.no_video:
            _run_replay(
                model,
                data,
                controller,
                actions,
                decimation=args.decimation,
                real_time=False,
                eef_trace=eef_trace,
            )
        else:
            with VideoRecorder(model, args.camera, video_path, args.video_width, args.video_height, video_fps) as recorder:
                recorder.capture(data)
                _run_replay(
                    model,
                    data,
                    controller,
                    actions,
                    decimation=args.decimation,
                    real_time=False,
                    eef_trace=eef_trace,
                    video_recorder=recorder,
                )
        _summarize_eef_error(eef_trace, recorded_eef, label="mujoco", metrics_path=args.metrics_path)


if __name__ == "__main__":
    main()
