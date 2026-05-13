from __future__ import annotations

import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .session import IsaacMuJoCoRenderConfig, IsaacMuJoCoRenderSession


@dataclass(frozen=True)
class Hdf5ActionReplayConfig(IsaacMuJoCoRenderConfig):
    dataset: str = ""
    mujoco_video_path: str = "videos/mujoco_isaac_replays/cube_000000.mp4"
    metrics_path: str | None = None
    ee_body_name: str = "link6"
    env_spacing: float = 3.0
    real_time: bool = False


def configure_cube_replay_reset(env_cfg) -> None:
    """Replace the task reset with the fixed cube-stack replay reset."""

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from uwlab_tasks.manager_based.manipulation.omnireset import mdp as task_mdp

    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        return
    env_cfg.events.reset_from_reset_states = EventTerm(
        func=task_mdp.FixedRobotWorkspaceTaskPairReset,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "table_cfg": SceneEntityCfg("table"),
            "robot_pose": (-0.535, -0.21, 0.8, 1.0, 0.0, 0.0, 0.0),
            "table_pose": (0.0, 0.0, 0.799375, 1.0, 0.0, 0.0, 0.0),
            "insertive_object_pose": (-0.30, -0.20, 0.87, 1.0, 0.0, 0.0, 0.0),
            "receptive_object_pose": (-0.30, -0.20, 0.84, 1.0, 0.0, 0.0, 0.0),
            "robot_xy_jitter_m": 0.0,
            "workspace_x_range": (-0.4, -0.2),
            "workspace_y_range": (-0.3, -0.1),
            "insertive_workspace_x_range": (-0.4, -0.2),
            "insertive_workspace_y_range": (-0.3, -0.1),
            "insertive_object_color": (0.0, 1.0, 0.0),
            "receptive_object_color": (1.0, 0.0, 0.0),
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "log_every_reset": False,
            "sync_visuals": True,
        },
    )
    if hasattr(env_cfg.events, "align_deploy_scene_to_robosuite_table"):
        align_params = env_cfg.events.align_deploy_scene_to_robosuite_table.params
        align_params["robot_xy_jitter_m"] = 0.0
        align_params["task_object_color_range"] = None
        align_params["insertive_object_color"] = (0.0, 1.0, 0.0)
        align_params["receptive_object_color"] = (1.0, 0.0, 0.0)


class Hdf5ActionReplayRunner:
    """Replay recorded actions in Isaac and render mirrored state in MuJoCo."""

    def __init__(self, env, config: Hdf5ActionReplayConfig, *, simulation_app=None):
        self.env = env
        self.config = config
        self.simulation_app = simulation_app
        self.ee_body_id = self._find_single_body_id(env.unwrapped.scene["robot"], config.ee_body_name)
        self.render_session = IsaacMuJoCoRenderSession(env, config)

    def close(self) -> None:
        self.render_session.close()

    def run_all(self) -> list[dict[str, object]]:
        dataset_files = self._resolve_dataset_files(self.config.dataset)
        print(f"[INFO] Found {len(dataset_files)} rollout file(s) to replay.")
        summaries = []
        for dataset_file in dataset_files:
            summaries.append(self.run_one(dataset_file, multiple_files=len(dataset_files) > 1))
        return summaries

    def run_one(self, dataset_file: Path, *, multiple_files: bool = False) -> dict[str, object]:
        rollout = self._read_rollout(dataset_file)
        actions = torch.tensor(rollout["actions"], dtype=torch.float32, device=self.env.unwrapped.device)
        print(f"[INFO] Loaded rollout: {dataset_file} actions={actions.shape[0]} x {actions.shape[1]}")
        source_env_origin = self._reset_to_recorded_initial_state(rollout)
        step_dt = float(self.env.unwrapped.step_dt)
        print(
            f"[INFO] Replaying with Isaac physics at {1.0 / step_dt:.3f} Hz; "
            f"rendering MuJoCo camera={self.config.mujoco_camera}"
        )

        recorder = self._make_recorder(dataset_file, multiple_files, fps=1.0 / step_dt)
        eef_trace = [self._eef_local()]
        if recorder is not None:
            recorder.append(self._render_frame(step=0))

        for step_id, action in enumerate(actions):
            if self.simulation_app is not None and not self.simulation_app.is_running():
                break
            start_time = time.time()
            self.env.step(action.unsqueeze(0))
            eef_trace.append(self._eef_local())
            if recorder is not None:
                recorder.append(self._render_frame(step=step_id + 1))
            if step_id % 10 == 0:
                print(f"[INFO] replay step {step_id + 1}/{actions.shape[0]}", flush=True)
            if self.config.real_time:
                sleep_time = step_dt - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        if recorder is not None:
            recorder.close()
            print(f"[INFO] saved MuJoCo-rendered video: {recorder.path}", flush=True)

        summary = self._summarize_eef_error(
            eef_trace,
            self._recorded_eef_local(rollout, source_env_origin),
        )
        metrics_path = self._metrics_output_path(dataset_file, multiple_files)
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[INFO] wrote eef metrics: {metrics_path}", flush=True)
        return summary

    def _render_frame(self, *, step: int):
        return self.render_session.render_frame(step=step)

    def _make_recorder(self, dataset_file: Path, multiple_files: bool, *, fps: float):
        return self.render_session.make_video_recorder(str(self._video_output_path(dataset_file, multiple_files)), fps=fps)

    def _reset_to_recorded_initial_state(self, rollout: dict[str, np.ndarray]) -> np.ndarray:
        unwrapped_env = self.env.unwrapped
        self.env.reset()
        source_env_origin = self._infer_source_env_origin(rollout)
        print(f"[INFO] Replay source env origin: {[round(float(v), 4) for v in source_env_origin]}", flush=True)
        self._write_object_pose(
            "insertive_object",
            rollout["insertive_cube_pos"][0],
            rollout["insertive_cube_quat"][0],
            source_env_origin,
        )
        self._write_object_pose(
            "receptive_object",
            rollout["receptive_cube_pos"][0],
            rollout["receptive_cube_quat"][0],
            source_env_origin,
        )
        unwrapped_env.scene.write_data_to_sim()
        unwrapped_env.sim.forward()
        return source_env_origin

    def _write_object_pose(self, asset_name: str, pos: np.ndarray, quat: np.ndarray, source_env_origin: np.ndarray):
        unwrapped_env = self.env.unwrapped
        asset = unwrapped_env.scene[asset_name]
        device = unwrapped_env.device
        env_ids = torch.tensor([0], dtype=torch.long, device=device)
        target_pos = pos - source_env_origin + unwrapped_env.scene.env_origins[0].detach().cpu().numpy()
        pose = torch.tensor(np.concatenate((target_pos, quat), axis=0), dtype=torch.float32, device=device).unsqueeze(0)
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device), env_ids=env_ids)

    def _eef_local(self) -> np.ndarray:
        robot = self.env.unwrapped.scene["robot"]
        eef_world = robot.data.body_link_pos_w[0, self.ee_body_id]
        eef_local = eef_world - self.env.unwrapped.scene.env_origins[0]
        return eef_local.detach().cpu().numpy().astype(np.float64, copy=False)

    def _infer_source_env_origin(self, rollout: dict[str, np.ndarray]) -> np.ndarray:
        if "source_env_origin" in rollout:
            return rollout["source_env_origin"]
        receptive_xy = rollout["receptive_cube_pos"][0, :2]
        origin_x = round(float((receptive_xy[0] + 0.3) / self.config.env_spacing)) * self.config.env_spacing
        origin_y = round(float((receptive_xy[1] + 0.2) / self.config.env_spacing)) * self.config.env_spacing
        return np.asarray([origin_x, origin_y, 0.0], dtype=np.float32)

    @staticmethod
    def _recorded_eef_local(rollout: dict[str, np.ndarray], source_env_origin: np.ndarray) -> np.ndarray | None:
        if "eef_pos" not in rollout:
            return None
        return rollout["eef_pos"] - source_env_origin.reshape(1, 3)

    @staticmethod
    def _summarize_eef_error(replay_eef: list[np.ndarray], recorded_eef: np.ndarray | None) -> dict[str, object]:
        if recorded_eef is None:
            return {"label": "isaac_physics_mujoco_render", "count": 0}
        replay = np.asarray(replay_eef, dtype=np.float64)
        reference = np.asarray(recorded_eef, dtype=np.float64)
        count = min(len(replay), len(reference))
        replay = replay[:count]
        reference = reference[:count]
        error = np.linalg.norm(replay - reference, axis=1)
        summary = {
            "label": "isaac_physics_mujoco_render",
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
            "[INFO] isaac-physics/mujoco-render eef position error vs hdf5: "
            f"count={count} mean={summary['mean_m']:.6f}m "
            f"max={summary['max_m']:.6f}m final={summary['final_m']:.6f}m "
            f"rmse={summary['rmse_m']:.6f}m",
            flush=True,
        )
        return summary

    @staticmethod
    def _read_rollout(dataset_file: Path) -> dict[str, np.ndarray]:
        with h5py.File(dataset_file, "r") as h5_file:
            required_keys = [
                "actions",
                "obs/insertive_cube_pos",
                "obs/insertive_cube_quat",
                "obs/receptive_cube_pos",
                "obs/receptive_cube_quat",
            ]
            missing = [key for key in required_keys if key not in h5_file]
            if missing:
                raise KeyError(f"Missing required dataset keys in {dataset_file}: {missing}")
            rollout = {
                "actions": h5_file["actions"][:].astype(np.float32, copy=False),
                "insertive_cube_pos": h5_file["obs/insertive_cube_pos"][:].astype(np.float32, copy=False),
                "insertive_cube_quat": h5_file["obs/insertive_cube_quat"][:].astype(np.float32, copy=False),
                "receptive_cube_pos": h5_file["obs/receptive_cube_pos"][:].astype(np.float32, copy=False),
                "receptive_cube_quat": h5_file["obs/receptive_cube_quat"][:].astype(np.float32, copy=False),
            }
            if "obs/eef_pos" in h5_file:
                rollout["eef_pos"] = h5_file["obs/eef_pos"][:].astype(np.float32, copy=False)
            if "source_env_origin" in h5_file.attrs:
                rollout["source_env_origin"] = np.asarray(h5_file.attrs["source_env_origin"], dtype=np.float32)
            return rollout

    @staticmethod
    def _resolve_dataset_files(dataset_arg: str) -> list[Path]:
        dataset_path = Path(dataset_arg).expanduser()
        if any(char in dataset_arg for char in "*?[]"):
            paths = [Path(path) for path in glob.glob(str(dataset_path))]
        elif dataset_path.is_dir():
            paths = sorted(dataset_path.glob("*.hdf5")) + sorted(dataset_path.glob("*.h5"))
        else:
            paths = [dataset_path]
        files = sorted(path for path in paths if path.is_file())
        if not files:
            raise FileNotFoundError(f"No HDF5 files found from dataset={dataset_arg!r}")
        return files

    def _video_output_path(self, dataset_file: Path, multiple_files: bool) -> Path:
        video_path = Path(self.config.mujoco_video_path).expanduser()
        if multiple_files or video_path.suffix.lower() != ".mp4":
            return video_path / f"{dataset_file.stem}_isaac_physics_mujoco_render.mp4"
        return video_path

    def _metrics_output_path(self, dataset_file: Path, multiple_files: bool) -> Path | None:
        if self.config.metrics_path is None:
            return None
        metrics_path = Path(self.config.metrics_path).expanduser()
        if multiple_files or metrics_path.suffix.lower() != ".json":
            return metrics_path / f"{dataset_file.stem}_isaac_physics_mujoco_render_metrics.json"
        return metrics_path

    @staticmethod
    def _find_single_body_id(robot, body_name: str) -> int:
        body_ids, body_names = robot.find_bodies(body_name)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one body matching {body_name!r}, found {body_names}.")
        return int(body_ids[0])
