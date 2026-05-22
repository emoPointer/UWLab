from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mujoco_mirror import MuJoCoRenderMirror
from .renderer import MuJoCoOffscreenRenderer, VideoRecorder
from .state_extractors import IsaacLabStateExtractor


@dataclass(frozen=True)
class IsaacMuJoCoRenderConfig:
    """Configuration for mirroring Isaac state into a MuJoCo renderer."""

    mujoco_xml: str = "mujoco_arx5/models/arx5_robosuite_tabletop_dynamic.xml"
    mujoco_camera: str = "external_camera"
    mujoco_video_path: str = "videos/mujoco_isaac/render.mp4"
    mujoco_video_width: int = 640
    mujoco_video_height: int = 480
    mujoco_insertive_body: str = "insertive_cube"
    mujoco_receptive_body: str = "receptive_cube"
    randomize_light_angles: bool = False
    mujoco_light_yaw_range_deg: tuple[float, float] = (0.0, 360.0)
    mujoco_light_elevation_range_deg: tuple[float, float] = (35.0, 75.0)
    mujoco_light_distance_range: tuple[float, float] = (2.0, 3.0)
    mujoco_light_target: tuple[float, float, float] = (-0.3, -0.2, 0.84)
    mujoco_light_seed: int | None = None
    sync_camera_poses: bool = False
    mujoco_camera_body_map: tuple[tuple[str, str], ...] = (
        ("external_camera", "external_cam"),
        ("wrist_camera", "camera"),
    )
    robot_joint_names: tuple[str, ...] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "joint8",
    )
    record_video: bool = True


class IsaacMuJoCoRenderSession:
    """Reusable Isaac-physics/MuJoCo-render session.

    Isaac owns physics. This session mirrors env state into MuJoCo and renders
    frames; it is suitable for replay, deploy, and policy evaluation loops.
    """

    def __init__(self, env, config: IsaacMuJoCoRenderConfig):
        self.env = env
        self.config = config
        self._rng = np.random.default_rng(config.mujoco_light_seed)
        camera_body_map = dict(config.mujoco_camera_body_map) if config.sync_camera_poses else {}
        self.extractor = IsaacLabStateExtractor(
            object_names=("insertive_object", "receptive_object"),
            camera_names=tuple(camera_body_map.keys()),
            joint_names=config.robot_joint_names,
            env_index=0,
        )
        self.mirror = MuJoCoRenderMirror.from_xml_path(
            config.mujoco_xml,
            object_body_map={
                "insertive_object": config.mujoco_insertive_body,
                "receptive_object": config.mujoco_receptive_body,
            },
            camera_body_map=camera_body_map,
        )
        self.renderer = MuJoCoOffscreenRenderer(
            self.mirror.model,
            width=config.mujoco_video_width,
            height=config.mujoco_video_height,
        )
        if config.randomize_light_angles:
            self.randomize_light_angles()

    def render_frame(self, *, step: int | None = None, camera: str | None = None):
        self.sync(step=step)
        return self.render_synced_frame(camera=camera)

    def sync(self, *, step: int | None = None) -> None:
        state = self.extractor.extract(self.env, step=step)
        self.mirror.apply(state)

    def render_synced_frame(self, *, camera: str | None = None):
        return self.renderer.render(self.mirror.data, camera=camera or self.config.mujoco_camera)

    def render_frames(self, cameras: tuple[str, ...] | list[str], *, step: int | None = None) -> dict[str, object]:
        self.sync(step=step)
        return {camera: self.render_synced_frame(camera=camera) for camera in cameras}

    def randomize_light_angles(self) -> None:
        self.mirror.randomize_light_angles(
            rng=self._rng,
            yaw_range_deg=self.config.mujoco_light_yaw_range_deg,
            elevation_range_deg=self.config.mujoco_light_elevation_range_deg,
            distance_range=self.config.mujoco_light_distance_range,
            target=self.config.mujoco_light_target,
        )

    def make_video_recorder(self, path: str | None = None, *, fps: float) -> VideoRecorder | None:
        if not self.config.record_video:
            return None
        return VideoRecorder(path or self.config.mujoco_video_path, fps=fps)

    def close(self) -> None:
        self.renderer.close()
