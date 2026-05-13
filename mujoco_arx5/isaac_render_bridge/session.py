from __future__ import annotations

from dataclasses import dataclass

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
        self.extractor = IsaacLabStateExtractor(
            object_names=("insertive_object", "receptive_object"),
            joint_names=config.robot_joint_names,
            env_index=0,
        )
        self.mirror = MuJoCoRenderMirror.from_xml_path(
            config.mujoco_xml,
            object_body_map={
                "insertive_object": config.mujoco_insertive_body,
                "receptive_object": config.mujoco_receptive_body,
            },
        )
        self.renderer = MuJoCoOffscreenRenderer(
            self.mirror.model,
            width=config.mujoco_video_width,
            height=config.mujoco_video_height,
        )

    def render_frame(self, *, step: int | None = None, camera: str | None = None):
        state = self.extractor.extract(self.env, step=step)
        self.mirror.apply(state)
        return self.renderer.render(self.mirror.data, camera=camera or self.config.mujoco_camera)

    def make_video_recorder(self, path: str | None = None, *, fps: float) -> VideoRecorder | None:
        if not self.config.record_video:
            return None
        return VideoRecorder(path or self.config.mujoco_video_path, fps=fps)

    def close(self) -> None:
        self.renderer.close()
