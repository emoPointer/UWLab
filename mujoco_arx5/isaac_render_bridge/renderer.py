from __future__ import annotations

from pathlib import Path


class MuJoCoOffscreenRenderer:
    """Small wrapper around ``mujoco.Renderer`` for camera RGB frames."""

    def __init__(self, model, *, width: int = 640, height: int = 480, hidden_geom_groups: tuple[int, ...] = (2,)):
        import mujoco

        self._scene_option = mujoco.MjvOption()
        for group in hidden_geom_groups:
            if 0 <= group < len(self._scene_option.geomgroup):
                self._scene_option.geomgroup[group] = 0
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data, *, camera: str = "external_camera"):
        self._renderer.update_scene(data, camera=camera, scene_option=self._scene_option)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


class VideoRecorder:
    """Lazy imageio writer so importing the bridge does not require imageio."""

    def __init__(self, path: str | Path, *, fps: float):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.path, fps=fps, codec="libx264", format="FFMPEG")

    def append(self, frame) -> None:
        self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()
