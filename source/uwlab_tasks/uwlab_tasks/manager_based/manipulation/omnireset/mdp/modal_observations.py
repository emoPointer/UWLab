# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera


class ModalVisionObservation(ManagerTermBase):
    """Online SSI-style modal observation term.

    The term lazily extracts all modal tensors once per Isaac step and shares the
    result across ``depth_map``, ``bboxes``, and ``trajectory`` observation terms.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            return
        for extractor in getattr(self._env, "_uwlab_modal_vision_extractors", {}).values():
            extractor.clear_cache()

    def __call__(
        self,
        env: ManagerBasedEnv,
        modal_key: str,
        external_camera_cfg: SceneEntityCfg = SceneEntityCfg("external_camera"),
        wrist_camera_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
        ssi_root: str = "/home/emopointer/SSI-SimToReal",
        ssi_config: str = (
            "/home/emopointer/SSI-SimToReal/results/policy/"
            "0417_UWLab_delat_OSC_control_1447_seed42/config.yaml"
        ),
        trajectory_config: str = "/home/emopointer/UWLab/logs/trajectory_predict/config.yaml",
        trajectory_ckpt: str = "/home/emopointer/UWLab/logs/trajectory_predict/model_final.ckpt",
        table_prompt: str = "robot, red cube, green cube",
        wrist_prompt: str = "red cube, green cube",
        task_description: str = "Put the red block on the green block.",
        process_size: int = 400,
        model_size: int = 128,
        max_bboxes: int = 3,
        track_len: int = 16,
        num_track_ids: int = 32,
        depth_encoder: str = "vitb",
        batch_size: int = 16,
        depth_chunk_size: int = 16,
        device: str | None = None,
        allow_bert_download: bool = False,
    ) -> torch.Tensor:
        if modal_key not in {"depth_map", "bboxes", "trajectory"}:
            raise ValueError(f"Unsupported modal_key={modal_key!r}.")

        params = {
            "external_camera_cfg": external_camera_cfg,
            "wrist_camera_cfg": wrist_camera_cfg,
            "ssi_root": ssi_root,
            "ssi_config": ssi_config,
            "trajectory_config": trajectory_config,
            "trajectory_ckpt": trajectory_ckpt,
            "table_prompt": table_prompt,
            "wrist_prompt": wrist_prompt,
            "task_description": task_description,
            "process_size": int(process_size),
            "model_size": int(model_size),
            "max_bboxes": int(max_bboxes),
            "track_len": int(track_len),
            "num_track_ids": int(num_track_ids),
            "depth_encoder": depth_encoder,
            "batch_size": int(batch_size),
            "depth_chunk_size": int(depth_chunk_size),
            "device": device,
            "allow_bert_download": bool(allow_bert_download),
        }
        if _is_observation_manager_initializing(env):
            return _dummy_modal_observation(env, modal_key, params)
        extractor = _get_or_create_modal_extractor(env, params)
        return extractor.compute(env)[modal_key]


class _OnlineModalVisionExtractor:
    def __init__(self, params: dict[str, Any], env_device: str):
        self.params = params
        self.env_device = torch.device(env_device)
        self.model_device = torch.device(params["device"] or env_device)
        self.models: dict[str, Any] | None = None
        self.cache_key: tuple[int, int, int] | None = None
        self.cache: dict[str, torch.Tensor] | None = None

    def clear_cache(self) -> None:
        self.cache_key = None
        self.cache = None

    def compute(self, env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
        cache_key = _env_cache_key(env)
        if self.cache_key == cache_key and self.cache is not None:
            return self.cache

        self._ensure_models()
        table_400 = _camera_rgb_400(
            env,
            self.params["external_camera_cfg"],
            process_size=self.params["process_size"],
            top_right_crop=True,
        )
        wrist_400 = _camera_rgb_400(
            env,
            self.params["wrist_camera_cfg"],
            process_size=self.params["process_size"],
            top_right_crop=False,
        )

        table_det = self._detect_batched(table_400, self.params["table_prompt"])
        wrist_det = self._detect_batched(wrist_400, self.params["wrist_prompt"])

        depth = self._compute_depth(table_400, wrist_400)
        bboxes = self._pack_bboxes(table_det, wrist_det)
        trajectory = self._compute_trajectory(table_400, wrist_400, bboxes)

        self.cache_key = cache_key
        self.cache = {
            "depth_map": torch.from_numpy(depth).to(device=self.env_device, dtype=torch.float32),
            "bboxes": torch.from_numpy(bboxes).to(device=self.env_device, dtype=torch.float32),
            "trajectory": torch.from_numpy(trajectory).to(device=self.env_device, dtype=torch.float32),
        }
        return self.cache

    def _ensure_models(self) -> None:
        if self.models is not None:
            return

        ssi_root = Path(self.params["ssi_root"]).resolve()
        ssi_config = Path(self.params["ssi_config"]).resolve()
        trajectory_config = Path(self.params["trajectory_config"]).resolve()
        trajectory_ckpt = Path(self.params["trajectory_ckpt"]).resolve()
        for path in (ssi_root, ssi_config, trajectory_config, trajectory_ckpt):
            if not path.exists():
                raise FileNotFoundError(f"Modal vision dependency does not exist: {path}")

        _setup_ssi_imports(ssi_root)
        old_cwd = os.getcwd()
        try:
            os.chdir(ssi_root)
            from atm.model.track_transformer import TrackTransformer
            from atm.utils.depth_utils import DepthPreprocessor, get_depth_emb_a800, init_depth_generator
            from atm.utils.ground_dino_utils import Grounded_SAM2
            _patch_grounding_dino_ms_deform_attn_fallback()

            ssi_cfg = OmegaConf.load(ssi_config)
            detector = Grounded_SAM2(ssi_cfg.mask_cfg, init_grounding_dino=True)
            detector_device = (
                next(detector.detector.parameters()).device
                if hasattr(detector, "detector")
                else self.model_device
            )

            depth_root = Path(ssi_cfg.depth_cfg.depth_generator_root)
            if not depth_root.is_absolute():
                depth_root = ssi_root / depth_root
            depth_generator = init_depth_generator(
                str(depth_root),
                encoder=self.params["depth_encoder"] or ssi_cfg.depth_cfg.encoder,
                return_feat_only=False,
            )
            depth_device = next(depth_generator.parameters()).device

            traj_cfg = OmegaConf.load(trajectory_config)
            track_model = TrackTransformer(**traj_cfg.model_cfg).to(self.model_device).eval()
            state = torch.load(trajectory_ckpt, map_location="cpu")
            track_model.load_state_dict(state)
            for param in track_model.parameters():
                param.requires_grad = False
        finally:
            os.chdir(old_cwd)

        task_emb = _load_task_embedding(
            self.params["task_description"],
            cache_dir=ssi_root / "data" / "bert_cache",
            device=self.model_device,
            allow_download=self.params["allow_bert_download"],
        )

        self.models = {
            "detector": detector,
            "detector_device": detector_device,
            "depth_generator": depth_generator,
            "depth_device": depth_device,
            "depth_preprocessor": DepthPreprocessor(),
            "get_depth_emb_a800": get_depth_emb_a800,
            "track_model": track_model,
            "task_emb": task_emb,
        }

    def _detect_batched(self, images: np.ndarray, prompt: str) -> list[dict[str, Any]]:
        assert self.models is not None
        results: list[dict[str, Any]] = []
        batch_size = max(int(self.params["batch_size"]), 1)
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(self.models["detector_device"])
            boxes, confidences, labels = self.models["detector"].predict_batch(
                tensor,
                text_prompts=[prompt] * len(batch),
                only_generate_bbox=True,
            )
            for box_i, conf_i, label_i in zip(boxes, confidences, labels):
                box_arr = np.asarray(box_i, dtype=np.float32).reshape(-1, 4)
                if isinstance(conf_i, torch.Tensor):
                    conf_arr = conf_i.detach().cpu().numpy().astype(np.float32).reshape(-1)
                else:
                    conf_arr = np.asarray(conf_i, dtype=np.float32).reshape(-1)
                if isinstance(label_i, Iterable) and not isinstance(label_i, str):
                    label_list = list(label_i)
                else:
                    label_list = [str(label_i)]
                if len(label_list) < len(box_arr):
                    label_list += [""] * (len(box_arr) - len(label_list))
                results.append({"boxes": box_arr, "confidences": conf_arr, "labels": label_list[: len(box_arr)]})
        return results

    @torch.no_grad()
    def _compute_depth(self, table_400: np.ndarray, wrist_400: np.ndarray) -> np.ndarray:
        assert self.models is not None
        images = np.stack([table_400, wrist_400], axis=1)
        tensor = torch.from_numpy(images).float().permute(0, 1, 4, 2, 3).unsqueeze(2)
        tensor = tensor.to(self.models["depth_device"])
        depth = self.models["get_depth_emb_a800"](
            self.models["depth_generator"],
            self.models["depth_preprocessor"],
            tensor,
            n_obs_steps=1,
            chunk_size=max(int(self.params["depth_chunk_size"]), 1),
            target_size=(self.params["model_size"], self.params["model_size"]),
        )
        return depth[:, :, 0].detach().cpu().numpy().astype(np.float32)

    def _pack_bboxes(self, table_det: list[dict[str, Any]], wrist_det: list[dict[str, Any]]) -> np.ndarray:
        bboxes = np.zeros((len(table_det), 2, self.params["max_bboxes"], 5), dtype=np.float32)
        for env_id, det in enumerate(table_det):
            bboxes[env_id, 0] = _select_and_scale_boxes(
                det["boxes"],
                det["confidences"],
                src_size=self.params["process_size"],
                dst_size=self.params["model_size"],
                max_bboxes=self.params["max_bboxes"],
            )
        for env_id, det in enumerate(wrist_det):
            bboxes[env_id, 1] = _select_and_scale_boxes(
                det["boxes"],
                det["confidences"],
                src_size=self.params["process_size"],
                dst_size=self.params["model_size"],
                max_bboxes=self.params["max_bboxes"],
            )
        return bboxes

    @torch.no_grad()
    def _compute_trajectory(self, table_400: np.ndarray, wrist_400: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        assert self.models is not None
        model_size = int(self.params["model_size"])
        num_track_ids = int(self.params["num_track_ids"])
        track_len = int(self.params["track_len"])

        table_128 = _resize_uint8_images(table_400, model_size)
        wrist_128 = _resize_uint8_images(wrist_400, model_size)
        view_images = np.stack([table_128, wrist_128], axis=1)

        points = np.zeros((len(table_400), 2, num_track_ids, 2), dtype=np.float32)
        for env_id in range(len(table_400)):
            points[env_id, 0] = _sample_points_in_boxes(bboxes[env_id, 0, :, :4], num_track_ids, model_size)
            points[env_id, 1] = _sample_points_in_boxes(bboxes[env_id, 1, :, :4], num_track_ids, model_size)

        flat_images = view_images.reshape(-1, model_size, model_size, 3)
        flat_points = points.reshape(-1, num_track_ids, 2)
        batch_size = max(int(self.params["batch_size"]), 1)
        flat_tracks: list[np.ndarray] = []
        for start in range(0, len(flat_images), batch_size):
            image_batch = flat_images[start : start + batch_size]
            points_batch = flat_points[start : start + batch_size]
            query = torch.from_numpy(points_batch / float(model_size)).float().to(self.model_device)
            query = query.unsqueeze(1).repeat(1, track_len, 1, 1)
            vid = torch.from_numpy(image_batch).float().permute(0, 3, 1, 2).unsqueeze(1).to(self.model_device)
            task_emb = self.models["task_emb"].repeat(len(image_batch), 1)
            pred, _ = self.models["track_model"].reconstruct(vid, query, task_emb, p_img=0)
            pred = pred[:, :track_len, :num_track_ids].detach().cpu().numpy().astype(np.float32)
            pred *= float(model_size)
            pred[:, 0] = points_batch
            flat_tracks.append(pred)

        tracks = np.concatenate(flat_tracks, axis=0)
        return tracks.reshape(len(table_400), 2, track_len, num_track_ids, 2)


def _get_or_create_modal_extractor(env: ManagerBasedEnv, params: dict[str, Any]) -> _OnlineModalVisionExtractor:
    registry = getattr(env, "_uwlab_modal_vision_extractors", None)
    if registry is None:
        registry = {}
        setattr(env, "_uwlab_modal_vision_extractors", registry)
    key = _extractor_key(params)
    if key not in registry:
        registry[key] = _OnlineModalVisionExtractor(params, env.device)
    return registry[key]


def _extractor_key(params: dict[str, Any]) -> tuple[Any, ...]:
    return (
        params["external_camera_cfg"].name,
        params["wrist_camera_cfg"].name,
        params["ssi_root"],
        params["ssi_config"],
        params["trajectory_config"],
        params["trajectory_ckpt"],
        params["table_prompt"],
        params["wrist_prompt"],
        params["task_description"],
        params["process_size"],
        params["model_size"],
        params["max_bboxes"],
        params["track_len"],
        params["num_track_ids"],
        params["depth_encoder"],
        params["batch_size"],
        params["depth_chunk_size"],
        params["device"],
        params["allow_bert_download"],
    )


def _env_cache_key(env: ManagerBasedEnv) -> tuple[int, int, int]:
    common_step = int(getattr(env, "common_step_counter", 0))
    sim_step = int(getattr(env, "_sim_step_counter", 0))
    episode_sum = int(getattr(env, "episode_length_buf", torch.zeros(1, device=env.device)).sum().item())
    return common_step, sim_step, episode_sum


def _is_observation_manager_initializing(env: ManagerBasedEnv) -> bool:
    return not hasattr(env, "observation_manager") or getattr(env, "observation_manager", None) is None


def _dummy_modal_observation(env: ManagerBasedEnv, modal_key: str, params: dict[str, Any]) -> torch.Tensor:
    num_envs = int(env.num_envs)
    device = torch.device(env.device)
    model_size = int(params["model_size"])
    if modal_key == "depth_map":
        return torch.zeros((num_envs, 2, 1, model_size, model_size), device=device, dtype=torch.float32)
    if modal_key == "bboxes":
        return torch.zeros((num_envs, 2, int(params["max_bboxes"]), 5), device=device, dtype=torch.float32)
    if modal_key == "trajectory":
        return torch.zeros(
            (num_envs, 2, int(params["track_len"]), int(params["num_track_ids"]), 2),
            device=device,
            dtype=torch.float32,
        )
    raise ValueError(f"Unsupported modal_key={modal_key!r}.")


def _setup_ssi_imports(ssi_root: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    for path in (ssi_root, ssi_root / "Grounded_SAM_2", ssi_root / "Depth_Anything_V2"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # TrackTransformer only needs atm.policy.vilt_modules.language_modules.
    # Avoid executing atm.policy.__init__, which imports diffusion policy deps
    # that are not needed for online trajectory prediction in Isaac.
    policy_root = ssi_root / "atm" / "policy"
    vilt_root = policy_root / "vilt_modules"
    if "atm.policy" not in sys.modules:
        policy_module = types.ModuleType("atm.policy")
        policy_module.__path__ = [str(policy_root)]
        sys.modules["atm.policy"] = policy_module
    if "atm.policy.vilt_modules" not in sys.modules:
        vilt_module = types.ModuleType("atm.policy.vilt_modules")
        vilt_module.__path__ = [str(vilt_root)]
        sys.modules["atm.policy.vilt_modules"] = vilt_module


def _patch_grounding_dino_ms_deform_attn_fallback() -> None:
    try:
        from grounding_dino.groundingdino.models.GroundingDINO import ms_deform_attn
    except Exception:
        return
    if hasattr(ms_deform_attn, "_C"):
        return
    if getattr(ms_deform_attn, "_uwlab_pytorch_fallback", False):
        return

    class _PytorchMultiScaleDeformableAttnFunction:
        @staticmethod
        def apply(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        ):
            return ms_deform_attn.multi_scale_deformable_attn_pytorch(
                value,
                value_spatial_shapes,
                sampling_locations,
                attention_weights,
            )

    ms_deform_attn.MultiScaleDeformableAttnFunction = _PytorchMultiScaleDeformableAttnFunction
    ms_deform_attn._uwlab_pytorch_fallback = True
    print("[UWLab] GroundingDINO _C extension is unavailable; using PyTorch ms_deform_attn fallback.")


def _load_task_embedding(task_description: str, cache_dir: Path, device: torch.device, allow_download: bool) -> torch.Tensor:
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


def _camera_rgb_400(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    process_size: int,
    top_right_crop: bool,
) -> np.ndarray:
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output["rgb"]
    if images.ndim != 4:
        raise ValueError(f"Expected camera tensor shaped (N,H,W,C), got {tuple(images.shape)}.")
    if images.shape[-1] < 3:
        raise ValueError(f"Expected RGB camera output with at least 3 channels, got {images.shape[-1]}.")
    images = images[..., :3]

    height, width = int(images.shape[-3]), int(images.shape[-2])
    if top_right_crop:
        if height < process_size or width < process_size:
            raise ValueError(
                f"Camera {sensor_cfg.name!r} image ({height}, {width}) is smaller than crop {process_size}."
            )
        images = images[:, :process_size, width - process_size : width, :]
        return images.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()

    nchw = images.permute(0, 3, 1, 2).to(dtype=torch.float32)
    if (height, width) != (process_size, process_size):
        nchw = F.interpolate(nchw, size=(process_size, process_size), mode="bilinear", antialias=True)
    nhwc = nchw.permute(0, 2, 3, 1)
    return nhwc.detach().cpu().round().clamp(0, 255).to(torch.uint8).numpy()


def _resize_uint8_images(images: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(images).float().permute(0, 3, 1, 2)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", antialias=True)
    return tensor.permute(0, 2, 3, 1).round().clamp(0, 255).byte().numpy()


def _select_and_scale_boxes(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    src_size: int,
    dst_size: int,
    max_bboxes: int,
) -> np.ndarray:
    packed = np.zeros((max_bboxes, 5), dtype=np.float32)
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return packed
    conf = conf[: len(boxes)]
    order = np.argsort(-conf)[:max_bboxes]
    selected = boxes[order].copy()
    selected[:, 0::2] = np.clip(selected[:, 0::2], 0, src_size)
    selected[:, 1::2] = np.clip(selected[:, 1::2], 0, src_size)
    selected *= float(dst_size) / float(src_size)
    packed[: len(order), :4] = selected
    packed[: len(order), 4] = np.clip(conf[order], 0.0, 1.0)
    return packed


def _sample_points_in_boxes(boxes_xyxy: np.ndarray, num_points: int, image_size: int) -> np.ndarray:
    valid = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.astype(np.float32)
        if x2 > x1 + 1 and y2 > y1 + 1:
            valid.append((x1, y1, x2, y2))

    if not valid:
        grid_side = int(np.ceil(np.sqrt(num_points)))
        grid = np.linspace(0.2, 0.8, grid_side, dtype=np.float32)
        points = np.array([(x * image_size, y * image_size) for y in grid for x in grid], dtype=np.float32)
        return points[:num_points]

    points = []
    for point_idx in range(num_points):
        x1, y1, x2, y2 = valid[point_idx % len(valid)]
        local = point_idx // len(valid)
        gx = (local % 4 + 0.5) / 4.0
        gy = ((local // 4) % 4 + 0.5) / 4.0
        points.append((x1 + gx * (x2 - x1), y1 + gy * (y2 - y1)))
    return np.asarray(points, dtype=np.float32)
