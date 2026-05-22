# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg)


def bbox_to_confidence_mask(
    bboxes: torch.Tensor,
    image_size: int | tuple[int, int] = 128,
) -> torch.Tensor:
    """Convert padded xyxy+confidence boxes to confidence masks.

    Args:
        bboxes: Tensor shaped ``(B, V, N, 5)`` with ``x1, y1, x2, y2, confidence``.
            Padding boxes use negative coordinates and contribute zero.
        image_size: Output mask size. An integer means square ``H=W=image_size``.

    Returns:
        Tensor shaped ``(B, V, 1, H, W)``. Overlapping boxes use max confidence.
    """

    if bboxes.ndim != 4 or bboxes.shape[-1] != 5:
        raise ValueError(f"Expected bboxes shaped (B, V, N, 5), got {tuple(bboxes.shape)}.")
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size

    bboxes = bboxes.to(dtype=torch.float32)
    coords = bboxes[..., :4]
    confidence = bboxes[..., 4].clamp(min=0.0)
    padding = coords[..., 0] < 0
    confidence = confidence.masked_fill(padding, 0.0)

    x1 = coords[..., 0].clamp(0, width)[..., None, None]
    y1 = coords[..., 1].clamp(0, height)[..., None, None]
    x2 = coords[..., 2].clamp(0, width)[..., None, None]
    y2 = coords[..., 3].clamp(0, height)[..., None, None]

    y_range = torch.arange(height, device=bboxes.device, dtype=torch.float32)
    x_range = torch.arange(width, device=bboxes.device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
    x_grid = x_grid.view(1, 1, 1, height, width)
    y_grid = y_grid.view(1, 1, 1, height, width)

    inside = (x_grid >= x1) & (x_grid < x2) & (y_grid >= y1) & (y_grid < y2)
    mask_per_box = torch.where(inside, confidence[..., None, None], torch.zeros((), device=bboxes.device))
    mask = mask_per_box.amax(dim=2)
    return mask.unsqueeze(2)


class _PatchMapEncoder(nn.Module):
    """Small patch encoder used for depth maps and bbox confidence masks."""

    def __init__(
        self,
        *,
        input_channels: int = 1,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 256,
        conv_channels: int = 64,
        no_patch_embed_bias: bool = False,
    ) -> None:
        super().__init__()
        if image_size % (2 * patch_size) != 0:
            raise ValueError(
                f"image_size must be divisible by 2 * patch_size, got image_size={image_size}, patch_size={patch_size}."
            )
        self.input_channels = input_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_grid_size = image_size // patch_size // 2
        self.num_patches = self.patch_grid_size * self.patch_grid_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            conv_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=not no_patch_embed_bias,
        )
        self.bn = nn.BatchNorm2d(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images shaped (B, C, H, W), got {tuple(images.shape)}.")
        if images.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {images.shape[1]}.")
        x = images.to(dtype=torch.float32)
        x = self.conv(x)
        x = self.proj(x)
        return self.bn(x)


class DepthMapEncoder(_PatchMapEncoder):
    """DepthAnything map encoder.

    Default input/output path:
        ``(B, 1, 128, 128) -> (B, 256, 8, 8)``.
    """


class MaskPatchEncoderWithoutRGB(_PatchMapEncoder):
    """BBox confidence-mask encoder.

    Default input/output path:
        ``(B, 1, 128, 128) -> (B, 256, 8, 8)``.
    """


class DepthMaskCrossAttentionFusion(nn.Module):
    """Fuse bbox mask tokens with depth tokens by mask-to-depth attention."""

    def __init__(
        self,
        *,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}.")
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(2 * embed_dim)
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, depth_tokens: torch.Tensor, mask_tokens: torch.Tensor) -> torch.Tensor:
        if depth_tokens.shape != mask_tokens.shape:
            raise ValueError(
                f"depth_tokens and mask_tokens must have the same shape, got "
                f"{tuple(depth_tokens.shape)} and {tuple(mask_tokens.shape)}."
            )
        depth = depth_tokens.to(dtype=torch.float32)
        mask = mask_tokens.to(dtype=torch.float32)
        depth_norm = self.kv_norm(depth)
        attn_out, _ = self.cross_attn(self.q_norm(mask), depth_norm, depth_norm, need_weights=False)
        mask = mask + self.attn_drop(attn_out)
        mask = mask + self.ffn(self.ffn_norm(mask))
        return self.out_proj(self.out_norm(torch.cat([depth, mask], dim=-1)))


class TrackPatchEmbed(nn.Module):
    """Embed visual trajectory points into track tokens."""

    def __init__(
        self,
        *,
        track_len: int = 16,
        num_track_ids: int = 32,
        patch_size: int = 16,
        input_dim: int = 4,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if track_len % patch_size != 0:
            raise ValueError(f"track_len must be divisible by patch_size, got {track_len} and {patch_size}.")
        self.track_len = track_len
        self.num_track_ids = num_track_ids
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_patches_per_track = track_len // patch_size
        self.num_patches = self.num_patches_per_track * num_track_ids
        self.conv = nn.Conv1d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        """Embed tracks shaped ``(B, track_len, num_track_ids, input_dim)``.

        Returns:
            Tensor shaped ``(B, num_patches_per_track, num_track_ids, embed_dim)``.
        """

        if tracks.ndim != 4:
            raise ValueError(f"Expected tracks shaped (B, L, N, C), got {tuple(tracks.shape)}.")
        batch_size, track_len, num_tracks, input_dim = tracks.shape
        if track_len != self.track_len or num_tracks != self.num_track_ids or input_dim != self.input_dim:
            raise ValueError(
                f"Expected tracks shaped (B, {self.track_len}, {self.num_track_ids}, {self.input_dim}), "
                f"got {tuple(tracks.shape)}."
            )
        x = tracks.to(dtype=torch.float32).permute(0, 2, 3, 1).reshape(batch_size * num_tracks, input_dim, track_len)
        x = self.conv(x)
        x = x.reshape(batch_size, num_tracks, self.embed_dim, self.num_patches_per_track)
        return x.permute(0, 3, 1, 2)


class SpatialTokenTransformer(nn.Module):
    """CLS-token self-attention encoder over visual and trajectory tokens."""

    def __init__(
        self,
        *,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}.")
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shaped (B, N, C), got {tuple(tokens.shape)}.")
        return self.out_norm(self.encoder(tokens))


@dataclass(frozen=True)
class SSIStyleModalEncoderShapes:
    num_visual_tokens: int
    num_track_tokens: int
    num_total_tokens: int
    embed_dim: int


class SSIStyleModalEncoder(nn.Module):
    """Encode depth, bbox, and trajectory modalities into one CLS feature."""

    def __init__(
        self,
        *,
        num_views: int = 2,
        image_size: int = 128,
        max_bbox_num: int = 3,
        track_len: int = 16,
        num_track_ids: int = 32,
        track_patch_size: int = 16,
        embed_dim: int = 256,
        depth_encoder: dict[str, Any] | None = None,
        mask_encoder: dict[str, Any] | None = None,
        track_encoder: dict[str, Any] | None = None,
        cross_attention: dict[str, Any] | None = None,
        spatial_transformer: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.image_size = image_size
        self.max_bbox_num = max_bbox_num
        self.track_len = track_len
        self.num_track_ids = num_track_ids
        self.track_patch_size = track_patch_size
        self.embed_dim = embed_dim

        depth_cfg = {"image_size": image_size, "embed_dim": embed_dim, **_as_plain_dict(depth_encoder)}
        mask_cfg = {"image_size": image_size, "embed_dim": embed_dim, **_as_plain_dict(mask_encoder)}
        track_cfg = {
            "track_len": track_len,
            "num_track_ids": num_track_ids,
            "patch_size": track_patch_size,
            "input_dim": 2 + num_views,
            "embed_dim": embed_dim,
            **_as_plain_dict(track_encoder),
        }
        cross_cfg = {"embed_dim": embed_dim, **_as_plain_dict(cross_attention)}
        spatial_cfg = {"embed_dim": embed_dim, **_as_plain_dict(spatial_transformer)}

        self.depth_encoder = DepthMapEncoder(**depth_cfg)
        self.mask_encoder = MaskPatchEncoderWithoutRGB(**mask_cfg)
        self.track_encoder = TrackPatchEmbed(**track_cfg)
        self.depth_mask_fusion = DepthMaskCrossAttentionFusion(**cross_cfg)
        self.spatial_transformer = SpatialTokenTransformer(**spatial_cfg)

        num_patches_per_view = self.depth_encoder.num_patches
        if self.mask_encoder.num_patches != num_patches_per_view:
            raise ValueError("Depth and mask encoders must produce the same number of patches.")
        self.num_visual_tokens = num_views * num_patches_per_view
        self.num_track_tokens = num_views * self.track_encoder.num_patches
        self.num_total_tokens = 1 + self.num_visual_tokens + self.num_track_tokens

        self.depth_pos_embed = nn.Parameter(torch.randn(1, self.num_visual_tokens, embed_dim) * 0.02)
        self.mask_pos_embed = nn.Parameter(torch.randn(1, self.num_visual_tokens, embed_dim) * 0.02)
        self.track_pos_embed = nn.Parameter(torch.randn(1, self.num_track_tokens, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.visual_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.track_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    @property
    def shapes(self) -> SSIStyleModalEncoderShapes:
        return SSIStyleModalEncoderShapes(
            num_visual_tokens=self.num_visual_tokens,
            num_track_tokens=self.num_track_tokens,
            num_total_tokens=self.num_total_tokens,
            embed_dim=self.embed_dim,
        )

    def forward(self, depth_map: torch.Tensor, bboxes: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        depth_tokens = self.encode_depth(depth_map)
        mask_tokens = self.encode_mask(bboxes)
        visual_tokens = self.depth_mask_fusion(depth_tokens, mask_tokens) + self.visual_type_embed
        track_tokens = self.encode_trajectory(trajectory) + self.track_type_embed
        tokens = torch.cat([visual_tokens, track_tokens], dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.spatial_transformer(torch.cat([cls, tokens], dim=1))
        return encoded[:, 0]

    def encode_depth(self, depth_map: torch.Tensor) -> torch.Tensor:
        if depth_map.ndim != 5:
            raise ValueError(f"Expected depth_map shaped (B, V, 1, H, W), got {tuple(depth_map.shape)}.")
        batch_size, num_views, channels, height, width = depth_map.shape
        if num_views != self.num_views or channels != 1 or height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected depth_map shaped (B, {self.num_views}, 1, {self.image_size}, {self.image_size}), "
                f"got {tuple(depth_map.shape)}."
            )
        x = depth_map.reshape(batch_size * num_views, channels, height, width)
        x = self.depth_encoder(x)
        x = x.flatten(2).transpose(1, 2)
        return x.reshape(batch_size, self.num_visual_tokens, self.embed_dim) + self.depth_pos_embed

    def encode_mask(self, bboxes: torch.Tensor) -> torch.Tensor:
        if bboxes.ndim != 4:
            raise ValueError(f"Expected bboxes shaped (B, V, N, 5), got {tuple(bboxes.shape)}.")
        batch_size, num_views, max_bbox_num, box_dim = bboxes.shape
        if num_views != self.num_views or max_bbox_num != self.max_bbox_num or box_dim != 5:
            raise ValueError(
                f"Expected bboxes shaped (B, {self.num_views}, {self.max_bbox_num}, 5), got {tuple(bboxes.shape)}."
            )
        masks = bbox_to_confidence_mask(bboxes, self.image_size)
        x = masks.reshape(batch_size * num_views, 1, self.image_size, self.image_size)
        x = self.mask_encoder(x)
        x = x.flatten(2).transpose(1, 2)
        return x.reshape(batch_size, self.num_visual_tokens, self.embed_dim) + self.mask_pos_embed

    def encode_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 5:
            raise ValueError(f"Expected trajectory shaped (B, V, L, N, 2), got {tuple(trajectory.shape)}.")
        batch_size, num_views, track_len, num_tracks, xy_dim = trajectory.shape
        if (
            num_views != self.num_views
            or track_len != self.track_len
            or num_tracks != self.num_track_ids
            or xy_dim != 2
        ):
            raise ValueError(
                f"Expected trajectory shaped (B, {self.num_views}, {self.track_len}, {self.num_track_ids}, 2), "
                f"got {tuple(trajectory.shape)}."
            )
        one_hot = torch.eye(num_views, device=trajectory.device, dtype=trajectory.dtype)
        one_hot = one_hot.view(1, num_views, 1, 1, num_views).expand(batch_size, -1, track_len, num_tracks, -1)
        tracks = torch.cat([trajectory.to(dtype=torch.float32), one_hot.to(dtype=torch.float32)], dim=-1)
        tracks = tracks.reshape(batch_size * num_views, track_len, num_tracks, 2 + num_views)
        x = self.track_encoder(tracks)
        x = x.reshape(batch_size, num_views * self.track_encoder.num_patches, self.embed_dim)
        return x + self.track_pos_embed
