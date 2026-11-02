# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unlimited-OCR v2 vision tower (SAM + CLIP + projector) on MLX."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np
import torch

from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_mlx import (
    load_checkpoint,
)

logger = logging.getLogger(__name__)

Features = Union[torch.Tensor, mx.array]


class _MlxVisionTower:
    """SAM + CLIP + projector on MLX; pages -> [num_image_tokens, n_embed]."""

    # Max views per SAM pass: 8 for 1024 px (~0.4 GB bias each), 16 for <= 640 px tiles.
    MAX_BATCH = 8
    MAX_BATCH_SMALL = 16

    def __init__(
        self,
        model_dir: str,
        dtype: mx.Dtype = mx.bfloat16,
        warmup: bool = True,
    ):
        from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_clip import (
            load_clip_and_projector,
        )
        from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_sam import (
            load_sam_encoder,
        )

        self.dtype = dtype
        self.sam = load_sam_encoder(model_dir, dtype)
        self.clip, self.projector = load_clip_and_projector(model_dir, dtype)
        extra = load_checkpoint(
            model_dir, ("model.image_newline", "model.view_seperator")
        )
        self.image_newline = extra["model.image_newline"].astype(dtype)
        self.view_seperator = extra["model.view_seperator"].astype(dtype)
        mx.eval(self.image_newline, self.view_seperator)
        # Build position tables before mx.compile traces _encode_tiles.
        self._precompute_position_tables()
        self._encode = mx.compile(self._encode_tiles)
        if warmup:
            self.warmup()

    def _precompute_position_tables(self) -> None:
        tables = []
        for img_size in (640, 1024):
            grid = img_size // 16
            tables.append(self.sam.abs_pos(grid))
            for blk in self.sam.blocks:
                size = grid if blk.window_size == 0 else blk.window_size
                tables.extend(blk.attn.rel_tables(size, size))
            clip_tokens = (img_size // 64) ** 2 + 1
            tables.append(self.clip.embeddings.positions(clip_tokens))
        mx.eval(*tables)

    def warmup(
        self,
        shapes: tuple[tuple[int, int, int], ...] = (
            (1, 1024, 1024),
            (6, 640, 640),
            (12, 640, 640),
        ),
    ) -> None:
        """Pre-compile common (batch, H, W) tile shapes at load time."""
        for b, h, w in shapes:
            dummy = mx.zeros((b, h, w, 3), dtype=self.dtype)
            mx.eval(self._encode(dummy))
        mx.clear_cache()

    def _max_batch_for_shape(self, shape_hwc: tuple[int, ...]) -> int:
        h, w = shape_hwc[0], shape_hwc[1]
        return self.MAX_BATCH_SMALL if max(h, w) <= 640 else self.MAX_BATCH

    def _encode_tiles(self, x: mx.array) -> mx.array:
        """``_encode_ocr1_features``: NHWC tiles -> [B, h*w, n_embed]."""
        sam = self.sam(x)
        sam = sam.reshape(sam.shape[0], -1, sam.shape[-1])
        clip = self.clip(sam)
        return self.projector(mx.concatenate([clip[:, 1:], sam], axis=-1))

    def _encode_grouped(self, tiles: list[mx.array]) -> list[mx.array]:
        """``_encode_ocr1_features_grouped``: batch equal shapes, <= max_batch."""
        out: list[Optional[mx.array]] = [None] * len(tiles)
        groups: dict = {}
        for idx, t in enumerate(tiles):
            groups.setdefault(tuple(t.shape[1:]), []).append(idx)
        for shape_hwc, idxs in groups.items():
            max_b = self._max_batch_for_shape(shape_hwc)
            batch = mx.concatenate([tiles[i] for i in idxs], axis=0)
            feats = mx.concatenate(
                [
                    self._encode(batch[s : s + max_b])
                    for s in range(0, batch.shape[0], max_b)
                ],
                axis=0,
            )
            pos = 0
            for i in idxs:
                out[i] = feats[pos : pos + tiles[i].shape[0]]
                pos += tiles[i].shape[0]
        return out  # type: ignore[return-value]

    def _format_global(self, feats: mx.array) -> mx.array:
        _, hw, n = feats.shape
        h = w = math.isqrt(hw)
        grid = feats.reshape(h, w, n)
        newline = mx.broadcast_to(self.image_newline, (h, 1, n))
        return mx.concatenate([grid, newline], axis=1).reshape(-1, n)

    def _format_local(self, feats: mx.array, tiles_w: int, tiles_h: int) -> mx.array:
        _, hw, n = feats.shape
        h = w = math.isqrt(hw)
        grid = feats.reshape(tiles_h, tiles_w, h, w, n).transpose(0, 2, 1, 3, 4)
        grid = grid.reshape(tiles_h * h, tiles_w * w, n)
        newline = mx.broadcast_to(self.image_newline, (tiles_h * h, 1, n))
        return mx.concatenate([grid, newline], axis=1).reshape(-1, n)

    def _encode_pages(
        self,
        globals_: list[mx.array],
        locals_: list[Optional[mx.array]],
        shapes: list[tuple[int, int]],
    ) -> mx.array:
        global_feats = self._encode_grouped(globals_)
        local_idx = [j for j, t in enumerate(locals_) if t is not None]
        local_feats: dict[int, mx.array] = {}
        if local_idx:
            encoded = self._encode_grouped([locals_[j] for j in local_idx])
            local_feats = dict(zip(local_idx, encoded))

        pages = []
        sep = self.view_seperator[None, :]
        for jdx, feats in enumerate(global_feats):
            parts = [self._format_global(feats), sep]
            if jdx in local_feats:
                parts.insert(0, self._format_local(local_feats[jdx], *shapes[jdx]))
            pages.append(mx.concatenate(parts, axis=0))
        out = mx.concatenate(pages, axis=0)
        mx.eval(out)
        # SAM peaks at 1.3-1.7 GB; do not keep that cached next to the KV.
        mx.clear_cache()
        return out

    def encode_raw_pages(
        self,
        raw_images: list[Any],
        *,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        tiles_wh: Optional[list[tuple[int, int]]] = None,
    ) -> mx.array:
        """GPU preprocess raw ``uint8 [H, W, 3]`` pages and encode them."""
        from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_preprocess import (
            preprocess_page_mlx,
        )

        globals_, locals_, shapes = [], [], []
        for idx, img in enumerate(raw_images):
            if isinstance(img, mx.array):
                raw = img
            elif isinstance(img, torch.Tensor):
                raw = mx.array(img.detach().cpu().numpy())
            else:
                raw = mx.array(np.asarray(img, dtype=np.uint8))
            t_wh = tiles_wh[idx] if tiles_wh is not None else None
            g, c, shape = preprocess_page_mlx(
                raw,
                cropping=crop_mode,
                base_size=base_size,
                image_size=image_size,
                tiles_wh=t_wh,
                dtype=self.dtype,
            )
            globals_.append(g)
            locals_.append(c)
            shapes.append(shape)
        return self._encode_pages(globals_, locals_, shapes)

    def encode_pixels(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        has_local_crops: Optional[list[bool]] = None,
    ) -> mx.array:
        """Encode pixel values into image features."""
        from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_sam import (
            nchw_to_mlx,
        )

        def to_mlx(t: torch.Tensor) -> mx.array:
            # Round to bf16 first, as the torch path's ``.to(bfloat16)`` does.
            return nchw_to_mlx(t.to(torch.bfloat16), self.dtype)

        globals_, locals_, shapes = [], [], []
        for jdx in range(images_spatial_crop.size(0)):
            patches = images_crop[jdx][0]
            tiles_w, tiles_h = (int(v) for v in images_spatial_crop[jdx][0])
            n_tiles = tiles_w * tiles_h
            if 0 < n_tiles < patches.shape[0]:
                patches = patches[:n_tiles]
            use_local = (
                has_local_crops[jdx]
                if has_local_crops is not None
                else torch.sum(patches).item() != 0
            )
            globals_.append(to_mlx(pixel_values[jdx]))
            locals_.append(to_mlx(patches) if use_local else None)
            shapes.append((tiles_w, tiles_h))

        return self._encode_pages(globals_, locals_, shapes)


class UocrV2VisionTower:
    """Encodes ``MultimodalDataItem`` lists into [num_image_tokens, hidden]."""

    def __init__(self, model_dir: str, hf_config: dict):
        # Pixels are read from host memory.
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.mlx = _MlxVisionTower(model_dir)
        logger.info("Unlimited-OCR v2 vision tower on MLX (bfloat16)")

    @torch.no_grad()
    def encode_pixels(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        has_local_crops: Optional[list[bool]] = None,
    ) -> mx.array:
        """Per-page tensors (``_pixel_values_to_embedding`` layout) -> features.

        ``pixel_values`` [n, 1, 3, H, W], ``images_crop`` [n, 1, P, 3, h, w],
        ``images_spatial_crop`` [n, 1, 2] (tiles w, h).
        """
        return self.mlx.encode_pixels(
            pixel_values, images_crop, images_spatial_crop, has_local_crops
        )

    @torch.no_grad()
    def encode_items(self, mm_items: list[Any]) -> mx.array:
        """``MultimodalDataItem`` list (processor output) -> features.

        Same stacking as ``UnlimitedOCRForCausalLM._process_image_input``;
        items with different crop geometries are encoded one by one, like
        the v2 override.  When items carry raw ``uint8 [H, W, 3]`` pages
        (``uocr_mlx_raw_mode``), preprocessing runs on Metal GPU directly.
        """
        from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM

        if mm_items and all(
            "uocr_mlx_raw_mode" in it.model_specific_data for it in mm_items
        ):
            modes = [
                tuple(it.model_specific_data["uocr_mlx_raw_mode"]) for it in mm_items
            ]
            if len(set(modes)) > 1:
                return mx.concatenate(
                    [self.encode_items([it]) for it in mm_items], axis=0
                )
            base_size, image_size, crop_mode = modes[0]
            tiles = [
                (
                    int(it.images_spatial_crop.flatten()[0]),
                    int(it.images_spatial_crop.flatten()[1]),
                )
                for it in mm_items
            ]
            return self.mlx.encode_raw_pages(
                [it.feature for it in mm_items],
                base_size=int(base_size),
                image_size=int(image_size),
                crop_mode=bool(crop_mode),
                tiles_wh=tiles,
            )

        if len(mm_items) > 1:
            shapes = {tuple(item.images_crop.shape) for item in mm_items}
            if len(shapes) > 1:
                return mx.concatenate(
                    [self.encode_items([it]) for it in mm_items], axis=0
                )
        flags = UnlimitedOCRForCausalLM._collect_mm_flag(mm_items, "has_local_crops")
        pv = torch.stack([item.feature for item in mm_items], dim=0)
        crops = torch.stack([item.images_crop for item in mm_items], dim=0)
        spatial = torch.cat([item.images_spatial_crop for item in mm_items], 0)
        pv = pv.view(pv.shape[0] * pv.shape[1], 1, *pv.shape[2:])
        crops = crops.view(crops.shape[0] * crops.shape[1], 1, *crops.shape[2:])
        spatial = spatial.view(
            spatial.shape[0] * spatial.shape[1], 1, *spatial.shape[2:]
        )
        return self.encode_pixels(pv, crops, spatial, flags)

    @staticmethod
    def to_mlx(features: Features, dtype=mx.bfloat16) -> mx.array:
        """MLX copy of the features (fp32 hop for torch: numpy has no bf16)."""
        if isinstance(features, mx.array):
            return features.astype(dtype)
        arr = features.detach().to(device="cpu", dtype=torch.float32).numpy()
        return mx.array(np.ascontiguousarray(arr)).astype(dtype)
