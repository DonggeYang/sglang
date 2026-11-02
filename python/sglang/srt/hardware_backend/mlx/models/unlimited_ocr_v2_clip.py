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
"""MLX port of the Unlimited-OCR CLIP-L tower and linear projector.

Mirrors ``VitModel`` (``build_clip_l``) and ``MlpProjector`` (``linear``) in
``models/deepseek_ocr.py``.  CLIP never sees pixels here: its patch embeddings
are the SAM features, exactly as in ``_encode_ocr1_features``.
"""

from __future__ import annotations

import json
import math
import os
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_mlx import (
    load_checkpoint,
)
from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_sam import (
    LayerNorm,
    torch_to_mx,
)

_CLIP_PREFIX = "model.vision_model."
_PROJ_PREFIX = "model.projector.layers."

HIDDEN = 1024
LAYERS = 24
HEADS = 16
FFN = 4096
LN_EPS = 1e-5


# quick_gelu in fp32, one rounding: op by op in bf16 drifts ~50% more than
# the torch MPS version per MLP.
@partial(mx.compile, shapeless=True)
def _quick_gelu_fp32(x: mx.array) -> mx.array:
    h = x.astype(mx.float32)
    return (h * mx.sigmoid(1.702 * h)).astype(x.dtype)


class ClipAttention(nn.Module):
    """CLIP-L self-attention."""

    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(HIDDEN, HIDDEN * 3, bias=True)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.scale = (HIDDEN // HEADS) ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, HEADS, -1).transpose(2, 0, 3, 1, 4)
        out = mx.fast.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], scale=self.scale
        )
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, S, HIDDEN))


class ClipMLP(nn.Module):
    """CLIP-L MLP with quick_gelu."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, FFN, bias=True)
        self.fc2 = nn.Linear(FFN, HIDDEN, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(_quick_gelu_fp32(self.fc1(x)))


class ClipLayer(nn.Module):
    """Pre-norm CLIP-L encoder layer."""

    def __init__(self):
        super().__init__()
        self.self_attn = ClipAttention()
        self.mlp = ClipMLP()
        self.layer_norm1 = LayerNorm(HIDDEN, LN_EPS)
        self.layer_norm2 = LayerNorm(HIDDEN, LN_EPS)

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.layer_norm1(x))
        return h + self.mlp(self.layer_norm2(h))


class ClipEmbeddings(nn.Module):
    """Class token plus resized absolute position embeddings."""

    def __init__(self, num_positions: int = 257):
        super().__init__()
        self.class_embedding = mx.zeros((HIDDEN,))
        self.position_embedding = nn.Embedding(num_positions, HIDDEN)
        self._pos: dict = {}

    def positions(self, num_tokens: int) -> mx.array:
        """``get_abs_pos``: [num_tokens, HIDDEN], grid part bicubic-resized."""
        table = self.position_embedding.weight
        if num_tokens == table.shape[0]:
            return table
        hit = self._pos.get(num_tokens)
        if hit is None:
            import torch
            import torch.nn.functional as F

            src = int(math.sqrt(table.shape[0] - 1))
            tgt = int(math.sqrt(num_tokens - 1))
            grid = np.array(table[1:].astype(mx.float32)).reshape(1, src, src, HIDDEN)
            grid = torch.from_numpy(grid).permute(0, 3, 1, 2)
            grid = F.interpolate(
                grid,
                size=(tgt, tgt),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            )
            grid = torch_to_mx(
                grid.permute(0, 2, 3, 1).reshape(-1, HIDDEN), table.dtype
            )
            hit = mx.concatenate([table[:1], grid], axis=0)
            self._pos[num_tokens] = hit
        return hit

    def __call__(self, patch_embeds: mx.array) -> mx.array:
        B = patch_embeds.shape[0]
        cls = mx.broadcast_to(self.class_embedding, (B, 1, HIDDEN)).astype(
            patch_embeds.dtype
        )
        x = mx.concatenate([cls, patch_embeds], axis=1)
        return x + self.positions(x.shape[1])


class ClipVisionModel(nn.Module):
    """``VitModel.forward(x, patch_embeds)`` with SAM features as patches."""

    def __init__(self):
        super().__init__()
        self.embeddings = ClipEmbeddings()
        self.pre_layrnorm = LayerNorm(HIDDEN, LN_EPS)
        self.layers = [ClipLayer() for _ in range(LAYERS)]

    def __call__(self, patch_embeds: mx.array) -> mx.array:
        x = self.pre_layrnorm(self.embeddings(patch_embeds))
        for layer in self.layers:
            x = layer(x)
        return x


def _load_prefixed(model_dir: str, prefix: str) -> dict[str, mx.array]:
    return {
        key[len(prefix) :]: value
        for key, value in load_checkpoint(model_dir, (prefix,)).items()
    }


def clip_weights_from_state(state: dict[str, mx.array]) -> list[tuple[str, mx.array]]:
    """Checkpoint names (under ``vision_model.``) -> MLX parameter list."""
    out = []
    for name, value in state.items():
        if name.endswith("position_ids") or name.startswith(
            "embeddings.patch_embedding."
        ):
            continue  # patch embeddings come from SAM
        out.append((name.replace("transformer.layers.", "layers."), value))
    return out


def _load_projector(model_dir: str, dtype: mx.Dtype) -> nn.Module:
    """The linear projector; quantized like the language tower in mlx-vlm exports."""
    state = _load_prefixed(model_dir, _PROJ_PREFIX)
    if "scales" not in state:
        out_dim, in_dim = state["weight"].shape
        projector = nn.Linear(in_dim, out_dim, bias=True)
        projector.load_weights(
            [(k, v.astype(dtype)) for k, v in state.items()], strict=True
        )
        return projector
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    quant = config.get("quantization") or config["quantization_config"]
    group_size, bits = int(quant["group_size"]), int(quant["bits"])
    out_dim = state["weight"].shape[0]
    in_dim = state["scales"].shape[1] * group_size
    projector = nn.QuantizedLinear(
        in_dim,
        out_dim,
        bias=True,
        group_size=group_size,
        bits=bits,
        mode=quant.get("mode", "affine"),
    )
    projector.load_weights(
        [(k, v if k == "weight" else v.astype(dtype)) for k, v in state.items()],
        strict=True,
    )
    return projector


def load_clip_and_projector(
    model_dir: str, dtype: mx.Dtype = mx.bfloat16
) -> tuple[ClipVisionModel, nn.Module]:
    """Load the CLIP-L tower and the linear projector from ``model_dir``."""
    clip = ClipVisionModel()
    weights = clip_weights_from_state(_load_prefixed(model_dir, _CLIP_PREFIX))
    clip.load_weights([(k, v.astype(dtype)) for k, v in weights], strict=True)
    projector = _load_projector(model_dir, dtype)
    mx.eval(clip.parameters(), projector.parameters())
    return clip, projector
