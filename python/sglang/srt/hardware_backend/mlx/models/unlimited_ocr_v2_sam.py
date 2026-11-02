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
"""MLX port of the Unlimited-OCR SAM ViT-B image encoder.

Mirrors ``ImageEncoderViT`` in ``models/deepseek_ocr.py`` (``build_sam_vit_b``)
but keeps activations NHWC, the native MLX conv layout.  Input-size dependent
tables (interpolated absolute positions, decomposed relative positions) are
built once per geometry with the torch reference helpers, so they match the
CUDA/MPS path exactly.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_mlx import (
    load_checkpoint,
)

_PREFIX = "model.sam_model."

EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
GLOBAL_ATTN_INDEXES = (2, 5, 8, 11)
WINDOW_SIZE = 14
PATCH_SIZE = 16
OUT_CHANS = 256
LN_EPS = 1e-6


def torch_to_mx(t, dtype: mx.Dtype) -> mx.array:
    """Convert a torch tensor to an MLX array."""
    import torch

    return mx.array(t.detach().to("cpu", torch.float32).numpy()).astype(dtype)


def mx_to_torch(a: mx.array):
    """Same-dtype torch copy (bf16 stays bf16), for the torch table helpers."""
    import torch

    t = torch.from_numpy(np.array(a.astype(mx.float32)))
    return t.to(torch.bfloat16) if a.dtype == mx.bfloat16 else t


# torch's bf16 GELU / LayerNorm compute in fp32 and round once; MLX's bf16
# versions round intermediates, which adds ~50% to the per-block drift.
@partial(mx.compile, shapeless=True)
def _gelu_fp32(x: mx.array) -> mx.array:
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


_LN_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    const device T* xr = x + (size_t)row * D;
    float v[PER];
    float s = 0.f;
    for (uint i = 0; i < PER; ++i) { v[i] = float(xr[tid + i * TG]); s += v[i]; }
    threadgroup float buf[TG / 32];
    s = simd_sum(s);
    if (thread_index_in_simdgroup == 0) buf[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.f;
    for (uint i = 0; i < TG / 32; ++i) mean += buf[i];
    mean /= float(D);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q = 0.f;
    for (uint i = 0; i < PER; ++i) { float d = v[i] - mean; q += d * d; }
    q = simd_sum(q);
    if (thread_index_in_simdgroup == 0) buf[simdgroup_index_in_threadgroup] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.f;
    for (uint i = 0; i < TG / 32; ++i) var += buf[i];
    float rstd = metal::precise::rsqrt(var / float(D) + 1.0f / float(EPS_INV));
    device T* o = out + (size_t)row * D;
    for (uint i = 0; i < PER; ++i) {
        uint c = tid + i * TG;
        o[c] = T((v[i] - mean) * rstd * float(w[c]) + float(b[c]));
    }
"""
_ln_kernel = None


def layer_norm_fp32(
    x: mx.array, weight: mx.array, bias: mx.array, eps: float
) -> mx.array:
    """LayerNorm over the last axis: fp32 math, one rounding to ``x.dtype``."""
    global _ln_kernel
    if _ln_kernel is None:
        _ln_kernel = mx.fast.metal_kernel(
            name="uocr_vision_layer_norm",
            input_names=["x", "w", "b"],
            output_names=["out"],
            source=_LN_SOURCE,
        )
    D = x.shape[-1]
    tg = 256 if D % 256 == 0 else 128
    assert D % tg == 0, D
    eps_inv = round(1.0 / eps)  # template args are ints
    return _ln_kernel(
        inputs=[x, weight, bias],
        template=[
            ("T", x.dtype),
            ("D", D),
            ("TG", tg),
            ("PER", D // tg),
            ("EPS_INV", eps_inv),
        ],
        grid=(x.size // D * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


class LayerNorm(nn.Module):
    """LayerNorm implementation for SAM ViT-B."""

    def __init__(self, dim: int, eps: float = LN_EPS):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return layer_norm_fp32(x, self.weight, self.bias, self.eps)


class Attention(nn.Module):
    """Attention implementation with relative position bias."""

    def __init__(self, dim: int, num_heads: int, rel_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rel_pos_h = mx.zeros((rel_len, self.head_dim))
        self.rel_pos_w = mx.zeros((rel_len, self.head_dim))
        self._rel_tables: dict = {}

    def rel_tables(self, h: int, w: int) -> tuple[mx.array, mx.array]:
        """``get_rel_pos`` tables [q, k, head_dim] for an (h, w) query grid."""
        hit = self._rel_tables.get((h, w))
        if hit is None:
            from transformers.models.vitdet.modeling_vitdet import get_rel_pos

            def table(param: mx.array, size: int) -> mx.array:
                ref = get_rel_pos(size, size, mx_to_torch(param))
                return torch_to_mx(ref, param.dtype)

            hit = (table(self.rel_pos_h, h), table(self.rel_pos_w, w))
            self._rel_tables[(h, w)] = hit
        return hit

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        heads, hd = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, H * W, 3, heads, hd).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Decomposed relative position bias (mvitv2), from the unscaled q:
        # bias[q, (kh, kw)] = q . Rh[qh, kh] + q . Rw[qw, kw]. Contiguous
        # rel_h / rel_w make the [q, k] broadcast add ~2x faster (same bits).
        rh, rw = self.rel_tables(H, W)
        r_q = q.reshape(B * heads, H, W, hd)
        rel_h = mx.contiguous(mx.einsum("bhwc,hkc->bhwk", r_q, rh))
        rel_w = mx.contiguous(mx.einsum("bhwc,wkc->bhwk", r_q, rw))
        bias = (rel_h[..., :, None] + rel_w[..., None, :]).reshape(
            B, heads, H * W, H * W
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=bias)
        out = out.transpose(0, 2, 1, 3).reshape(B, H, W, C)
        return self.proj(out)


class MLPBlock(nn.Module):
    """MLP block for SAM ViT-B."""

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(_gelu_fp32(self.lin1(x)))


def window_partition(x: mx.array, ws: int) -> tuple[mx.array, tuple[int, int]]:
    """Partition image into windows."""
    B, H, W, C = x.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h or pad_w:
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // ws, ws, Wp // ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws, ws, C), (Hp, Wp)


def window_unpartition(
    windows: mx.array, ws: int, pad_hw: tuple[int, int], hw: tuple[int, int]
) -> mx.array:
    """Reconstruct image from windows."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // ws // ws)
    x = windows.reshape(B, Hp // ws, Wp // ws, ws, ws, -1).transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class Block(nn.Module):
    """Transformer block for SAM ViT-B."""

    def __init__(self, dim: int, num_heads: int, window_size: int, grid: int):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        rel = grid if window_size == 0 else window_size
        self.attn = Attention(dim, num_heads, 2 * rel - 1)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLPBlock(dim, dim * 4)
        self.window_size = window_size

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


# Channel LayerNorm (neck); channels are already the last axis in NHWC.
LayerNorm2d = LayerNorm


class Conv2dFp32(nn.Conv2d):
    """Conv in fp32. MLX's bf16 3x3 stride-1 conv (Winograd) has ~6% relative
    error at 256 channels, against ~0.2% (output rounding) for the others."""

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv2d(
            x.astype(mx.float32),
            self.weight.astype(mx.float32),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if "bias" in self:
            y = y + self.bias.astype(mx.float32)
        return y.astype(x.dtype)


class SamImageEncoder(nn.Module):
    """SAM ViT-B; ``__call__`` takes NHWC images, returns NHWC [B, h/64, w/64, 1024]."""

    def __init__(self, img_size: int = 1024, net_3_out_channels: int = 1024):
        super().__init__()
        grid = img_size // PATCH_SIZE
        self.patch_embed = nn.Conv2d(
            3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=True
        )
        self.pos_embed = mx.zeros((1, grid, grid, EMBED_DIM))
        self.blocks = [
            Block(
                EMBED_DIM,
                NUM_HEADS,
                0 if i in GLOBAL_ATTN_INDEXES else WINDOW_SIZE,
                grid,
            )
            for i in range(DEPTH)
        ]
        self.neck = [
            nn.Conv2d(EMBED_DIM, OUT_CHANS, kernel_size=1, bias=False),
            LayerNorm2d(OUT_CHANS),
            Conv2dFp32(OUT_CHANS, OUT_CHANS, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(OUT_CHANS),
        ]
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            512, net_3_out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self._abs_pos: dict = {}

    def abs_pos(self, size: int) -> mx.array:
        """``get_abs_pos_sam``: bicubic (antialias) resize of the 64x64 table."""
        if size == self.pos_embed.shape[1]:
            return self.pos_embed
        hit = self._abs_pos.get(size)
        if hit is None:
            import torch
            import torch.nn.functional as F

            # Same recipe as get_abs_pos_sam: resize in fp32, round to dtype.
            pe = self.pos_embed.astype(mx.float32)
            pe = torch.from_numpy(np.array(pe)).permute(0, 3, 1, 2)
            pe = F.interpolate(
                pe,
                size=(size, size),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            )
            hit = torch_to_mx(pe.permute(0, 2, 3, 1), self.pos_embed.dtype)
            self._abs_pos[size] = hit
        return hit

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        x = x + self.abs_pos(x.shape[1])
        for blk in self.blocks:
            x = blk(x)
        for layer in self.neck:
            x = layer(x)
        return self.net_3(self.net_2(x))


def sam_weights_from_state(
    state: dict[str, mx.array], model: SamImageEncoder
) -> list[tuple[str, mx.array]]:
    """Checkpoint -> MLX parameter list of ``model``.

    Torch-layout convs go OIHW -> OHWI; mlx-vlm exports store them as OHWI.
    """
    shapes = {name: p.shape for name, p in tree_flatten(model.parameters())}
    out = []
    for key, value in state.items():
        if not key.startswith(_PREFIX):
            continue
        name = key[len(_PREFIX) :].replace("patch_embed.proj.", "patch_embed.")
        if value.ndim == 4 and value.shape != shapes[name]:
            value = value.transpose(0, 2, 3, 1)
        if value.shape != shapes[name]:
            raise ValueError(
                f"SAM weight {key}: shape {value.shape}, expected {shapes[name]}"
            )
        out.append((name, value))
    return out


def load_sam_encoder(model_dir: str, dtype: mx.Dtype = mx.bfloat16) -> SamImageEncoder:
    """Load SAM image encoder from model directory."""
    model = SamImageEncoder()
    state = load_checkpoint(model_dir, (_PREFIX,))
    weights = [(k, v.astype(dtype)) for k, v in sam_weights_from_state(state, model)]
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def nchw_to_mlx(images, dtype: Optional[mx.Dtype] = mx.bfloat16) -> mx.array:
    """torch NCHW image batch -> MLX NHWC (fp32 hop: numpy has no bfloat16)."""
    import torch

    arr = images.detach().to("cpu", torch.float32).permute(0, 2, 3, 1).numpy()
    arr = mx.array(np.ascontiguousarray(arr))
    return arr.astype(dtype) if dtype is not None else arr
