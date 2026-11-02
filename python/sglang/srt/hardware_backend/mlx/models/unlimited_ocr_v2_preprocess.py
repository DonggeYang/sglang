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
"""Metal GPU image preprocessing for Unlimited-OCR v2 on MLX.

Bit-exact port of Pillow's fixed-point antialias bicubic resize, with padding,
tiling, normalization and the bf16 cast fused in; emits NHWC ``mx.array``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import mlx.core as mx
import numpy as np

from sglang.srt.configs.deepseek_ocr import find_closest_aspect_ratio

# Pillow ImagingResample.c uses 22-bit fixed-point filter weights (8 + 22 <= 30
# signed bits) with round-half-up at +0.5 (1 << 21) between the two passes.
_PRECISION_BITS = 22

_HORIZ_U8_SOURCE = """
    uint pos = thread_position_in_grid.x;
    uint total = H * OUT_W;
    if (pos >= total) return;
    uint y = pos / OUT_W;
    uint ox = pos % OUT_W;
    int x0 = xmin[ox];
    const device int* kw = kk + ox * KSIZE;
    const device uint8_t* row = img + (size_t)y * IN_W * 3;
    int ss0 = 1 << 21, ss1 = 1 << 21, ss2 = 1 << 21;
    for (int k = 0; k < KSIZE; ++k) {
        int w = kw[k];
        if (w != 0) {
            const device uint8_t* p = row + (x0 + k) * 3;
            ss0 += int(p[0]) * w;
            ss1 += int(p[1]) * w;
            ss2 += int(p[2]) * w;
        }
    }
    device uint8_t* o = out + (size_t)pos * 3;
    o[0] = uint8_t(clamp(ss0 >> 22, 0, 255));
    o[1] = uint8_t(clamp(ss1 >> 22, 0, 255));
    o[2] = uint8_t(clamp(ss2 >> 22, 0, 255));
"""

_VERT_U8_SOURCE = """
    uint pos = thread_position_in_grid.x;
    uint total = OUT_H * W;
    if (pos >= total) return;
    uint oy = pos / W;
    uint x = pos % W;
    int y0 = ymin[oy];
    const device int* kw = kk + oy * KSIZE;
    const device uint8_t* col = img + x * 3;
    int ss0 = 1 << 21, ss1 = 1 << 21, ss2 = 1 << 21;
    for (int k = 0; k < KSIZE; ++k) {
        int w = kw[k];
        if (w != 0) {
            const device uint8_t* p = col + (size_t)(y0 + k) * W * 3;
            ss0 += int(p[0]) * w;
            ss1 += int(p[1]) * w;
            ss2 += int(p[2]) * w;
        }
    }
    device uint8_t* o = out + (size_t)pos * 3;
    o[0] = uint8_t(clamp(ss0 >> 22, 0, 255));
    o[1] = uint8_t(clamp(ss1 >> 22, 0, 255));
    o[2] = uint8_t(clamp(ss2 >> 22, 0, 255));
"""

_VERT_PAD_NORM_SOURCE = """
    uint pos = thread_position_in_grid.x;
    uint total = PAD_H * PAD_W;
    if (pos >= total) return;
    uint py = pos / PAD_W;
    uint px = pos % PAD_W;
    device T* o = out + (size_t)pos * 3;
    if (py < TOP || py >= TOP + OUT_H || px < LEFT || px >= LEFT + OUT_W) {
        o[0] = T((float(FILL_R) * (1.0f / 255.0f) - 0.5f) / 0.5f);
        o[1] = T((float(FILL_G) * (1.0f / 255.0f) - 0.5f) / 0.5f);
        o[2] = T((float(FILL_B) * (1.0f / 255.0f) - 0.5f) / 0.5f);
        return;
    }
    uint oy = py - TOP;
    uint ox = px - LEFT;
    int y0 = ymin[oy];
    const device int* kw = kk + oy * KSIZE;
    const device uint8_t* col = horiz + ox * 3;
    int ss0 = 1 << 21, ss1 = 1 << 21, ss2 = 1 << 21;
    for (int k = 0; k < KSIZE; ++k) {
        int w = kw[k];
        if (w != 0) {
            const device uint8_t* p = col + (size_t)(y0 + k) * OUT_W * 3;
            ss0 += int(p[0]) * w;
            ss1 += int(p[1]) * w;
            ss2 += int(p[2]) * w;
        }
    }
    uint8_t r0 = uint8_t(clamp(ss0 >> 22, 0, 255));
    uint8_t r1 = uint8_t(clamp(ss1 >> 22, 0, 255));
    uint8_t r2 = uint8_t(clamp(ss2 >> 22, 0, 255));
    o[0] = T((float(r0) * (1.0f / 255.0f) - 0.5f) / 0.5f);
    o[1] = T((float(r1) * (1.0f / 255.0f) - 0.5f) / 0.5f);
    o[2] = T((float(r2) * (1.0f / 255.0f) - 0.5f) / 0.5f);
"""

_HORIZ_PAD_NORM_SOURCE = """
    uint pos = thread_position_in_grid.x;
    uint total = PAD_H * PAD_W;
    if (pos >= total) return;
    uint py = pos / PAD_W;
    uint px = pos % PAD_W;
    device T* o = out + (size_t)pos * 3;
    if (py < TOP || py >= TOP + OUT_H || px < LEFT || px >= LEFT + OUT_W) {
        o[0] = T((float(FILL_R) * (1.0f / 255.0f) - 0.5f) / 0.5f);
        o[1] = T((float(FILL_G) * (1.0f / 255.0f) - 0.5f) / 0.5f);
        o[2] = T((float(FILL_B) * (1.0f / 255.0f) - 0.5f) / 0.5f);
        return;
    }
    uint oy = py - TOP;
    uint ox = px - LEFT;
    int x0 = xmin[ox];
    const device int* kw = kk + ox * KSIZE;
    const device uint8_t* row = vert + (size_t)oy * IN_W * 3;
    int ss0 = 1 << 21, ss1 = 1 << 21, ss2 = 1 << 21;
    for (int k = 0; k < KSIZE; ++k) {
        int w = kw[k];
        if (w != 0) {
            const device uint8_t* p = row + (x0 + k) * 3;
            ss0 += int(p[0]) * w;
            ss1 += int(p[1]) * w;
            ss2 += int(p[2]) * w;
        }
    }
    uint8_t r0 = uint8_t(clamp(ss0 >> 22, 0, 255));
    uint8_t r1 = uint8_t(clamp(ss1 >> 22, 0, 255));
    uint8_t r2 = uint8_t(clamp(ss2 >> 22, 0, 255));
    o[0] = T((float(r0) * (1.0f / 255.0f) - 0.5f) / 0.5f);
    o[1] = T((float(r1) * (1.0f / 255.0f) - 0.5f) / 0.5f);
    o[2] = T((float(r2) * (1.0f / 255.0f) - 0.5f) / 0.5f);
"""

_VERT_TILES_NORM_SOURCE = """
    uint pos = thread_position_in_grid.x;
    uint total = OUT_H * OUT_W;
    if (pos >= total) return;
    uint oy = pos / OUT_W;
    uint ox = pos % OUT_W;
    int y0 = ymin[oy];
    const device int* kw = kk + oy * KSIZE;
    const device uint8_t* col = horiz + ox * 3;
    int ss0 = 1 << 21, ss1 = 1 << 21, ss2 = 1 << 21;
    for (int k = 0; k < KSIZE; ++k) {
        int w = kw[k];
        if (w != 0) {
            const device uint8_t* p = col + (size_t)(y0 + k) * OUT_W * 3;
            ss0 += int(p[0]) * w;
            ss1 += int(p[1]) * w;
            ss2 += int(p[2]) * w;
        }
    }
    uint8_t r0 = uint8_t(clamp(ss0 >> 22, 0, 255));
    uint8_t r1 = uint8_t(clamp(ss1 >> 22, 0, 255));
    uint8_t r2 = uint8_t(clamp(ss2 >> 22, 0, 255));
    uint tw = ox / TILE_S;
    uint lx = ox % TILE_S;
    uint th = oy / TILE_S;
    uint ly = oy % TILE_S;
    uint tile_idx = th * TILES_W + tw;
    size_t dst_idx = ((size_t)tile_idx * TILE_S * TILE_S + (size_t)ly * TILE_S + lx) * 3;
    device T* o = out + dst_idx;
    o[0] = T((float(r0) * (1.0f / 255.0f) - 0.5f) / 0.5f);
    o[1] = T((float(r1) * (1.0f / 255.0f) - 0.5f) / 0.5f);
    o[2] = T((float(r2) * (1.0f / 255.0f) - 0.5f) / 0.5f);
"""

_HORIZ_TILES_NORM_SOURCE = """
    uint pos = thread_position_in_grid.x;
    uint total = OUT_H * OUT_W;
    if (pos >= total) return;
    uint oy = pos / OUT_W;
    uint ox = pos % OUT_W;
    int x0 = xmin[ox];
    const device int* kw = kk + ox * KSIZE;
    const device uint8_t* row = vert + (size_t)oy * IN_W * 3;
    int ss0 = 1 << 21, ss1 = 1 << 21, ss2 = 1 << 21;
    for (int k = 0; k < KSIZE; ++k) {
        int w = kw[k];
        if (w != 0) {
            const device uint8_t* p = row + (x0 + k) * 3;
            ss0 += int(p[0]) * w;
            ss1 += int(p[1]) * w;
            ss2 += int(p[2]) * w;
        }
    }
    uint8_t r0 = uint8_t(clamp(ss0 >> 22, 0, 255));
    uint8_t r1 = uint8_t(clamp(ss1 >> 22, 0, 255));
    uint8_t r2 = uint8_t(clamp(ss2 >> 22, 0, 255));
    uint tw = ox / TILE_S;
    uint lx = ox % TILE_S;
    uint th = oy / TILE_S;
    uint ly = oy % TILE_S;
    uint tile_idx = th * TILES_W + tw;
    size_t dst_idx = ((size_t)tile_idx * TILE_S * TILE_S + (size_t)ly * TILE_S + lx) * 3;
    device T* o = out + dst_idx;
    o[0] = T((float(r0) * (1.0f / 255.0f) - 0.5f) / 0.5f);
    o[1] = T((float(r1) * (1.0f / 255.0f) - 0.5f) / 0.5f);
    o[2] = T((float(r2) * (1.0f / 255.0f) - 0.5f) / 0.5f);
"""

_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = (
            mx.fast.metal_kernel(
                name="uocr_bicubic_horiz_u8",
                input_names=["img", "xmin", "kk"],
                output_names=["out"],
                source=_HORIZ_U8_SOURCE,
            ),
            mx.fast.metal_kernel(
                name="uocr_bicubic_vert_u8",
                input_names=["img", "ymin", "kk"],
                output_names=["out"],
                source=_VERT_U8_SOURCE,
            ),
            mx.fast.metal_kernel(
                name="uocr_bicubic_vert_pad_norm",
                input_names=["horiz", "ymin", "kk"],
                output_names=["out"],
                source=_VERT_PAD_NORM_SOURCE,
            ),
            mx.fast.metal_kernel(
                name="uocr_bicubic_horiz_pad_norm",
                input_names=["vert", "xmin", "kk"],
                output_names=["out"],
                source=_HORIZ_PAD_NORM_SOURCE,
            ),
            mx.fast.metal_kernel(
                name="uocr_bicubic_vert_tiles_norm",
                input_names=["horiz", "ymin", "kk"],
                output_names=["out"],
                source=_VERT_TILES_NORM_SOURCE,
            ),
            mx.fast.metal_kernel(
                name="uocr_bicubic_horiz_tiles_norm",
                input_names=["vert", "xmin", "kk"],
                output_names=["out"],
                source=_HORIZ_TILES_NORM_SOURCE,
            ),
        )
    return _KERNELS


def _cubic_filter(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    out = np.zeros_like(x, dtype=np.float64)
    m1 = x < 1.0
    m2 = (x >= 1.0) & (x < 2.0)
    out[m1] = (1.5 * x[m1] - 2.5) * x[m1] * x[m1] + 1.0
    out[m2] = ((-0.5 * x[m2] + 2.5) * x[m2] - 4.0) * x[m2] + 2.0
    return out


@lru_cache(maxsize=256)
def _bicubic_coeffs(in_size: int, out_size: int) -> tuple[mx.array, mx.array, int]:
    """Pillow ``_precompute_coeffs`` for ``Image.BICUBIC`` in 22-bit fixed point."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    ksize = int(np.ceil(support)) * 2 + 1
    centers = (np.arange(out_size, dtype=np.float64) + 0.5) * scale
    xmin = np.floor(centers - support + 0.5).astype(np.int32)
    xmax = np.full(out_size, ksize, dtype=np.int32)
    xmax = np.where(xmin < 0, xmax + xmin, xmax)
    xmin = np.maximum(xmin, 0)
    xmax = np.minimum(xmin + xmax, in_size) - xmin
    ks = np.arange(ksize)[None, :]
    x_arg = (ks + xmin[:, None] - centers[:, None] + 0.5) / filterscale
    w = _cubic_filter(x_arg)
    w = np.where(ks < xmax[:, None], w, 0.0)
    w /= np.sum(w, axis=1, keepdims=True)
    fp_scale = float(1 << _PRECISION_BITS)
    kk = np.where(
        w < 0,
        (w * fp_scale - 0.5).astype(np.int32),
        (w * fp_scale + 0.5).astype(np.int32),
    )
    xmin_mx, kk_mx = mx.array(xmin), mx.array(kk)
    mx.eval(xmin_mx, kk_mx)
    return xmin_mx, kk_mx, ksize


def _identity_coeffs(size: int) -> tuple[mx.array, mx.array, int]:
    """1-tap identity weights ``1 << 22`` when ``in_size == out_size``."""
    xmin = mx.arange(size, dtype=mx.int32)
    kk = mx.full((size, 1), 1 << _PRECISION_BITS, dtype=mx.int32)
    return xmin, kk, 1


def _coeffs_or_identity(in_size: int, out_size: int) -> tuple[mx.array, mx.array, int]:
    if in_size == out_size:
        return _identity_coeffs(in_size)
    return _bicubic_coeffs(in_size, out_size)


def _vertical_first(in_w: int, in_h: int, out_w: int, out_h: int) -> bool:
    """Pillow ``ImagingResample`` pass order: vertical first iff ``y`` shrinks more."""
    return (in_h - out_h > 0) and (in_h - out_h > (in_w - out_w) * 2)


@lru_cache(maxsize=16)
def _target_ratios(min_num: int = 2, max_num: int = 16) -> list[tuple[int, int]]:
    ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(ratios, key=lambda x: x[0] * x[1])


def select_crop_grid(
    width: int, height: int, *, cropping: bool, image_size: int = 640
) -> tuple[int, int]:
    """Tile grid ``(tiles_w, tiles_h)`` chosen by ``UnlimitedOCRHFProcessor``."""
    if not cropping or (width <= 640 and height <= 640):
        return (1, 1)
    return find_closest_aspect_ratio(
        width / height, _target_ratios(), width, height, image_size
    )


def _contain_geometry(
    width: int, height: int, pad_size: int
) -> tuple[int, int, int, int]:
    """``PIL.ImageOps.contain`` + ``ImageOps.pad(..., centering=(0.5, 0.5))``."""
    im_ratio = width / height
    if im_ratio != 1.0:
        if im_ratio > 1.0:
            new_h = round(height / width * pad_size)
            gw, gh = pad_size, new_h if new_h != pad_size else pad_size
        else:
            new_w = round(width / height * pad_size)
            gw, gh = new_w if new_w != pad_size else pad_size, pad_size
    else:
        gw, gh = pad_size, pad_size
    gw = max(1, gw)
    gh = max(1, gh)
    left = round((pad_size - gw) * 0.5) if gw != pad_size else 0
    top = round((pad_size - gh) * 0.5) if gh != pad_size else 0
    return gw, gh, left, top


def _horiz_u8(k_horiz, raw: mx.array, in_h: int, in_w: int, out_w: int) -> mx.array:
    if in_w == out_w:
        return raw
    xmin, kk, ksize = _bicubic_coeffs(in_w, out_w)
    return k_horiz(
        inputs=[raw, xmin, kk],
        template=[("H", in_h), ("IN_W", in_w), ("OUT_W", out_w), ("KSIZE", ksize)],
        grid=(in_h * out_w, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(in_h, out_w, 3)],
        output_dtypes=[mx.uint8],
    )[0]


def _vert_u8(k_vert, raw: mx.array, in_h: int, in_w: int, out_h: int) -> mx.array:
    if in_h == out_h:
        return raw
    ymin, kk, ksize = _bicubic_coeffs(in_h, out_h)
    return k_vert(
        inputs=[raw, ymin, kk],
        template=[("OUT_H", out_h), ("W", in_w), ("KSIZE", ksize)],
        grid=(out_h * in_w, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(out_h, in_w, 3)],
        output_dtypes=[mx.uint8],
    )[0]


def preprocess_page_mlx(
    raw_hwc_u8: mx.array,
    *,
    cropping: bool,
    base_size: int = 1024,
    image_size: int = 640,
    tiles_wh: Optional[tuple[int, int]] = None,
    fill_rgb: tuple[int, int, int] = (127, 127, 127),
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[mx.array, Optional[mx.array], tuple[int, int]]:
    """GPU preprocess one RGB ``uint8`` ``[H, W, 3]`` image into NHWC MLX arrays.

    Returns ``(global_view [1, pad, pad, 3], local_crops [N, S, S, 3] | None,
    (tiles_w, tiles_h))``, matching ``UnlimitedOCRHFProcessor._preprocess_page``
    followed by ``nchw_to_mlx(..., dtype)``.
    """
    H, W, C = raw_hwc_u8.shape
    if C != 3 or raw_hwc_u8.dtype != mx.uint8:
        raise ValueError(
            f"expected uint8 [H, W, 3] image, got {raw_hwc_u8.dtype} {raw_hwc_u8.shape}"
        )
    (
        k_horiz,
        k_vert,
        k_vert_pad,
        k_horiz_pad,
        k_vert_tiles,
        k_horiz_tiles,
    ) = _kernels()
    if tiles_wh is None:
        tiles_w, tiles_h = select_crop_grid(
            W, H, cropping=cropping, image_size=image_size
        )
    else:
        tiles_w, tiles_h = int(tiles_wh[0]), int(tiles_wh[1])

    pad_size = base_size if cropping else image_size
    if image_size <= 640 and not cropping:
        gw, gh, left, top = image_size, image_size, 0, 0
    else:
        gw, gh, left, top = _contain_geometry(W, H, pad_size)

    if _vertical_first(W, H, gw, gh):
        vg = _vert_u8(k_vert, raw_hwc_u8, H, W, gh)
        xmin_g, kx_g, ks_gx = _coeffs_or_identity(W, gw)
        global_view = k_horiz_pad(
            inputs=[vg, xmin_g, kx_g],
            template=[
                ("T", dtype),
                ("PAD_H", pad_size),
                ("PAD_W", pad_size),
                ("OUT_H", gh),
                ("OUT_W", gw),
                ("IN_W", W),
                ("TOP", top),
                ("LEFT", left),
                ("KSIZE", ks_gx),
                ("FILL_R", int(fill_rgb[0])),
                ("FILL_G", int(fill_rgb[1])),
                ("FILL_B", int(fill_rgb[2])),
            ],
            grid=(pad_size * pad_size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, pad_size, pad_size, 3)],
            output_dtypes=[dtype],
        )[0]
    else:
        hg = _horiz_u8(k_horiz, raw_hwc_u8, H, W, gw)
        ymin_g, ky_g, ks_gy = _coeffs_or_identity(H, gh)
        global_view = k_vert_pad(
            inputs=[hg, ymin_g, ky_g],
            template=[
                ("T", dtype),
                ("PAD_H", pad_size),
                ("PAD_W", pad_size),
                ("OUT_H", gh),
                ("OUT_W", gw),
                ("TOP", top),
                ("LEFT", left),
                ("KSIZE", ks_gy),
                ("FILL_R", int(fill_rgb[0])),
                ("FILL_G", int(fill_rgb[1])),
                ("FILL_B", int(fill_rgb[2])),
            ],
            grid=(pad_size * pad_size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, pad_size, pad_size, 3)],
            output_dtypes=[dtype],
        )[0]

    local_crops: Optional[mx.array] = None
    if tiles_w > 1 or tiles_h > 1:
        cw, ch = tiles_w * image_size, tiles_h * image_size
        if _vertical_first(W, H, cw, ch):
            vc = _vert_u8(k_vert, raw_hwc_u8, H, W, ch)
            xmin_c, kx_c, ks_cx = _coeffs_or_identity(W, cw)
            local_crops = k_horiz_tiles(
                inputs=[vc, xmin_c, kx_c],
                template=[
                    ("T", dtype),
                    ("OUT_H", ch),
                    ("OUT_W", cw),
                    ("IN_W", W),
                    ("TILE_S", image_size),
                    ("TILES_W", tiles_w),
                    ("KSIZE", ks_cx),
                ],
                grid=(ch * cw, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(tiles_w * tiles_h, image_size, image_size, 3)],
                output_dtypes=[dtype],
            )[0]
        else:
            hc = _horiz_u8(k_horiz, raw_hwc_u8, H, W, cw)
            ymin_c, ky_c, ks_cy = _coeffs_or_identity(H, ch)
            local_crops = k_vert_tiles(
                inputs=[hc, ymin_c, ky_c],
                template=[
                    ("T", dtype),
                    ("OUT_H", ch),
                    ("OUT_W", cw),
                    ("TILE_S", image_size),
                    ("TILES_W", tiles_w),
                    ("KSIZE", ks_cy),
                ],
                grid=(ch * cw, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(tiles_w * tiles_h, image_size, image_size, 3)],
                output_dtypes=[dtype],
            )[0]

    return global_view, local_crops, (tiles_w, tiles_h)
