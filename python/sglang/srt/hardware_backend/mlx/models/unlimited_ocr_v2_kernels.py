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
"""Fused Metal kernels for one-query DASH decode (Unlimited-OCR v2).

``dash_route`` (block scores, page isolation, top-M, entmax-1.5 bias) and
``dash_attend`` (selected remote blocks + local window) replace ~60 small MLX
kernels per layer with two launches; both accumulate in fp32. One launch
serves a group of decode requests, each reading its own cache buffers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import mlx.core as mx

# Largest block count one routing threadgroup holds (16 KB of fp32 scores):
# 4096 blocks of 32 tokens cover a 131072-token context.
MAX_ROUTE_BLOCKS = 4096
_ROUTE_THREADS = 256
# Requests per launch. dash_attend binds q, order, bias, meta, params and out
# plus keys/values per request: 6 + 2 * 12 = 30 of Metal's 31 buffer slots.
MAX_GROUP = 12
# mlx (custom_kernel, 0.32) binds inputs with fewer than 8 elements as
# "constant"; the per-request pointers below must be "device", so pad up to it.
MIN_DEVICE_SIZE = 8

_ROUTE_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint NT = THREADS;
    SELECT_BUFFERS
    const int nb = meta[b * 2];
    const int qblk = meta[b * 2 + 1];
    const float idx_scale = params[0];
    const float prior_floor = params[1];
    const float sigma = params[2];

    threadgroup float qs[DI];
    threadgroup float xs[MAXB];
    threadgroup uchar flags[MAXB];   // bit0 remote, bit1 candidate, bit2 taken
    threadgroup float red_v[NT];
    threadgroup int red_i[NT];
    threadgroup int any_cand;
    threadgroup int sel_idx[TOPM];
    threadgroup float sel_x[TOPM];

    for (uint d = tid; d < DI; d += NT) qs[d] = static_cast<float>(idx_q[b * DI + d]);
    if (tid == 0) any_cand = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_sqrt_d = 1.0f / metal::sqrt(float(DI));
    int local_any = 0;
    for (int j = tid; j < nb; j += NT) {
        float z = 0.0f;
        for (uint d = 0; d < DI; ++d) z += qs[d] * kbar[j * DI + d];
        z = metal::clamp(z * inv_sqrt_d, -1.0e4f, 1.0e4f);
        bool remote = j <= qblk - LOCAL;
        bool cand = remote && (USE_ALLOW == 0 || allow[j] != 0);
        xs[j] = z * idx_scale;
        flags[j] = uchar(remote ? 1 : 0) | uchar(cand ? 2 : 0);
        local_any |= cand ? 1 : 0;
    }
    if (local_any) any_cand = 1;  // benign race: every writer stores 1
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // An image filter that empties the candidates falls back to all remote.
    for (int j = tid; j < nb; j += NT) {
        uchar f = flags[j];
        bool cand = any_cand ? ((f & 2) != 0) : ((f & 1) != 0);
        flags[j] = cand ? uchar(2) : uchar(0);
        if (!cand) xs[j] = -1.0e4f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int M = min(TOPM, nb);
    for (int s = 0; s < M; ++s) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int j = tid; j < nb; j += NT) {
            if ((flags[j] & 4) == 0) {
                float v = xs[j];
                if (v > bv || (v == bv && j < bi)) { bv = v; bi = j; }
            }
        }
        red_v[tid] = bv;
        red_i[tid] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint w = NT / 2; w > 0; w >>= 1) {
            if (tid < w) {
                float ov = red_v[tid + w];
                int oi = red_i[tid + w];
                if (ov > red_v[tid] || (ov == red_v[tid] && oi < red_i[tid])) {
                    red_v[tid] = ov;
                    red_i[tid] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            sel_idx[s] = red_i[0];
            sel_x[s] = red_v[0];
            flags[red_i[0]] |= uchar(4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        // entmax-1.5 over the M selected scores (already descending).
        float z0 = sel_x[0];
        float xsrt[TOPM];
        for (int s = 0; s < M; ++s) xsrt[s] = (sel_x[s] - z0) * 0.5f;
        float csum = 0.0f, csq = 0.0f, tau_star = 0.0f;
        int support = 0;
        float taus[TOPM];
        for (int r = 0; r < M; ++r) {
            csum += xsrt[r];
            csq += xsrt[r] * xsrt[r];
            float rho = float(r + 1);
            float mean = csum / rho;
            float mean_sq = csq / rho;
            float ss = rho * (mean_sq - mean * mean);
            float delta = (1.0f - ss) / rho;
            float tau = mean - metal::sqrt(metal::max(delta, 0.0f));
            taus[r] = tau;
            if (tau <= xsrt[r]) support += 1;
        }
        support = max(support, 1);
        tau_star = taus[support - 1];
        float logp[TOPM];
        bool valid[TOPM];
        float mu = 0.0f;
        int cnt = 0;
        for (int s = 0; s < M; ++s) {
            float p = metal::max(xsrt[s] - tau_star, 0.0f);
            p = p * p;
            logp[s] = metal::log(metal::max(p, prior_floor));
            valid[s] = ((flags[sel_idx[s]] & 2) != 0) && (s < BUDGET);
            if (valid[s]) { mu += logp[s]; cnt += 1; }
        }
        mu = mu / float(max(cnt, 1));
        for (int s = 0; s < TOPM; ++s) {
            bool ok = s < M && valid[s];
            order[b * TOPM + s] = ok ? sel_idx[s] : -1;
            bias[b * TOPM + s] = ok ? (logp[s] - mu) / sigma : 0.0f;
        }
    }
"""

_ATTEND_SOURCE = """
    uint h = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    const uint NT = THREADS;
    SELECT_BUFFERS
    const int q_pos = meta[b];
    const float scale = params[0];
    const int NREM = TOPM * BS;
    const int NK = NREM + LW;

    threadgroup float qs[D];
    threadgroup float logit[TOPM * BS + LW];
    threadgroup int kpos[TOPM * BS + LW];
    threadgroup float red[NT];

    for (uint d = tid; d < D; d += NT) qs[d] = static_cast<float>(q[(b * H + h) * D + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (int kk = tid; kk < NK; kk += NT) {
        int pos;
        float blk_bias = 0.0f;
        bool ok;
        if (kk < NREM) {
            int blk = order[b * TOPM + kk / BS];
            pos = blk * BS + (kk % BS);
            ok = blk >= 0 && pos <= q_pos - LW;
            blk_bias = ok ? bias[b * TOPM + kk / BS] : 0.0f;
        } else {
            pos = q_pos - (LW - 1) + (kk - NREM);
            ok = pos >= 0;
        }
        float v = -INFINITY;
        if (ok) {
            const device T* kr = keys + (size_t(pos) * H + h) * D;
            float acc = 0.0f;
            for (uint d = 0; d < D; ++d) acc += qs[d] * static_cast<float>(kr[d]);
            v = acc * scale + blk_bias;
        }
        logit[kk] = v;
        kpos[kk] = ok ? pos : -1;
        local_max = metal::max(local_max, v);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = NT / 2; w > 0; w >>= 1) {
        if (tid < w) red[tid] = metal::max(red[tid], red[tid + w]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (int kk = tid; kk < NK; kk += NT) {
        float p = kpos[kk] >= 0 ? metal::exp(logit[kk] - m) : 0.0f;
        logit[kk] = p;
        local_sum += p;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = NT / 2; w > 0; w >>= 1) {
        if (tid < w) red[tid] += red[tid + w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / red[0];

    for (uint d = tid; d < D; d += NT) {
        float acc = 0.0f;
        for (int kk = 0; kk < NK; ++kk) {
            int pos = kpos[kk];
            if (pos >= 0) acc += logit[kk] * static_cast<float>(values[(size_t(pos) * H + h) * D + d]);
        }
        out[(b * H + h) * D + d] = acc * inv;
    }
"""


def _select(ctype: str, names: tuple[str, ...], n: int) -> str:
    """Metal that points each ``names[i]`` at request ``b``'s ``names[i]_<b>`` input."""
    decl = "".join(f"const device {ctype}* {x} = {x}_0;\n" for x in names)
    cases = "".join(
        f"case {b}: " + " ".join(f"{x} = {x}_{b};" for x in names) + " break;\n"
        for b in range(1, n)
    )
    return decl + (f"switch (b) {{\n{cases}default: break;\n}}" if cases else "")


@lru_cache(maxsize=None)
def _route_kernel(n: int, use_allow: bool):
    per_req = [f"kbar_{b}" for b in range(n)]
    select = _select("float", ("kbar",), n)
    if use_allow:
        per_req += [f"allow_{b}" for b in range(n)]
        select += "\n" + _select("uint8_t", ("allow",), n)
    else:
        select += "\nconst device uint8_t* allow = nullptr;"
    return mx.fast.metal_kernel(
        name=f"uocr_dash_route_{n}_{int(use_allow)}",
        input_names=["idx_q", *per_req, "meta", "params"],
        output_names=["order", "bias"],
        source=_ROUTE_SOURCE.replace("SELECT_BUFFERS", select),
    )


@lru_cache(maxsize=None)
def _attend_kernel(n: int):
    per_req = [f"keys_{b}" for b in range(n)] + [f"values_{b}" for b in range(n)]
    return mx.fast.metal_kernel(
        name=f"uocr_dash_attend_{n}",
        input_names=["q", *per_req, "order", "bias", "meta", "params"],
        output_names=["out"],
        source=_ATTEND_SOURCE.replace(
            "SELECT_BUFFERS", _select("T", ("keys", "values"), n)
        ),
    )


def _groups(n: int) -> list[tuple[int, int]]:
    """Even ``[lo, hi)`` splits of ``n`` requests, at most :data:`MAX_GROUP` each."""
    count = (n + MAX_GROUP - 1) // MAX_GROUP
    size = (n + count - 1) // count
    return [(lo, min(n, lo + size)) for lo in range(0, n, size)]


def _device_sized(a: mx.array) -> mx.array:
    """``a`` zero-padded to at least :data:`MIN_DEVICE_SIZE` elements."""
    short = MIN_DEVICE_SIZE - a.size
    return a if short <= 0 else mx.pad(a.reshape(-1), (0, short))


def _cat(parts: list[mx.array]) -> mx.array:
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


@lru_cache(maxsize=None)
def route_params(idx_scale: float, prior_floor: float, sigma: float) -> mx.array:
    """Per-layer constant input of :func:`dash_route` (cached)."""
    return mx.array([idx_scale, prior_floor, sigma], dtype=mx.float32)


@lru_cache(maxsize=None)
def attend_params(scale: float) -> mx.array:
    """Per-layer constant input of :func:`dash_attend` (cached)."""
    return mx.array([scale], dtype=mx.float32)


def dash_route(
    idx_q: mx.array,
    kbars: list[mx.array],
    allows: Optional[list[mx.array]],
    meta: mx.array,
    params: mx.array,
    *,
    local_blocks: int,
    topm: int,
    budget: int,
) -> tuple[mx.array, mx.array]:
    """B one-query routings -> ``(order [B, topm] int32, -1 unused; bias [B, topm])``.

    ``idx_q`` [B, d]; ``kbars[b]`` [>= nb_b, d] fp32 (rows past ``nb_b`` are
    ignored); ``allows[b]`` [>= nb_b] uint8, or ``allows`` None; ``meta`` int32
    ``[B, 2]`` rows ``(nb, q_blk)`` (shared by all layers of a step); ``params``
    from :func:`route_params`. ``nb`` must not exceed :data:`MAX_ROUTE_BLOCKS`.
    """
    d = idx_q.shape[-1]
    use_allow = allows is not None
    orders, biases = [], []
    for lo, hi in _groups(len(kbars)):
        n = hi - lo
        per_req = kbars[lo:hi] + (
            [_device_sized(a) for a in allows[lo:hi]] if use_allow else []
        )
        order, bias = _route_kernel(n, use_allow)(
            inputs=[idx_q[lo:hi], *per_req, meta[lo:hi], params],
            template=[
                ("DI", d),
                ("LOCAL", local_blocks),
                ("TOPM", topm),
                ("BUDGET", budget),
                ("MAXB", MAX_ROUTE_BLOCKS),
                ("THREADS", _ROUTE_THREADS),
                ("USE_ALLOW", int(use_allow)),
            ],
            grid=(_ROUTE_THREADS, n, 1),
            threadgroup=(_ROUTE_THREADS, 1, 1),
            output_shapes=[(n, topm), (n, topm)],
            output_dtypes=[mx.int32, mx.float32],
        )
        orders.append(order)
        biases.append(bias)
    return _cat(orders), _cat(biases)


def dash_attend(
    q: mx.array,
    keys: list[mx.array],
    values: list[mx.array],
    order: mx.array,
    bias: mx.array,
    meta: mx.array,
    params: mx.array,
    *,
    block_size: int,
    local_window: int,
) -> mx.array:
    """B queries, all heads -> [B, H, D] fp32.

    ``q`` [B, H, D]; ``keys[b]`` / ``values[b]`` [cap_b, H, D] (rows past the
    query are never read); ``order`` / ``bias`` from :func:`dash_route`;
    ``meta`` int32 ``[B]`` query positions; ``params`` from :func:`attend_params`.
    """
    _, n_heads, d = q.shape
    threads = 128
    outs = []
    for lo, hi in _groups(len(keys)):
        n = hi - lo
        (out,) = _attend_kernel(n)(
            inputs=[
                q[lo:hi],
                *keys[lo:hi],
                *values[lo:hi],
                order[lo:hi],
                bias[lo:hi],
                meta[lo:hi],
                params,
            ],
            template=[
                ("T", keys[0].dtype),
                ("H", n_heads),
                ("D", d),
                ("BS", block_size),
                ("LW", local_window),
                ("TOPM", order.shape[1]),
                ("THREADS", threads),
            ],
            grid=(n_heads * threads, n, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(n, n_heads, d)],
            output_dtypes=[mx.float32],
        )
        outs.append(out)
    return _cat(outs)
