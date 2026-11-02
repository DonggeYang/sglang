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
"""Unlimited-OCR v2 (DASH) language tower for the SGLang MLX backend.

Port of ``models/unlimited_ocr_v2.py`` and the DASH attention backend: stage-0
``head_cls`` block summaries, entmax-1.5 block router with page isolation,
local window + selected remote blocks attention, and page2d RoPE. Vision
features enter as ``input_embeddings``.
"""

from __future__ import annotations

import glob
import json
import math
import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import msgspec
import numpy as np
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_kernels import (
    MAX_ROUTE_BLOCKS,
    MIN_DEVICE_SIZE,
    attend_params,
    dash_attend,
    dash_route,
    route_params,
)

# Multimodal placeholders are rewritten above this value (schedule_batch.py).
MM_PAD_SHIFT_VALUE = 1_000_000
# Same sentinels as dash_router.py / dash_stage0.py.
ENTMAX_NEG = -1.0e4
ACT_CLAMP = 1.0e4
# Largest finite fp16 (prefill attention runs in fp16).
FP16_MAX = 65504.0
# Prompt tokens of the cheapest page (page2d_rope.MIN_PAGE_TOKENS).
MIN_PAGE_TOKENS = 74


def max_pages_for(context_len: int) -> int:
    """Page-bucket count a request of ``context_len`` tokens can reach."""
    return max(1, int(context_len)) // MIN_PAGE_TOKENS + 2


def _per_layer(cfg: dict, key: str, layer_idx: int) -> Optional[float]:
    """``cfg[key][slot]`` where slot counts sparse layers before ``layer_idx``."""
    arr = cfg.get(key)
    if not isinstance(arr, (list, tuple)) or not arr:
        return None
    freq = cfg.get("sparse_attention_freq")
    slot = layer_idx
    if isinstance(freq, (list, tuple)) and layer_idx < len(freq):
        slot = sum(1 for f in freq[:layer_idx] if f)
    if not 0 <= slot < len(arr):
        return None
    return float(arr[slot])


class DashConfig(msgspec.Struct):
    """Checkpoint ``sparse_attention_config`` resolved for inference."""

    block_size: int
    local_blocks: int
    index_dim: int
    budget_remote: int
    topm: int
    sigma: float
    image_only: bool
    image_bucket_constraint: bool
    image_token_id: int
    page_token_id: Optional[int]
    per_page_image_tokens: int
    idx_scale: list[float]
    prior_floor: list[float]
    rope_tok_theta: float
    rope_page_theta: float
    rope_page_pairs: int
    rope_page_mod: int
    main_tok_freq_mode: str
    index_tok_freq_mode: str
    query_chunk: int

    @property
    def local_window(self) -> int:
        """Local attention window in tokens."""
        return self.local_blocks * self.block_size

    @classmethod
    def from_sparse_config(cls, cfg: dict, num_layers: int) -> DashConfig:
        """Validate the checkpoint config and resolve per-layer router knobs.

        Mirrors ``router_from_sparse_config`` and ``_rope_from_cfg``; raises on
        modes the CUDA backend also rejects instead of silently diverging.
        """
        if str(cfg.get("dash_stage0", "indexer")).lower() != "head_cls":
            raise NotImplementedError("Unlimited-OCR v2 MLX needs dash_stage0=head_cls")
        if int(cfg.get("sparse_num_index_heads", 1)) != 1:
            raise NotImplementedError("the DASH router is single-head")
        for key in ("sparse_main_rope_mode", "sparse_rope_mode"):
            if str(cfg.get(key, "1d")).lower() != "page2d":
                raise NotImplementedError(f"{key} must be page2d")
        if int(cfg.get("sparse_rope_page_span", 0) or 0) > 0:
            raise NotImplementedError("sparse_rope_page_span > 0 is not implemented")
        index_dim = int(cfg.get("sparse_index_dim", 128))
        local_blocks = max(1, int(cfg.get("sparse_local_block", 1)))
        topk_blocks = int(cfg.get("sparse_topk_blocks", 16))
        budget = cfg.get("dash_topk_blocks") or cfg.get("dash_image_fallback_remote")
        budget = int(budget) if budget else max(1, topk_blocks - local_blocks)
        scale_default = float(cfg.get("dash_scaling_factor", 1.0))
        floor_default = float(cfg.get("dash_prior_floor", 1.0e-2))
        idx_scale, prior_floor = [], []
        for layer in range(num_layers):
            s = _per_layer(cfg, "dash_scale_per_layer", layer)
            f = _per_layer(cfg, "dash_floor_per_layer", layer)
            idx_scale.append((scale_default if s is None else s) / math.sqrt(index_dim))
            prior_floor.append(floor_default if f is None else f)
        pt = cfg.get("page_token_id")
        return cls(
            block_size=int(cfg.get("sparse_block_size", 128)),
            local_blocks=local_blocks,
            index_dim=index_dim,
            budget_remote=max(1, budget),
            topm=int(cfg.get("dash_infer_topm", 0) or 16),
            sigma=float(cfg.get("dash_sigma", 1.0)),
            image_only=bool(cfg.get("dash_image_only", True)),
            image_bucket_constraint=bool(cfg.get("image_bucket_constraint", False)),
            image_token_id=int(cfg.get("image_token_id", 128815)),
            page_token_id=int(pt) if pt is not None else None,
            per_page_image_tokens=max(1, int(cfg.get("per_page_image_tokens", 273))),
            idx_scale=idx_scale,
            prior_floor=prior_floor,
            rope_tok_theta=float(cfg.get("sparse_rope_tok_theta", 10000.0)),
            rope_page_theta=float(cfg.get("sparse_rope_page_theta", 256.0)),
            rope_page_pairs=int(cfg.get("sparse_rope_page_pairs", 8)),
            rope_page_mod=int(cfg.get("sparse_rope_page_mod", 0) or 0),
            main_tok_freq_mode=str(
                cfg.get("sparse_main_rope_tok_freq_mode", "truncate")
            ),
            index_tok_freq_mode=str(cfg.get("sparse_rope_tok_freq_mode", "resample")),
            query_chunk=int(cfg.get("dash_query_chunk", 2048) or 2048),
        )


class ModelArgs(msgspec.Struct):
    """Language tower hyperparameters."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    moe_layer_freq: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    vocab_size: int
    rms_norm_eps: float
    eos_token_id: int
    dash: DashConfig

    @property
    def head_dim(self) -> int:
        """Per-head dimension."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf_config(cls, config: dict) -> ModelArgs:
        """Text fields from ``language_config`` with top-level fallbacks."""
        text = dict(config)
        text.update(config.get("language_config") or config.get("text_config") or {})
        sparse = config.get("sparse_attention_config") or text.get(
            "sparse_attention_config"
        )
        if not isinstance(sparse, dict) or sparse.get("training_stage") != "dash":
            raise ValueError("not an Unlimited-OCR v2 (DASH) checkpoint")
        if text.get("use_mla"):
            raise NotImplementedError("MLA language towers are not supported")
        if str(text.get("scoring_func", "softmax")) != "softmax":
            raise NotImplementedError("only softmax MoE routing is supported")
        if str(text.get("topk_method", "greedy")) != "greedy":
            raise NotImplementedError("only greedy MoE top-k is supported")
        num_layers = int(text["num_hidden_layers"])
        eos = text.get("eos_token_id", 1)
        return cls(
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=num_layers,
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            intermediate_size=int(text["intermediate_size"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            n_routed_experts=int(text["n_routed_experts"]),
            n_shared_experts=int(text.get("n_shared_experts") or 0),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            first_k_dense_replace=int(text.get("first_k_dense_replace", 0)),
            moe_layer_freq=int(text.get("moe_layer_freq", 1)),
            norm_topk_prob=bool(text.get("norm_topk_prob", False)),
            routed_scaling_factor=float(text.get("routed_scaling_factor", 1.0)),
            vocab_size=int(text["vocab_size"]),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            eos_token_id=int(eos[0] if isinstance(eos, list) else eos),
            dash=DashConfig.from_sparse_config(sparse, num_layers),
        )


def is_image_token(token_id: int, image_token_id: int) -> bool:
    """Whether ``token_id`` is an image placeholder (raw or pad-shifted)."""
    return token_id >= MM_PAD_SHIFT_VALUE or token_id == image_token_id


class PageState:
    """Per-request page bookkeeping, one token at a time on the CPU.

    Combines ``PageIsolationState.step`` (query page + per-slot image page) and
    ``PageCoordState.step`` (page2d ``(tok_pos, page_pos)``); both share the
    marker count ``mc`` and the prompt-page count ``pp``.  Tokens at absolute
    positions ``>= gen_start`` are generated (response pages restart at 0).
    """

    def __init__(self, dash: DashConfig, max_pages: int):
        self.image_token_id = dash.image_token_id
        self.page_token_id = dash.page_token_id
        self.per_page = dash.per_page_image_tokens
        self.max_pages = max(1, int(max_pages))
        self.mc = 0  # <PAGE> markers seen
        self.ir = 0  # image tokens seen
        self.pp = 0  # marker count as of the last image token
        self.cnt: dict[int, int] = {}
        self.slot_img_page = np.zeros((0,), dtype=np.int32)

    @property
    def num_tokens(self) -> int:
        """Number of tokens stepped so far."""
        return int(self.slot_img_page.shape[0])

    def step(self, ids: list[int], gen_start: int):
        """Advance over ``ids``; returns ``(q_page, tok_pos, page_pos)`` int arrays."""
        start = self.num_tokens
        n = len(ids)
        q_page = np.empty(n, dtype=np.int32)
        tok_pos = np.empty(n, dtype=np.int64)
        page_pos = np.empty(n, dtype=np.int64)
        img_page = np.empty(n, dtype=np.int32)
        mc, ir, pp = self.mc, self.ir, self.pp
        for i, t in enumerate(ids):
            is_img = is_image_token(t, self.image_token_id)
            if self.page_token_id is not None and t == self.page_token_id:
                mc += 1
            if is_img:
                ir += 1
                pp = max(pp, mc)
            is_gen = start + i >= gen_start
            # Isolation: image page of image tokens, query page of everything.
            if is_img:
                p = mc - 1 if mc > 0 else (ir - 1) // self.per_page
                img_page[i] = p
                q_page[i] = p
            else:
                img_page[i] = -1
                if is_gen:
                    r = mc - 1 - pp
                    q_page[i] = r if r >= 0 else -1
                else:
                    q_page[i] = mc - 1 if mc > pp else -1
            # page2d coordinates.
            in_prefill = (not is_gen) and (is_img or mc > pp)
            page = mc if in_prefill else min(max(mc - pp, 0), pp)
            page = min(max(page, 0), self.max_pages - 1)
            c = self.cnt.get(page, 0)
            tok_pos[i] = c
            self.cnt[page] = c + 1
            page_pos[i] = page
        self.mc, self.ir, self.pp = mc, ir, pp
        self.slot_img_page = np.concatenate([self.slot_img_page, img_page])
        return q_page, tok_pos, page_pos

    def block_allow(self, q_page: np.ndarray, num_blocks: int, block_size: int):
        """``same_page | open_row`` per query row over the request's blocks."""
        pages = np.full(num_blocks * block_size, -1, dtype=np.int32)
        pages[: self.num_tokens] = self.slot_img_page[: num_blocks * block_size]
        pages = pages.reshape(num_blocks, block_size)
        bmax = pages.max(axis=1)
        bmin = np.where(pages >= 0, pages, np.iinfo(np.int32).max).min(axis=1)
        bmin = np.where(bmax >= 0, bmin, -1)
        qp = q_page[:, None]
        same = (bmin[None, :] >= 0) & (bmin[None, :] <= qp) & (qp <= bmax[None, :])
        same &= qp >= 0
        open_row = (qp[:, 0] < 0) | ~same.any(axis=1)
        return same | open_row[:, None]


class LazyPageState:
    """Device-side twin of :class:`PageState` for generated tokens.

    Advances on an unevaluated token, so a decode step can be built on the
    previous step's lazy output (chained decode) without a host sync. Only
    generated-token rules are modelled; block page bounds are frozen at
    creation (generated text adds no image blocks). The host ``PageState``
    must be advanced with the real tokens afterwards (``commit``).
    """

    def __init__(self, ps: PageState, dash: DashConfig, max_blocks: int):
        self.page_token_id = -1 if dash.page_token_id is None else dash.page_token_id
        self.image_token_id = dash.image_token_id
        self.per_page = ps.per_page
        self.max_pages = ps.max_pages
        self.mc = mx.array(ps.mc, dtype=mx.int32)
        self.ir = mx.array(ps.ir, dtype=mx.int32)
        self.pp = mx.array(ps.pp, dtype=mx.int32)
        cnt = np.zeros(ps.max_pages, dtype=np.int32)
        for page, c in ps.cnt.items():
            cnt[page] = c
        self.cnt = mx.array(cnt)
        n = ps.num_tokens
        pages = np.full(max_blocks * dash.block_size, -1, dtype=np.int32)
        pages[: min(n, pages.shape[0])] = ps.slot_img_page[: pages.shape[0]]
        pages = pages.reshape(max_blocks, dash.block_size)
        bmax = pages.max(axis=1)
        bmin = np.where(pages >= 0, pages, np.iinfo(np.int32).max).min(axis=1)
        bmin = np.where(bmax >= 0, bmin, -1)
        self.bmin, self.bmax = mx.array(bmin), mx.array(bmax)

    def step(self, tok: mx.array, num_blocks: int):
        """One generated token (int32 [1]) -> (q_page, tok_pos, page_pos, allow)."""
        is_mark = (tok == self.page_token_id).astype(mx.int32)
        is_img = (tok >= MM_PAD_SHIFT_VALUE) | (tok == self.image_token_id)
        mc = self.mc + is_mark
        ir = self.ir + is_img.astype(mx.int32)
        pp = mx.where(is_img, mx.maximum(self.pp, mc), self.pp)
        page = mx.clip(mx.minimum(mx.maximum(mc - pp, 0), pp), 0, self.max_pages - 1)
        tok_pos = mx.take(self.cnt, page)
        self.cnt = self.cnt + (mx.arange(self.max_pages) == page).astype(mx.int32)
        img_page = mx.where(mc > 0, mc - 1, (ir - 1) // self.per_page)
        resp = mc - 1 - pp
        q_page = mx.where(is_img, img_page, mx.where(resp >= 0, resp, -1))
        bmin, bmax = self.bmin[:num_blocks], self.bmax[:num_blocks]
        same = (bmin >= 0) & (bmin <= q_page) & (q_page <= bmax) & (q_page >= 0)
        open_row = (q_page < 0) | ~mx.any(same)
        allow = same | open_row
        self.mc, self.ir, self.pp = mc, ir, pp
        return q_page, tok_pos, page, allow


class PageRope2D:
    """``(tok_pos, page_pos)`` NeoX RoPE tables (``page2d_rope.PageRope2D``)."""

    def __init__(self, dim: int, dash: DashConfig, tok_freq_mode: str):
        half = dim // 2
        pairs = dash.rope_page_pairs
        if not 0 < pairs < half:
            raise ValueError(f"page_pairs={pairs} must be in (0, {half})")
        n_tok = half - pairs
        ar = np.arange(n_tok, dtype=np.float32)
        if tok_freq_mode.lower() == "truncate":
            inv_tok = 1.0 / (dash.rope_tok_theta ** (ar * 2 / dim))
        else:
            inv_tok = 1.0 / (dash.rope_tok_theta ** (ar / n_tok))
        inv_page = 1.0 / (
            dash.rope_page_theta ** (np.arange(pairs, dtype=np.float32) / pairs)
        )
        self.dim = dim
        self.page_mod = dash.rope_page_mod
        self.signature = (dim, n_tok, pairs, tok_freq_mode.lower())
        self._inv_tok = mx.array(inv_tok.astype(np.float32))
        self._inv_page = mx.array(inv_page.astype(np.float32))

    def cos_sin(self, tok_pos: np.ndarray, page_pos: np.ndarray, dtype):
        """[T] int coordinates -> cos/sin [T, dim] (fp32 math, cast to dtype)."""
        if self.page_mod > 0:
            page_pos = np.remainder(page_pos, self.page_mod)
        t = mx.array(tok_pos.astype(np.float32))[:, None]
        p = mx.array(page_pos.astype(np.float32))[:, None]
        ang = mx.concatenate([t * self._inv_tok[None], p * self._inv_page[None]], -1)
        emb = mx.concatenate([ang, ang], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)

    def cos_sin_lazy(self, tok_pos: mx.array, page_pos: mx.array, dtype):
        """:meth:`cos_sin` on (possibly unevaluated) int32 ``mx.array`` coordinates."""
        if self.page_mod > 0:
            page_pos = mx.remainder(page_pos, self.page_mod)
        t = tok_pos.astype(mx.float32)[:, None]
        p = page_pos.astype(mx.float32)[:, None]
        ang = mx.concatenate([t * self._inv_tok[None], p * self._inv_page[None]], -1)
        emb = mx.concatenate([ang, ang], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """NeoX rotate-half on ``x`` [T, H, D] in x's dtype (training parity)."""
    half = x.shape[-1] // 2
    rot = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


class GemmaRMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * (1 + w)`` in fp32 (sglang GemmaRMSNorm)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps
        self._w_plus_1: Optional[mx.array] = None

    def __call__(self, x: mx.array) -> mx.array:
        w1 = self._w_plus_1
        if w1 is None:
            w1 = 1.0 + self.weight.astype(mx.float32)
            self._w_plus_1 = w1
        return mx.fast.rms_norm(x.astype(mx.float32), w1, self.eps).astype(x.dtype)


def entmax15(x: mx.array) -> mx.array:
    """1.5-entmax over the last axis (``dash_router.entmax15``), fp32."""
    x = (x - mx.max(x, axis=-1, keepdims=True)) / 2
    xs = -mx.sort(-x, axis=-1)
    m = x.shape[-1]
    rho = mx.arange(1, m + 1, dtype=x.dtype)
    mean = mx.cumsum(xs, axis=-1) / rho
    mean_sq = mx.cumsum(xs * xs, axis=-1) / rho
    ss = rho * (mean_sq - mean * mean)
    delta = (1 - ss) / rho
    tau = mean - mx.sqrt(mx.maximum(delta, 0))
    support = mx.maximum(mx.sum(tau <= xs, axis=-1, keepdims=True), 1)
    tau_star = mx.take_along_axis(tau, (support - 1).astype(mx.int32), axis=-1)
    return mx.maximum(x - tau_star, 0) ** 2


# Router, stage-0 summaries and decode attention are compiled into a few fused
# kernels; one compiled function per constant set, retraced per input shape.


@lru_cache(maxsize=None)
def _compiled_route(
    block_size: int,
    local_blocks: int,
    idx_scale: float,
    topm: int,
    budget: int,
    floor: float,
    sigma: float,
    image_only: bool,
    index_dim: int,
):
    def route(idx_q, kbar, q_pos, allow):
        nb = kbar.shape[0]
        z = (idx_q.astype(mx.float32) @ kbar.T) / math.sqrt(index_dim)
        z = mx.clip(z, -ACT_CLAMP, ACT_CLAMP)
        q_blk = q_pos // block_size
        remote = mx.arange(nb)[None, :] <= (q_blk[:, None] - local_blocks)
        cand = remote
        if image_only:
            cand = remote & allow
            cand = mx.where(mx.any(cand, axis=-1, keepdims=True), cand, remote)
        x = mx.where(cand, z * idx_scale, ENTMAX_NEG)
        m = max(1, min(topm, nb))
        order = mx.argsort(-x, axis=-1)[:, :m]
        xv = mx.take_along_axis(x, order, axis=-1)
        prior = entmax15(xv)
        is_cand = mx.take_along_axis(cand, order, axis=-1)
        k = max(1, min(budget, m))
        valid = is_cand & (mx.arange(m)[None, :] < k)
        logp = mx.log(mx.maximum(prior, floor))
        cnt = mx.maximum(mx.sum(valid, axis=-1, keepdims=True), 1)
        mu = mx.sum(mx.where(valid, logp, 0.0), axis=-1, keepdims=True) / cnt
        bias = mx.where(valid, (logp - mu) / sigma, 0.0)
        return order, valid, bias

    return mx.compile(route)


@lru_cache(maxsize=None)
def _compiled_summaries(block_size: int, index_dim: int):
    def summaries(keys, first_pos, total, head_cls):
        """keys [n * bs, d] from block ``first_pos // bs`` -> [n, d] fp32."""
        keys = keys.astype(mx.float32).reshape(-1, block_size, index_dim)
        pos = first_pos + mx.arange(keys.shape[0] * block_size)
        valid = (pos < total).reshape(-1, block_size)
        keys = mx.where(valid[..., None], keys, 0.0)
        aw = (keys @ head_cls.astype(mx.float32)) / math.sqrt(index_dim)
        aw = mx.clip(aw, -ACT_CLAMP, ACT_CLAMP)
        row_max = mx.max(mx.where(valid, aw, -mx.inf), axis=-1, keepdims=True)
        row_max = mx.where(mx.any(valid, axis=-1, keepdims=True), row_max, 0)
        e = mx.where(valid, mx.exp(aw - row_max), 0.0)
        w = e / mx.maximum(mx.sum(e, axis=-1, keepdims=True), 1e-6)
        k_bar = mx.sum(w[..., None] * keys, axis=1)
        return mx.nan_to_num(k_bar, nan=0.0, posinf=0.0, neginf=0.0)

    return mx.compile(summaries)


@lru_cache(maxsize=None)
def _compiled_decode_attention(block_size: int, local_window: int, scale: float):
    def attend(qf, kb, vb, idx, valid, bias, q_pos, k_l, v_l):
        """One query over the selected remote blocks plus the local window."""
        n_kv, hd = kb.shape[-2], kb.shape[-1]
        k_r = kb[idx].reshape(-1, n_kv, hd)
        v_r = vb[idx].reshape(-1, n_kv, hd)
        pos_r = (idx[:, None] * block_size + mx.arange(block_size)[None, :]).reshape(-1)
        ok_r = mx.repeat(valid, block_size) & (pos_r <= q_pos - local_window)
        m_r = mx.where(ok_r, mx.repeat(bias, block_size), -mx.inf)
        kf = mx.concatenate([k_r, k_l]).astype(mx.float32).transpose(1, 0, 2)[None]
        vf = mx.concatenate([v_r, v_l]).astype(mx.float32).transpose(1, 0, 2)[None]
        mask = mx.concatenate([m_r, mx.zeros((k_l.shape[0],), dtype=mx.float32)])
        return mx.fast.scaled_dot_product_attention(
            qf, kf, vf, scale=scale, mask=mask[None, None, None, :]
        )

    return mx.compile(attend)


class _GrowBuffer:
    """Preallocated ``[cap, ...]`` array grown in ``step``-row increments."""

    def __init__(self, step: int = 512):
        self.step = step
        self.buf: Optional[mx.array] = None
        self.size = 0

    def write(self, rows: mx.array, start: int) -> mx.array:
        """Write ``rows`` at ``start`` (truncating past it); returns ``[:end]``."""
        end = start + rows.shape[0]
        if self.buf is None or end > self.buf.shape[0]:
            cap = ((end + self.step - 1) // self.step) * self.step
            new = mx.zeros((cap,) + tuple(rows.shape[1:]), dtype=rows.dtype)
            if self.buf is not None and start > 0:
                new[:start] = self.buf[:start]
            self.buf = new
        self.buf[start:end] = rows
        self.size = end
        return self.buf[:end]

    def view(self) -> Optional[mx.array]:
        """Written rows ``[:size]``, or None if empty."""
        return None if self.buf is None else self.buf[: self.size]


class DashLayerCache:
    """One layer's K/V, router index keys and fp32 block summaries."""

    def __init__(self):
        self.keys = _GrowBuffer()
        self.values = _GrowBuffer()
        self.index_keys = _GrowBuffer()
        self.kbar = _GrowBuffer(step=16)
        self.kbar_tokens = 0

    @property
    def state(self) -> list[mx.array]:
        """Arrays to evaluate together with the step."""
        views = (
            self.keys.view(),
            self.values.view(),
            self.index_keys.view(),
            self.kbar.view(),
        )
        return [v for v in views if v is not None]


class UocrV2RequestCache:
    """Per-request cache: per-layer DASH caches plus the CPU page state.

    ``offset`` is the number of committed tokens; ``gen_start`` the absolute
    position of the first generated token (prompt length).
    """

    def __init__(
        self, num_layers: int, dash: DashConfig, max_pages: int, gen_start: int
    ):
        self.layers = [DashLayerCache() for _ in range(num_layers)]
        self.page_state = PageState(dash, max_pages)
        self.offset = 0
        self.gen_start = int(gen_start)
        # Device page state of lazy decode steps and their uncommitted token count.
        self.lazy: Optional[LazyPageState] = None
        self.uncommitted = 0
        # RoPE page offset of a document window, added to pages > 0 (already folded by page_mod).
        self.page_shift = 0

    @property
    def state(self) -> list[mx.array]:
        """All layers' cache arrays."""
        return [a for layer in self.layers for a in layer.state]


class StepMeta(msgspec.Struct):
    """Per-forward metadata shared by every layer of one request."""

    start: int
    num_tokens: int
    cos_main: mx.array
    sin_main: mx.array
    cos_index: mx.array
    sin_index: mx.array
    allow: Optional[mx.array]  # [T, num_blocks] bool, or None (unconstrained)
    # One-token steps within MAX_ROUTE_BLOCKS take the fused Metal kernels.
    fused: bool = False
    # [>= num_blocks] uint8, at least MIN_DEVICE_SIZE long.
    kernel_allow: Optional[mx.array] = None
    # Kernel metadata of a single-request fused step, built at its first layer.
    single: Optional[FusedBatchMeta] = None


class FusedBatchMeta(msgspec.Struct, frozen=True):
    """Kernel metadata of one fused decode step, shared by all layers."""

    route_meta: mx.array  # int32 [B, 2]: (num_blocks, q_blk)
    attend_meta: mx.array  # int32 [B]: q_pos

    @classmethod
    def of(cls, metas: list[StepMeta], block_size: int) -> FusedBatchMeta:
        """For one-token steps at positions ``meta.start``."""
        starts = [m.start for m in metas]
        return cls(
            route_meta=mx.array(
                [[s // block_size + 1, s // block_size] for s in starts],
                dtype=mx.int32,
            ),
            attend_meta=mx.array(starts, dtype=mx.int32),
        )


def _kernel_allow(allow: mx.array) -> mx.array:
    """A [num_blocks] bool row as the fused route kernel's uint8 input."""
    short = MIN_DEVICE_SIZE - allow.shape[0]
    allow = allow.astype(mx.uint8)
    return allow if short <= 0 else mx.pad(allow, (0, short))


def _concat_linear_modules(modules: list[nn.Module]) -> nn.Module:
    """Concatenate Linear or QuantizedLinear modules along output axis 0."""
    first = modules[0]
    if isinstance(first, nn.QuantizedLinear):
        w = mx.concatenate([m.weight for m in modules], axis=0)
        s = mx.concatenate([m.scales for m in modules], axis=0)
        b = (
            mx.concatenate([m.biases for m in modules], axis=0)
            if first.get("biases") is not None
            else None
        )
        out = nn.QuantizedLinear(
            w.shape[1] * 32 // first.bits,
            w.shape[0],
            bias=False,
            group_size=first.group_size,
            bits=first.bits,
            mode=first.mode,
        )
        out.weight = w
        out.scales = s
        if b is not None:
            out.biases = b
        return out
    w = mx.concatenate([m.weight for m in modules], axis=0)
    out = nn.Linear(w.shape[1], w.shape[0], bias=False)
    out.weight = w
    return out


class DashAttention(nn.Module):
    """MHA over DASH block-sparse routing, page2d RoPE on both branches."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dash = args.dash
        self.layer_idx = layer_idx
        self.dash = dash
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        dim, d_idx = args.hidden_size, dash.index_dim
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.index_q_proj = nn.Linear(dim, d_idx, bias=False)
        self.index_k_proj = nn.Linear(dim, d_idx, bias=False)
        self.qkv_proj: Optional[nn.Module] = None
        self.index_qk_proj: Optional[nn.Module] = None
        self.index_q_norm = GemmaRMSNorm(d_idx, args.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(d_idx, args.rms_norm_eps)
        self.head_cls = mx.zeros((d_idx,))

    def fuse_projections(self) -> None:
        """Fuse Q/K/V and index_q/index_k into 2 projections instead of 5."""
        if self.qkv_proj is None and self.q_proj is not None:
            self.qkv_proj = _concat_linear_modules(
                [self.q_proj, self.k_proj, self.v_proj]
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        if self.index_qk_proj is None and self.index_q_proj is not None:
            self.index_qk_proj = _concat_linear_modules(
                [self.index_q_proj, self.index_k_proj]
            )
            self.index_q_proj = None
            self.index_k_proj = None

    def _block_summaries(self, idx_k: mx.array, first_block: int, total: int):
        """Stage-0 ``head_cls`` summaries [nb - first_block, d] fp32.

        ``idx_k`` holds index keys from position 0; rows past ``total`` are
        masked, so the cache's preallocated buffer can be passed as is.
        """
        bs, d = self.dash.block_size, self.dash.index_dim
        nb = (total + bs - 1) // bs
        keys = idx_k[first_block * bs : nb * bs]
        short = (nb - first_block) * bs - keys.shape[0]
        if short > 0:
            keys = mx.concatenate([keys, mx.zeros((short, d), dtype=keys.dtype)])
        return _compiled_summaries(bs, d)(
            keys, mx.array(first_block * bs), mx.array(total), self.head_cls
        )

    def _route(
        self,
        idx_q: mx.array,
        kbar: mx.array,
        q_pos: mx.array,
        allow: Optional[mx.array],
    ):
        """Remote block selection -> ``(order, valid, bias)``, each [T, M].

        ``DashRouterMath.route_and_select`` with the top-M entmax window:
        slot ``i`` reads block ``order[:, i]`` with ``bias[:, i]`` iff
        ``valid[:, i]``.
        """
        dash, layer = self.dash, self.layer_idx
        image_only = dash.image_only and allow is not None
        fn = _compiled_route(
            dash.block_size,
            dash.local_blocks,
            dash.idx_scale[layer],
            dash.topm,
            dash.budget_remote,
            dash.prior_floor[layer],
            dash.sigma,
            image_only,
            dash.index_dim,
        )
        if allow is None:
            allow = mx.ones((idx_q.shape[0], kbar.shape[0]), dtype=mx.bool_)
        return fn(idx_q, kbar, q_pos, allow)

    @staticmethod
    def _densify(order, valid, bias, nb: int):
        """Per-slot route -> ``(block_bias [T, nb], selected [T, nb])``."""
        hit = (order[..., None] == mx.arange(nb)[None, None, :]) & valid[..., None]
        selected = mx.any(hit, axis=1)
        block_bias = mx.sum(mx.where(hit, bias[..., None], 0.0), axis=1)
        return block_bias, selected

    def _attend_mask(self, block_bias, selected, q_pos, total: int) -> mx.array:
        """Additive [T, S] mask (``block_bias`` dtype): local 0, selected remote bias."""
        bs, lw = self.dash.block_size, self.dash.local_window
        blk = mx.where(selected, block_bias, -mx.inf)
        tok = mx.repeat(blk, bs, axis=-1)[:, :total]
        k_pos = mx.arange(total)[None, :]
        qp = q_pos[:, None]
        local = (k_pos <= qp) & (k_pos > qp - lw)
        remote_ok = k_pos <= qp - lw
        return mx.where(local, 0.0, mx.where(remote_ok, tok, -mx.inf))

    def _project(self, x, cos_m, sin_m, cos_i, sin_i):
        """Rows of ``x`` -> roped q/k/v [N, H, D] and router idx_q/idx_k [N, d]."""
        n = x.shape[0]
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(x)
            h_q = self.n_heads * self.head_dim
            h_k = self.n_kv_heads * self.head_dim
            q = qkv[:, :h_q].reshape(n, self.n_heads, self.head_dim)
            k = qkv[:, h_q : h_q + h_k].reshape(n, self.n_kv_heads, self.head_dim)
            v = qkv[:, h_q + h_k :].reshape(n, self.n_kv_heads, self.head_dim)
        else:
            q = self.q_proj(x).reshape(n, self.n_heads, self.head_dim)
            k = self.k_proj(x).reshape(n, self.n_kv_heads, self.head_dim)
            v = self.v_proj(x).reshape(n, self.n_kv_heads, self.head_dim)
        if self.index_qk_proj is not None:
            iqk = self.index_qk_proj(x)
            d_idx = iqk.shape[-1] // 2
            idx_q = self.index_q_norm(iqk[:, :d_idx])
            idx_k = self.index_k_norm(iqk[:, d_idx:])
        else:
            idx_q = self.index_q_norm(self.index_q_proj(x))
            idx_k = self.index_k_norm(self.index_k_proj(x))
        q = apply_rope(q, cos_m, sin_m)
        k = apply_rope(k, cos_m, sin_m)
        idx_q = apply_rope(idx_q[:, None, :], cos_i, sin_i)[:, 0]
        idx_k = apply_rope(idx_k[:, None, :], cos_i, sin_i)[:, 0]
        return q, k, v, idx_q, idx_k

    def _append(self, k, v, idx_k, cache: DashLayerCache, start: int):
        """Write rows at ``start``, refresh stale summaries -> ``(keys, values, kbar)``."""
        total = start + k.shape[0]
        bs = self.dash.block_size
        keys = cache.keys.write(k, start)
        values = cache.values.write(v, start)
        cache.index_keys.write(idx_k, start)
        # Skip kbar recompute mid-block unless a remote block is stale.
        q_blk_max = (total - 1) // bs
        max_remote = q_blk_max - self.dash.local_blocks
        nb = (total + bs - 1) // bs
        need_kbar = (
            k.shape[0] > 1
            or (total % bs == 0)
            or cache.kbar.buf is None
            or cache.kbar.buf.shape[0] < nb
            or (max_remote >= 0 and cache.kbar_tokens < (max_remote + 1) * bs)
        )
        if need_kbar:
            first_block = min(start, cache.kbar_tokens) // bs
            kbar = cache.kbar.write(
                self._block_summaries(cache.index_keys.buf, first_block, total),
                first_block,
            )
            cache.kbar_tokens = total
        else:
            kbar = cache.kbar.buf[:nb]
        return keys, values, kbar

    def _attend(
        self, q, k, v, idx_q, idx_k, cache: DashLayerCache, meta: StepMeta
    ) -> mx.array:
        """Append one request's rows at ``meta.start``, route, attend -> [T, H*D].

        fp32 for one-token steps, fp16 for prefill (:meth:`_attend_dense`).
        """
        T, start, allow = q.shape[0], meta.start, meta.allow
        total = start + T
        keys, values, kbar = self._append(k, v, idx_k, cache, start)
        if meta.fused:
            if meta.single is None:
                meta.single = FusedBatchMeta.of([meta], self.dash.block_size)
            return self._decode_fused(q, idx_q, [cache], [meta], meta.single)
        q_pos = mx.arange(start, total)
        order, valid, bias = self._route(idx_q, kbar, q_pos, allow)
        if T == 1:
            qf = q.astype(mx.float32).transpose(1, 0, 2)[None]
            out = self._attend_decode(qf, cache, order, valid, bias, total)
        else:
            out = self._attend_dense(
                q, keys, values, order, valid, bias, q_pos, kbar.shape[0]
            )
        return out[0].transpose(1, 0, 2).reshape(T, -1)

    def _decode_fused(
        self,
        q,
        idx_q,
        caches: list[DashLayerCache],
        metas: list[StepMeta],
        batch: FusedBatchMeta,
    ) -> mx.array:
        """One query per request through the fused route + attend kernels -> [B, H*D].

        Same math as ``_route`` + ``_attend_decode``; the tests check it against
        dense prefill and against the per-request path.
        """
        dash, layer = self.dash, self.layer_idx
        allows = [m.kernel_allow for m in metas]
        if not dash.image_only or allows[0] is None:
            allows = None
        assert allows is None or all(a is not None for a in allows)
        order, bias = dash_route(
            idx_q,
            [c.kbar.buf for c in caches],
            allows,
            batch.route_meta,
            route_params(dash.idx_scale[layer], dash.prior_floor[layer], dash.sigma),
            local_blocks=dash.local_blocks,
            topm=dash.topm,
            budget=dash.budget_remote,
        )
        out = dash_attend(
            q,
            [c.keys.buf for c in caches],
            [c.values.buf for c in caches],
            order,
            bias,
            batch.attend_meta,
            attend_params(self.scale),
            block_size=dash.block_size,
            local_window=dash.local_window,
        )
        return out.reshape(len(caches), -1)

    def _attend_dense(self, q, keys, values, order, valid, bias, q_pos, nb: int):
        """Prefill: dense SDPA under the [T, S] DASH mask, chunked over queries.

        Runs in fp16, about 3x faster than fp32 SDPA. The SDPA kernels need the
        additive mask in the input dtype, and fp16 rounds the router bias 8x
        finer than bf16 (|bias| <= 4.6 with a 1e-2 prior floor: 2e-3 vs 1.6e-2).
        Scores and softmax still accumulate in fp32.
        """
        total = keys.shape[0]
        block_bias, selected = self._densify(order, valid, bias, nb)

        # [S, H, D] -> [1, H, S, D], clipped so no bf16 value overflows fp16.
        def fp16(x):
            return mx.clip(x, -FP16_MAX, FP16_MAX).astype(mx.float16)

        qh = fp16(q).transpose(1, 0, 2)[None]
        kh = fp16(keys).transpose(1, 0, 2)[None]
        vh = fp16(values).transpose(1, 0, 2)[None]
        block_bias = block_bias.astype(mx.float16)
        T = qh.shape[2]
        chunk = max(1, self.dash.query_chunk)
        outs = []
        for c0 in range(0, T, chunk):
            c1 = min(T, c0 + chunk)
            mask = self._attend_mask(
                block_bias[c0:c1], selected[c0:c1], q_pos[c0:c1], total
            )
            outs.append(
                mx.fast.scaled_dot_product_attention(
                    qh[:, :, c0:c1], kh, vh, scale=self.scale, mask=mask[None, None]
                )
            )
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)

    def _attend_decode(self, qf, cache: DashLayerCache, order, valid, bias, total: int):
        """One query: gather the selected remote blocks plus the local window.

        Reads at most ``topm * block_size + local_window`` keys instead of the
        whole history; the keys left out are exactly the ones the dense mask
        sets to -inf, so the result is the same attention.
        """
        bs, lw = self.dash.block_size, self.dash.local_window
        q_pos = total - 1
        n_kv, hd = self.n_kv_heads, self.head_dim
        # The grow buffers' capacity is a multiple of the block size, so the
        # whole buffer views as [blocks, bs, H, D] without a copy.
        kb = cache.keys.buf.reshape(-1, bs, n_kv, hd)
        vb = cache.values.buf.reshape(-1, bs, n_kv, hd)
        lo = max(0, q_pos - lw + 1)
        return _compiled_decode_attention(bs, lw, self.scale)(
            qf,
            kb,
            vb,
            order[0],
            valid[0],
            bias[0],
            mx.array(q_pos),
            cache.keys.buf[lo:total],
            cache.values.buf[lo:total],
        )

    def __call__(self, x: mx.array, cache: DashLayerCache, meta: StepMeta):
        """``x`` [T, hidden] for one request -> attention output [T, hidden]."""
        q, k, v, idx_q, idx_k = self._project(
            x, meta.cos_main, meta.sin_main, meta.cos_index, meta.sin_index
        )
        out = self._attend(q, k, v, idx_q, idx_k, cache, meta)
        return self.o_proj(out.astype(x.dtype))

    def decode_batch(
        self,
        x: mx.array,
        caches: list[DashLayerCache],
        metas: list[StepMeta],
        rope: tuple,
        batch: Optional[FusedBatchMeta] = None,
    ) -> mx.array:
        """One token per request, projections batched -> [B, hidden].

        With ``batch`` the fused kernels serve all requests; otherwise each
        request routes and attends through :meth:`_attend`.
        """
        q, k, v, idx_q, idx_k = self._project(x, *rope)
        if batch is not None:
            for b, (cache, meta) in enumerate(zip(caches, metas)):
                self._append(
                    k[b : b + 1], v[b : b + 1], idx_k[b : b + 1], cache, meta.start
                )
            out = self._decode_fused(q, idx_q, caches, metas, batch)
            return self.o_proj(out.astype(x.dtype))
        outs = [
            self._attend(
                q[b : b + 1],
                k[b : b + 1],
                v[b : b + 1],
                idx_q[b : b + 1],
                idx_k[b : b + 1],
                cache,
                meta,
            )
            for b, (cache, meta) in enumerate(zip(caches, metas))
        ]
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        return self.o_proj(out.astype(x.dtype))


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SortedSwitchGLU(SwitchGLU):
    """``SwitchGLU`` that sorts (token, expert) pairs for any batch of 2+ tokens.

    Mirrors mlx-lm 0.32 ``SwitchGLU.__call__`` with sorting forced on. mlx-lm
    sorts only from 64 pairs on; a decode batch of 8 with top-6 has 48, and
    each pair then reads its expert's weights on its own. Sorted, the
    quantized ``gather_qmm`` reads each touched expert once per call, which is
    what bounds MoE decode on unified memory. Outputs match up to the
    fp32 summation order of the sorted kernel.
    Unquantized ``gather_mm`` is slower sorted at small batches (M5, 8 tokens),
    so BF16 experts keep mlx-lm's rule.
    """

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if indices.shape[0] <= 1 or not isinstance(self.up_proj, QuantizedSwitchLinear):
            return super().__call__(x, indices)
        k = indices.shape[-1]
        flat = mx.stop_gradient(indices).flatten()
        order = mx.argsort(flat)
        inv_order = mx.argsort(order)
        xs = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[order // k]
        idx = flat[order]
        up = self.up_proj(xs, idx, sorted_indices=True)
        gate = self.gate_proj(xs, idx, sorted_indices=True)
        out = self.down_proj(self.activation(up, gate), idx, sorted_indices=True)
        return mx.unflatten(out[inv_order], 0, indices.shape).squeeze(-2)


class MoE(nn.Module):
    """DeepseekMoE: softmax router, greedy top-k, optional shared experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.switch_mlp = SortedSwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.shared_experts = (
            MLP(args.hidden_size, args.moe_intermediate_size * args.n_shared_experts)
            if args.n_shared_experts
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        scores = mx.softmax(self.gate(x).astype(mx.float32), axis=-1)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k])
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        weights = weights * self.routed_scaling_factor
        y = self.switch_mlp(x, inds)
        y = mx.sum(y * weights[..., None].astype(y.dtype), axis=-2)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    """DASH attention plus dense MLP or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DashAttention(args, layer_idx)
        use_moe = (
            args.n_routed_experts
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        )
        self.mlp = (
            MoE(args) if use_moe else MLP(args.hidden_size, args.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, cache: DashLayerCache, meta: StepMeta):
        h = x + self.self_attn(self.input_layernorm(x), cache, meta)
        return h + self.mlp(self.post_attention_layernorm(h))

    def decode_batch(
        self, x, caches, metas, rope, batch: Optional[FusedBatchMeta] = None
    ):
        """One token per request through the layer."""
        h = x + self.self_attn.decode_batch(
            self.input_layernorm(x), caches, metas, rope, batch
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class UnlimitedOCRV2LanguageModel(nn.Module):
    """Embedding + DASH decoder stack + final norm + lm_head (batch size 1)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        head_dim, d_idx = args.head_dim, args.dash.index_dim
        self.rope_main = PageRope2D(head_dim, args.dash, args.dash.main_tok_freq_mode)
        self.rope_index = PageRope2D(d_idx, args.dash, args.dash.index_tok_freq_mode)

    def make_cache(
        self, prompt_len: int, context_len: int, page_base: int = 0
    ) -> UocrV2RequestCache:
        """New per-request cache sized for ``context_len``.

        ``page_base`` is a document window's global first page; like the CUDA
        backend it is folded by ``page_mod`` and is 0 without one.
        """
        cache = UocrV2RequestCache(
            self.args.num_hidden_layers,
            self.args.dash,
            max_pages_for(context_len),
            prompt_len,
        )
        mod = self.rope_main.page_mod
        cache.page_shift = int(page_base) % mod if mod > 0 else 0
        return cache

    def fuse_projections(self) -> None:
        """Fuse QKV and index Q/K projections across all decoder layers."""
        for layer in self.layers:
            layer.self_attn.fuse_projections()

    def embed(self, token_ids: list[int]) -> mx.array:
        """Text embeddings; image placeholders (pad values) embed as id 0."""
        dash = self.args.dash
        ids = [0 if is_image_token(t, dash.image_token_id) else t for t in token_ids]
        return self.embed_tokens(mx.array(ids, dtype=mx.int32))

    def _step_meta(self, token_ids: list[int], cache: UocrV2RequestCache, dtype):
        dash = self.args.dash
        if cache.uncommitted:
            raise RuntimeError(
                f"{cache.uncommitted} lazily decoded tokens were never committed"
            )
        cache.lazy = None  # the host state moves on; rebuild the device twin
        start, n = cache.offset, len(token_ids)
        q_page, tok_pos, page_pos = cache.page_state.step(token_ids, cache.gen_start)
        if cache.page_shift:
            # Page 0 holds the prefix and instruction, which have no document page.
            page_pos = np.where(page_pos > 0, page_pos + cache.page_shift, page_pos)
        cos_m, sin_m = self.rope_main.cos_sin(tok_pos, page_pos, dtype)
        if self.rope_index.signature == self.rope_main.signature:
            cos_i, sin_i = cos_m, sin_m
        else:
            cos_i, sin_i = self.rope_index.cos_sin(tok_pos, page_pos, dtype)
        allow_np = None
        nb = (start + n + dash.block_size - 1) // dash.block_size
        if dash.image_bucket_constraint and dash.image_only:
            allow_np = cache.page_state.block_allow(q_page, nb, dash.block_size)
        allow = None if allow_np is None else mx.array(allow_np)
        meta = StepMeta(start, n, cos_m, sin_m, cos_i, sin_i, allow)
        if n == 1 and nb <= MAX_ROUTE_BLOCKS:
            meta.fused = True
            if allow is not None:
                meta.kernel_allow = _kernel_allow(allow[0])
        return meta

    def _lazy_step_meta(self, tok: mx.array, cache: UocrV2RequestCache, dtype):
        """:meth:`_step_meta` for one generated, possibly unevaluated token."""
        dash = self.args.dash
        start = cache.offset
        nb = start // dash.block_size + 1
        if cache.lazy is None:
            cache.lazy = LazyPageState(cache.page_state, dash, MAX_ROUTE_BLOCKS)
        _, tok_pos, page_pos, allow = cache.lazy.step(tok, nb)
        if cache.page_shift:
            page_pos = mx.where(page_pos > 0, page_pos + cache.page_shift, page_pos)
        cos_m, sin_m = self.rope_main.cos_sin_lazy(tok_pos, page_pos, dtype)
        if self.rope_index.signature == self.rope_main.signature:
            cos_i, sin_i = cos_m, sin_m
        else:
            cos_i, sin_i = self.rope_index.cos_sin_lazy(tok_pos, page_pos, dtype)
        use_allow = dash.image_bucket_constraint and dash.image_only
        meta = StepMeta(
            start, 1, cos_m, sin_m, cos_i, sin_i, allow[None] if use_allow else None
        )
        meta.fused = True
        if use_allow:
            meta.kernel_allow = _kernel_allow(allow)
        cache.uncommitted += 1
        return meta

    def lazy_decode_ok(self, caches: list[UocrV2RequestCache]) -> bool:
        """Whether every cache can take a lazily built (fused-kernel) step."""
        bs = self.args.dash.block_size
        return all(c.offset // bs + 1 <= MAX_ROUTE_BLOCKS for c in caches)

    def commit(self, cache: UocrV2RequestCache, token_ids: list[int]) -> None:
        """Advance the host page state over tokens decoded through the lazy path."""
        cache.page_state.step(token_ids, cache.gen_start)
        cache.uncommitted -= len(token_ids)

    def hidden_states(
        self,
        token_ids: list[int],
        cache: UocrV2RequestCache,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        """Run ``token_ids`` (appended at ``cache.offset``) -> normed [T, hidden]."""
        h = self.embed(token_ids) if input_embeddings is None else input_embeddings
        meta = self._step_meta(token_ids, cache, h.dtype)
        for layer, layer_cache in zip(self.layers, cache.layers):
            h = layer(h, layer_cache, meta)
        cache.offset += len(token_ids)
        return self.norm(h)

    def __call__(
        self,
        token_ids: list[int],
        cache: UocrV2RequestCache,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        """Last-token logits [1, vocab] fp32."""
        h = self.hidden_states(token_ids, cache, input_embeddings)
        return self.lm_head(h[-1:]).astype(mx.float32)

    def decode_batch(self, token_ids, caches: list[UocrV2RequestCache]) -> mx.array:
        """One new token per request -> logits [B, vocab] fp32.

        Dense parts run batched. When every request fits the fused kernels,
        routing and attention run batched too (one launch per group of at most
        ``MAX_GROUP`` requests); otherwise each request routes on its own.
        ``token_ids`` may be a lazy int32 ``mx.array``: the caller must then
        :meth:`commit` the tokens once known.
        """
        if isinstance(token_ids, mx.array):
            dash = self.args.dash
            toks = token_ids.astype(mx.int32)
            is_img = (toks >= MM_PAD_SHIFT_VALUE) | (toks == dash.image_token_id)
            h = self.embed_tokens(mx.where(is_img, 0, toks))
            metas = [
                self._lazy_step_meta(toks[b : b + 1], c, h.dtype)
                for b, c in enumerate(caches)
            ]
        else:
            h = self.embed(token_ids)
            metas = [
                self._step_meta([t], c, h.dtype) for t, c in zip(token_ids, caches)
            ]
        rope = tuple(
            mx.concatenate([getattr(m, name) for m in metas], axis=0)
            for name in ("cos_main", "sin_main", "cos_index", "sin_index")
        )
        batch = None
        if all(m.fused for m in metas):
            batch = FusedBatchMeta.of(metas, self.args.dash.block_size)
        for li, layer in enumerate(self.layers):
            h = layer.decode_batch(
                h, [c.layers[li] for c in caches], metas, rope, batch
            )
        for c in caches:
            c.offset += 1
        return self.lm_head(self.norm(h)).astype(mx.float32)


_VISION_KEYS = (
    "image_newline",
    ".projector",
    "vision_model",
    "sam_model",
    "view_seperator",
)

# mlx-vlm exports (e.g. mlx-community/Unlimited-OCR-2-4bit) drop the HF
# ``model.`` root and nest the language tower under ``language_model.``.
_MLX_VLM_KEY_PREFIXES = (
    ("language_model.model.", "model."),
    ("language_model.lm_head.", "lm_head."),
    ("vision_model.", "model.vision_model."),
    ("sam_model.", "model.sam_model."),
    ("projector.", "model.projector."),
    ("image_newline", "model.image_newline"),
    ("view_separator", "model.view_seperator"),
)


def checkpoint_key(name: str) -> str:
    """A checkpoint key in the HF layout; HF and pre-quantized keys pass as is."""
    for src, dst in _MLX_VLM_KEY_PREFIXES:
        if name.startswith(src):
            return dst + name[len(src) :]
    return name


def load_checkpoint(model_dir: str, prefixes: tuple[str, ...] = ("",)) -> dict:
    """The (lazily loaded) checkpoint tensors under ``prefixes``, HF key layout."""
    out: dict[str, mx.array] = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        for key, value in mx.load(path).items():
            key = checkpoint_key(key)
            if key.startswith(prefixes):
                out[key] = value
    return out


def sanitize_weights(weights: dict[str, mx.array], args: ModelArgs) -> dict:
    """Checkpoint keys -> this module tree (``_rewrite_v2_names`` parity).

    Vision keys are dropped (the PyTorch vision tower loads them), the trained
    ``dash_router`` shadows the dead ``indexer`` copy, ``mtp`` is dropped and
    routed experts are stacked for ``SwitchGLU``.
    """
    out: dict[str, mx.array] = {}
    dash_layers = {
        k.split(".self_attn.dash_router.")[0]
        for k in weights
        if ".self_attn.dash_router." in k
    }
    for name, w in weights.items():
        if name.startswith("mtp.") or "rotary_emb.inv_freq" in name:
            continue
        if any(key in name for key in _VISION_KEYS):
            continue
        for branch in (".self_attn.dash_router.", ".self_attn.indexer."):
            layer, sep, tail = name.partition(branch)
            if not sep:
                continue
            if branch.endswith("indexer.") and layer in dash_layers:
                name = None
            elif tail == "head_cls":
                name = f"{layer}.self_attn.head_cls"
            else:
                name = f"{layer}.self_attn.index_{tail}"
            break
        if name is None:
            continue
        if name.startswith("model."):
            name = name[len("model.") :]
        out[name] = w
    for layer in range(args.num_hidden_layers):
        prefix = f"layers.{layer}.mlp"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                key0 = f"{prefix}.experts.0.{proj}.{suffix}"
                if key0 not in out:
                    continue
                stacked = [
                    out.pop(f"{prefix}.experts.{e}.{proj}.{suffix}")
                    for e in range(args.n_routed_experts)
                ]
                out[f"{prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(stacked)
    return out


_LAYER_SELECTIONS = ("full", "moe_only", "no_lm_head")


def _quantize_predicate(path: str, module: nn.Module, mode: str = "full") -> bool:
    if not hasattr(module, "to_quantized"):
        return False
    if path.endswith(("index_q_proj", "index_k_proj", "index_qk_proj", "mlp.gate")):
        return False
    if mode == "moe_only":
        return "switch_mlp" in path
    if mode == "no_lm_head":
        return path != "lm_head"
    return True


def quantize_language_model(
    model: nn.Module,
    bits: int,
    group_size: int = 64,
    mode: str = "full",
):
    """Affine-quantize the language model weights.

    ``mode="full"`` quantizes the big projections, the experts and ``lm_head``
    while keeping router-branch projections (128-d) and the MoE gate in bf16.
    ``mode="no_lm_head"`` keeps ``lm_head`` in bf16 as well.
    ``mode="moe_only"`` quantizes only the 64 routed experts (``switch_mlp``).
    """

    def predicate(path: str, module: nn.Module) -> bool:
        return _quantize_predicate(path, module, mode=mode)

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    model.fuse_projections()
    mx.eval(model.parameters())
    mx.clear_cache()


def load_language_model(model_dir: str, dtype=mx.bfloat16, fuse: bool = True):
    """Load the MLX language tower; returns ``(model, args, hf_config)``.

    Accepts bf16 and pre-quantized checkpoints. Defaults ``MLX_ENABLE_TF32=0``:
    TF32 flips near-tied routing decisions and OCR coordinates.
    """
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    args = ModelArgs.from_hf_config(config)
    model = UnlimitedOCRV2LanguageModel(args)
    weights = sanitize_weights(load_checkpoint(model_dir), args)

    quant_cfg = config.get("quantization") or config.get("quantization_config")
    has_scales = any(k.endswith(".scales") for k in weights)
    if quant_cfg or has_scales:
        q_bits = int((quant_cfg or {}).get("bits", 8))
        q_group = int((quant_cfg or {}).get("group_size", 64))
        # ``mode`` is an MLX quantization mode in mlx-vlm exports ("affine")
        # and a layer selection of quantize_language_model in ours ("full").
        q_mode = str((quant_cfg or {}).get("mode", "full"))
        mlx_mode = "affine" if q_mode in _LAYER_SELECTIONS else q_mode
        names = set(weights)

        def class_pred(path: str, module: nn.Module) -> bool:
            if not hasattr(module, "to_quantized"):
                return False
            if f"{path}.scales" in names:
                return True
            if f"{path}.weight" in names:
                return False
            return _quantize_predicate(path, module, mode=q_mode)

        nn.quantize(
            model,
            group_size=q_group,
            bits=q_bits,
            mode=mlx_mode,
            class_predicate=class_pred,
        )
        if any(k.endswith("qkv_proj.weight") for k in weights):
            model.fuse_projections()

    weights = {
        k: (
            v
            if (
                v.dtype == dtype
                or v.dtype in (mx.uint32, mx.uint16, mx.uint8, mx.int32)
            )
            else v.astype(dtype)
        )
        for k, v in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    del weights
    if fuse:
        model.fuse_projections()
    model.eval()
    mx.eval(model.parameters())
    # Loading leaves the dropped vision tensors in MLX's buffer cache.
    mx.clear_cache()
    return model, args, config
