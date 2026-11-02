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
"""MLX runner for Unlimited-OCR v2 (DASH) on Apple Silicon.

Same interface as :class:`MlxModelRunner`, but each request owns a
:class:`UocrV2RequestCache` (DASH index keys, block summaries, page state)
instead of KV in the shared pool; the radix cache must be off.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import mlx.core as mx
import msgspec
import numpy as np
import psutil

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.model_runner import (
    _MLX_QUANTIZATION_PRESETS,
    MlxModelRunner,
    MlxPendingDecode,
    MlxPendingExtend,
    MlxPendingPrefill,
)
from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_mlx import (
    is_image_token,
    load_language_model,
    max_pages_for,
    quantize_language_model,
)
from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_vision import (
    UocrV2VisionTower,
)
from sglang.srt.hardware_backend.mlx.remote_code_gate import resolve_model_directory
from sglang.srt.hardware_backend.mlx.sampling import (
    GREEDY_PARAMS,
    MlxLogprobSpec,
    MlxSamplingParams,
)
from sglang.srt.models.unlimited_ocr_v2_support.document_reuse import (
    CONTEXT_PAGES,
    page_base,
)
from sglang.srt.models.unlimited_ocr_v2_support.window_params import KEY_LOOKAHEAD
from sglang.srt.sampling.custom_logit_processor import (
    NGRAM_OUTPUT_ONLY_KEY,
    CustomLogitProcessor,
    DeepseekOCRNoRepeatNGramLogitProcessor,
)

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.uocr_v2_doc_reuse import (
        MlxDocReuse,
        MlxPrefixSeed,
    )

logger = logging.getLogger(__name__)

# Default MLX buffer-cache cap (GB) when SGLANG_MLX_CACHE_LIMIT_GB is unset.
_DEFAULT_CACHE_LIMIT_GB = 1.0


@lru_cache(maxsize=64)
def _ngram_index_tables(ngram_size: int, window_size: int) -> tuple[mx.array, mx.array]:
    """Precomputed candidate prefix and next-token index tables for n-gram guard."""
    if ngram_size <= 0 or window_size < ngram_size:
        return mx.zeros((0, 0), dtype=mx.int32), mx.zeros((0,), dtype=mx.int32)
    n_cand = window_size - ngram_size + 1
    p_len = ngram_size - 1
    if p_len > 0:
        cand_idx = (
            np.arange(n_cand, dtype=np.int32)[:, None]
            + np.arange(p_len, dtype=np.int32)[None, :]
        )
    else:
        cand_idx = np.zeros((n_cand, 0), dtype=np.int32)
    next_pos = np.arange(n_cand, dtype=np.int32) + p_len
    return mx.array(cand_idx), mx.array(next_pos)


@lru_cache(maxsize=128)
def _compiled_uocr_logit_guard(
    vocab_size: int,
    ngram_size: int,
    window_size: int,
    page_guard_active: bool,
    page_token_id: int,
    eos_id: int,
    max_page_tokens: int,
    expected_pages: int,
    stop_only: bool,
    end_on_marker: bool,
    rep_min: int,
    rep_max: int,
    rep_min_count: int,
    has_whitelist: bool,
):
    """Compile a static-config in-graph MLX operator for DeepseekOCRNoRepeatNGramLogitProcessor."""
    cand_idx, next_pos = _ngram_index_tables(ngram_size, window_size)

    def _guard_row(
        logits_row: mx.array,
        win: mx.array,
        out_win: mx.array,
        n_out: mx.array,
        marks: mx.array,
        last_mark: mx.array,
        whitelist_arr: mx.array,
    ) -> mx.array:
        out = logits_row
        forced_tok = mx.array(-1, dtype=mx.int32)

        # Periodic tail repeat check (opt-in).
        if rep_max > 0 and rep_min_count >= 2 and 0 <= eos_id < vocab_size:
            lo = max(1, rep_min)
            rep_hit = mx.array(False)
            r_len = out_win.shape[0]
            for plen in range(lo, rep_max + 1):
                needed = plen * rep_min_count
                if needed > r_len:
                    break
                tail = out_win[r_len - needed :].reshape(rep_min_count, plen)
                valid = tail[0, 0] >= 0
                same = mx.all(tail[1:] == tail[0:1])
                rep_hit = rep_hit | (valid & same)
            forced_tok = mx.where(rep_hit, eos_id, forced_tok)

        # Page guard.
        if page_guard_active:
            cond1 = marks > expected_pages
            forced_tok = mx.where((forced_tok < 0) & cond1, eos_id, forced_tok)
            if not stop_only:
                if max_page_tokens > 0:
                    cond2 = (n_out - (last_mark + 1)) >= max_page_tokens
                    target2 = mx.where(marks < expected_pages, page_token_id, eos_id)
                    forced_tok = mx.where((forced_tok < 0) & cond2, target2, forced_tok)
                no_force = forced_tok < 0
                ban_eos = no_force & (marks < expected_pages)
                out = out.at[eos_id].add(mx.where(ban_eos, -mx.inf, 0.0))
                if not end_on_marker:
                    ban_page = no_force & (marks == expected_pages)
                    out = out.at[page_token_id].add(mx.where(ban_page, -mx.inf, 0.0))

        # N-gram repeat blocking.
        if ngram_size > 0 and window_size >= ngram_size:
            next_toks = win[next_pos]
            valid_pos = win[next_pos - (ngram_size - 1)] >= 0
            if ngram_size > 1:
                prefixes = win[cand_idx]
                cur_prefix = win[window_size - (ngram_size - 1) :][None, :]
                prefix_valid = mx.all(cur_prefix >= 0)
                matched = (
                    valid_pos & prefix_valid & mx.all(prefixes == cur_prefix, axis=-1)
                )
            else:
                matched = valid_pos
            in_vocab = (next_toks >= 0) & (next_toks < vocab_size)
            if has_whitelist:
                is_white = mx.any(next_toks[:, None] == whitelist_arr[None, :], axis=-1)
                should_ban = matched & in_vocab & (~is_white) & (forced_tok < 0)
            else:
                should_ban = matched & in_vocab & (forced_tok < 0)
            safe_idx = mx.where(should_ban, next_toks, 0)
            out = out.at[safe_idx].add(mx.where(should_ban, -mx.inf, 0.0))

        # Forced token: every other logit goes to -inf.
        is_forced = forced_tok >= 0
        f_idx = mx.maximum(forced_tok, 0)
        orig_val = out[f_idx]
        out = mx.where(is_forced, -mx.inf, out)
        out[f_idx] = mx.where(is_forced, 0.0, orig_val)
        return out

    return mx.compile(_guard_row)


class _MlxLazyGuardState:
    """Device-side twin of _MlxUocrGuardConfig for chained decode steps."""

    def __init__(
        self,
        cfg: _MlxUocrGuardConfig,
        full_seq: list[int],
    ):
        self.cfg = cfg
        w = max(cfg.window_size, 1)
        if cfg.output_only:
            seq = full_seq[cfg.prompt_len :]
        else:
            seq = full_seq
        tail = seq[-w:] if len(seq) >= w else ([-1] * (w - len(seq)) + list(seq))
        self.win = mx.array(tail, dtype=mx.int32)

        if cfg.rep_win_size > 0:
            r = cfg.rep_win_size
            out_seq = full_seq[cfg.prompt_len :]
            rtail = (
                out_seq[-r:]
                if len(out_seq) >= r
                else ([-1] * (r - len(out_seq)) + list(out_seq))
            )
            self.out_win = mx.array(rtail, dtype=mx.int32)
        else:
            self.out_win = cfg.dummy_out_win

        self.n_out = mx.array(cfg.n_out, dtype=mx.int32)
        self.marks = mx.array(cfg.marks, dtype=mx.int32)
        self.last_mark = mx.array(cfg.last_mark, dtype=mx.int32)

    def advance(self, tok_1d: mx.array) -> None:
        """Advance device window & page counters by one (possibly lazy) token [1]."""
        tok = tok_1d.astype(mx.int32)
        if self.win.shape[0] > 1:
            self.win = mx.concatenate([self.win[1:], tok])
        else:
            self.win = tok
        if self.cfg.rep_win_size > 0:
            if self.out_win.shape[0] > 1:
                self.out_win = mx.concatenate([self.out_win[1:], tok])
            else:
                self.out_win = tok
        if self.cfg.page_guard_active:
            is_mark = tok[0] == self.cfg.page_token_id
            self.marks = self.marks + is_mark.astype(mx.int32)
            self.last_mark = mx.where(is_mark, self.n_out, self.last_mark)
            self.n_out = self.n_out + 1


class _MlxUocrGuardConfig:
    """Per-request host & device state for MLX-native DeepseekOCRNoRepeatNGramLogitProcessor."""

    dummy_out_win = mx.array([-1], dtype=mx.int32)
    empty_whitelist = mx.zeros((0,), dtype=mx.int32)

    def __init__(
        self,
        req: Any,
        vocab_size: int,
        prompt_ids: list[int],
    ):
        params = req.sampling_params.custom_params or {}
        self.req = req
        self.vocab_size = int(vocab_size)
        self.prompt_len = len(prompt_ids)
        # Tokens the (re-)prefill feeds: a retracted request feeds its output too.
        # Only the chunk that ends here samples a token the guard counts.
        self.fill_len = self.prompt_len
        self.output_only = bool(params.get(NGRAM_OUTPUT_ONLY_KEY))

        try:
            self.rep_max = max(0, int(params.get("rep_max_pattern_size") or 0))
            self.rep_min_count = max(0, int(params.get("rep_min_count") or 0))
            self.rep_min = max(0, int(params.get("rep_min_pattern_size") or 0))
            eos_raw = params.get("eos_token_id")
            self.eos_id = int(eos_raw) if eos_raw is not None else -1
        except (TypeError, ValueError):
            self.rep_max = self.rep_min_count = self.rep_min = 0
            self.eos_id = -1

        self.rep_active = (
            self.rep_max > 0
            and self.rep_min_count >= 2
            and 0 <= self.eos_id < self.vocab_size
        )
        self.rep_win_size = self.rep_max * self.rep_min_count if self.rep_active else 0

        try:
            self.max_page_tokens = max(0, int(params.get("max_page_tokens") or 0))
            pt_raw = params.get("page_token_id")
            self.page_token_id = int(pt_raw) if pt_raw is not None else -1
        except (TypeError, ValueError):
            self.max_page_tokens = 0
            self.page_token_id = -1

        self.stop_only = bool(params.get("page_stop_only"))
        lookahead = int(params.get(KEY_LOOKAHEAD) or 0)
        # Same page count as the torch processor's ``_page_guard``.
        doc = req.uocr_doc
        context_pages = doc.context_pages if doc is not None else 0
        self.end_on_marker = lookahead > 0 or (doc is not None and doc.end_on_marker)
        self.expected_pages = (
            sum(1 for t in prompt_ids if t == self.page_token_id)
            - context_pages
            - lookahead
            if self.page_token_id >= 0
            else 0
        )

        self.page_guard_active = (
            0 <= self.page_token_id < self.vocab_size
            and 0 <= self.eos_id < self.vocab_size
            and self.expected_pages > 0
            and not (self.stop_only and not self.end_on_marker)
        )

        try:
            self.ngram_size = max(0, int(params.get("ngram_size") or 0))
            self.window_size = max(0, int(params.get("window_size") or 0))
        except (TypeError, ValueError):
            self.ngram_size = self.window_size = 0

        if self.ngram_size <= 0 or self.window_size < self.ngram_size:
            self.ngram_size = 0
            self.window_size = 0

        whitelist_ids = params.get("whitelist_token_ids") or []
        try:
            wl = sorted(
                {int(t) for t in whitelist_ids if 0 <= int(t) < self.vocab_size}
            )
        except (TypeError, ValueError):
            wl = []
        self.has_whitelist = len(wl) > 0
        self.whitelist_arr = (
            mx.array(wl, dtype=mx.int32) if self.has_whitelist else self.empty_whitelist
        )

        # Host page guard counters (kept in sync with committed tokens)
        self.n_out = 0
        self.marks = 0
        self.last_mark = -1
        self.lazy: Optional[_MlxLazyGuardState] = None

        self._compiled_fn = _compiled_uocr_logit_guard(
            self.vocab_size,
            self.ngram_size,
            self.window_size,
            self.page_guard_active,
            self.page_token_id,
            self.eos_id,
            self.max_page_tokens,
            self.expected_pages,
            self.stop_only,
            self.end_on_marker,
            self.rep_min,
            self.rep_max,
            self.rep_min_count,
            self.has_whitelist,
        )

    def commit_tokens(self, tokens: list[int]) -> None:
        """Advance host page-guard counters over newly committed output tokens."""
        for tok in tokens:
            if tok == self.page_token_id:
                self.marks += 1
                self.last_mark = self.n_out
            self.n_out += 1

    def apply_fresh(self, logits_row: mx.array, full_seq: list[int]) -> mx.array:
        """Build fresh lazy state from committed host state and edit logits_row [V]."""
        self.lazy = _MlxLazyGuardState(self, full_seq)
        st = self.lazy
        return self._compiled_fn(
            logits_row,
            st.win,
            st.out_win,
            st.n_out,
            st.marks,
            st.last_mark,
            self.whitelist_arr,
        )

    def apply_chained(
        self, logits_row: mx.array, prev_lazy_tok_1d: mx.array, full_seq: list[int]
    ) -> mx.array:
        """Advance device state by prev_lazy_tok_1d [1] and edit logits_row [V]."""
        if self.lazy is None:
            self.lazy = _MlxLazyGuardState(self, full_seq)
        self.lazy.advance(prev_lazy_tok_1d)
        st = self.lazy
        return self._compiled_fn(
            logits_row,
            st.win,
            st.out_win,
            st.n_out,
            st.marks,
            st.last_mark,
            self.whitelist_arr,
        )


class _ImageFeatures(msgspec.Struct):
    """A request's vision features and the absolute positions they fill."""

    positions: np.ndarray  # sorted absolute positions of image tokens
    features: mx.array  # [len(positions), hidden]


class RetainedRows(msgspec.Struct):
    """KV rows a finished document window keeps for its follow-up."""

    positions: np.ndarray  # sorted sequence positions of the rows
    layers: list  # per layer: (keys [n, H, D], values [n, H, D], index_keys [n, d])

    def index(self, positions: list[int]) -> mx.array:
        """Row indices of ``positions``; raises if one was not retained."""
        src = np.asarray(positions, dtype=np.int64)
        at = np.searchsorted(self.positions, src)
        if np.any(at >= self.positions.shape[0]) or np.any(
            self.positions[np.minimum(at, self.positions.shape[0] - 1)] != src
        ):
            raise RuntimeError("document reuse reads rows the window did not retain")
        return mx.array(at.astype(np.int32))


class _PendingSplice(msgspec.Struct):
    """Text tail spliced after a follow-up window's prefill (``_splice_text``)."""

    rows: Optional[RetainedRows]
    positions: list
    forced: list
    seq_len: int


class MlxDocReuseBackend:
    """What ``MlxDocReuse`` (``uocr_v2_doc_reuse.py``) reads off the runner.

    The scheduler-side bookkeeping runs unchanged on the stub pools; the KV
    itself lives in per-request caches, so retention copies the kept rows out
    of the finished request and a follow-up copies them into its new cache.
    """

    def __init__(self, runner: MlxUocrV2ModelRunner):
        dash = runner.args.dash
        self._runner = runner
        self.doc_reuse: Optional[MlxDocReuse] = None
        self.block_size = dash.block_size
        self.page_state = SimpleNamespace(
            page_token_id=dash.page_token_id,
            image_token_id=dash.image_token_id,
            per_page_image_tokens=dash.per_page_image_tokens,
        )
        self.coord_state = SimpleNamespace(max_pages=max_pages_for(runner._context_len))

    def attach_doc_reuse(self, doc_reuse: MlxDocReuse) -> None:
        """Keep the scheduler's ``MlxDocReuse``; the runner takes prefix seeds from it."""
        self.doc_reuse = doc_reuse

    def snapshot_rows(self, req_id: str, ranges) -> Optional[RetainedRows]:
        """Copy the rows in ``ranges`` out of a finished request's cache."""
        return self._runner.snapshot_rows(req_id, ranges)

    def committed_len(self, req_id: str) -> int:
        """Positions of the request's cache that hold KV (all but the next input)."""
        toks = self._runner._req_token_ids.get(req_id)
        return max(0, len(toks) - 1) if toks else 0

    def queue_splice(self, req_id: str, rows, positions, forced, seq_len: int) -> None:
        """Splice the text tail when the runner finalizes this prefill.

        Without overlap scheduling the prefill is already finalized here.
        """
        runner = self._runner
        runner._pending_splice[req_id] = _PendingSplice(
            rows, list(positions), list(forced), int(seq_len)
        )
        if len(runner._req_token_ids.get(req_id, ())) == seq_len + 1:
            runner._apply_splice(req_id, late=True)


class MlxUocrV2ModelRunner(MlxModelRunner):
    """Unlimited-OCR v2 runner: per-request DASH caches, no shared KV pool."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
        disable_radix_cache: bool = True,
        pool_size: int | None = None,
        mem_fraction_static: float = 0.8,
        quantization: str | None = None,
        revision: str | None = None,
        enable_sampling: bool = False,
        sampling_rng_seed: int = 0,
        deterministic_seeding: bool = False,
        context_len: int | None = None,
    ):
        if not disable_radix_cache:
            # A resumed prefix would skip the page counters (same as CUDA).
            raise ValueError("Unlimited-OCR v2 on MLX requires --disable-radix-cache")
        if quantization is not None and quantization not in _MLX_QUANTIZATION_PRESETS:
            raise NotImplementedError(
                f"--quantization={quantization} is not supported for "
                f"Unlimited-OCR v2 on MLX; use one of {sorted(_MLX_QUANTIZATION_PRESETS)}"
            )
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.disable_radix_cache = True
        self._quantization = quantization
        self._mem_fraction_static = mem_fraction_static
        self._enable_sampling = enable_sampling
        self._deterministic_seeding = deterministic_seeding
        self._sanitize_nan = envs.SGLANG_SANITIZE_NAN_LOGITS.get()
        self._rng_key = mx.random.key(sampling_rng_seed) if enable_sampling else None
        self._decode_step_ct = 0
        self._clear_steps = envs.SGLANG_MLX_CLEAR_CACHE_STEPS.get()
        self._attention_kv_pool = None
        self._req_to_token_pool = None
        self._trunk = None

        # Decode transients grow with the sequence; cap the MLX buffer cache.
        cache_limit_gb = envs.SGLANG_MLX_CACHE_LIMIT_GB.get()
        if cache_limit_gb is None:
            cache_limit_gb = _DEFAULT_CACHE_LIMIT_GB
        mx.set_cache_limit(int(cache_limit_gb * (1024**3)))

        start = time.time()
        model_dir = str(resolve_model_directory(model_path, revision=revision))
        if quantization is not None:
            self.model, self.args, hf_config = load_language_model(
                model_dir, fuse=False
            )
            if hf_config.get("quantization") or hf_config.get("quantization_config"):
                logger.info(
                    "Model at %s is already pre-quantized (%s); skipping runtime quantization",
                    model_dir,
                    hf_config.get("quantization")
                    or hf_config.get("quantization_config"),
                )
                self.model.fuse_projections()
            else:
                bits, group_size = _MLX_QUANTIZATION_PRESETS[quantization]
                quantize_language_model(self.model, bits, group_size)
                logger.info(
                    "Quantized the Unlimited-OCR v2 language tower: bits=%d "
                    "group_size=%d (%.2f GB active)",
                    bits,
                    group_size,
                    mx.get_active_memory() / 1024**3,
                )
        else:
            self.model, self.args, hf_config = load_language_model(model_dir)
        # The vision tower is built lazily: when images arrive already encoded
        # from the tokenizer process (precomputed_embeddings), it is never used.
        self._vision_dir = model_dir
        self._vision_hf_config = hf_config
        self._vision: Optional[UocrV2VisionTower] = None
        logger.info(
            "Unlimited-OCR v2 MLX model loaded in %.2fs (MLX buffer cache "
            "limit %.1f GB)",
            time.time() - start,
            cache_limit_gb,
        )
        # No mx.set_wired_limit: it pushed the rest of the system to swap on 16 GB machines.

        self._context_len = int(context_len or 131072)
        self._req_token_ids: dict[str, list[int]] = {}
        self._req_sampling: dict[str, MlxSamplingParams] = {}
        self._req_caches: dict[str, list[Any]] = {}
        self._req_pool_idx: dict[str, int] = {}
        self._req_synced_offset: dict[str, int] = {}
        self._req_images: dict[str, Optional[_ImageFeatures]] = {}
        self._req_guard: dict[str, Optional[_MlxUocrGuardConfig]] = {}
        self._pending_splice: dict[str, _PendingSplice] = {}
        self._pool_size = self._compute_pool_size(pool_size)
        # Document windows (--uocr-doc-split) keep and splice KV through it.
        self.doc_reuse_backend = MlxDocReuseBackend(self)

    def supports_native_custom_logit_processor(self, batch: Any) -> bool:
        """True if every custom logit processor in ``batch`` runs in-graph here."""
        sinfo = batch.sampling_info
        if sinfo is None or not sinfo.custom_logit_processor:
            return True
        return all(
            isinstance(entry.processor, DeepseekOCRNoRepeatNGramLogitProcessor)
            for entry in sinfo.custom_logit_processor.values()
        )

    def _init_req_guard(
        self, req_id: str, req: Any | None, prompt_ids: list[int]
    ) -> None:
        if req is None or not req.custom_logit_processor:
            self._req_guard[req_id] = None
            return
        proc = CustomLogitProcessor.from_str(req.custom_logit_processor)
        if not isinstance(proc, DeepseekOCRNoRepeatNGramLogitProcessor):
            self._req_guard[req_id] = None
            return
        cfg = _MlxUocrGuardConfig(req, self.args.vocab_size, prompt_ids)
        # A retracted request re-prefills with the output it already generated.
        if req.output_ids:
            cfg.commit_tokens(list(req.output_ids))
            cfg.fill_len += len(req.output_ids)
        self._req_guard[req_id] = cfg

    # -- sizing and no-op pool hooks --------------------------------------

    def _bytes_per_token(self) -> int:
        a = self.args
        kv = 2 * a.num_key_value_heads * a.head_dim * 2  # bf16 K + V
        index = a.dash.index_dim * 2 + a.dash.index_dim * 4 // a.dash.block_size
        return a.num_hidden_layers * (kv + index)

    def _compute_pool_size(self, explicit_size: int | None) -> int:
        """Token budget for the scheduler, from the memory left after load."""
        if explicit_size is not None:
            return explicit_size
        info = mx.device_info()
        limit = info.get("max_recommended_working_set_size", info.get("memory_size", 0))
        usable = int(limit * self._mem_fraction_static) - mx.get_active_memory()
        sys_avail = int(psutil.virtual_memory().available * self._mem_fraction_static)
        budget = max(min(usable, sys_avail), 0)
        size = max(budget // self._bytes_per_token(), 4096)
        logger.info(
            "Unlimited-OCR v2 MLX token budget: %d tokens (%.2f GB, %d B/token)",
            size,
            budget / 1024**3,
            self._bytes_per_token(),
        )
        return size

    def init_cache_pools(self, req_to_token_pool) -> None:
        """Keep the req-to-token pool; there is no shared KV pool."""
        self._req_to_token_pool = req_to_token_pool

    def _first_attention_cache(self, cache: list[Any]) -> Any:
        return cache[0]

    def _store_auxiliary_state(self, req_pool_idx: int, cache: list[Any]) -> None:
        return None

    def _release_cache(self, cache: list[Any]) -> None:
        return None

    def store_auxiliary_state_for_request(self, req_id: str) -> None:
        """No-op: state lives in the per-request cache."""
        return None

    def remove_request(self, req_id: str):
        """Drop a request's cache, image features and logit guard."""
        super().remove_request(req_id)
        self._req_images.pop(req_id, None)
        self._req_guard.pop(req_id, None)
        self._pending_splice.pop(req_id, None)
        if not self._req_caches:
            # Idle: return the finished requests' KV buffers to the system.
            mx.clear_cache()

    def clear(self):
        """Drop all per-request state."""
        super().clear()
        self._req_images.clear()
        self._req_guard.clear()
        self._pending_splice.clear()

    # -- document windows ----------------------------------------------------

    def snapshot_rows(self, req_id: str, ranges) -> Optional[RetainedRows]:
        """Evaluated copies of a request's KV rows in ``ranges``; None if absent."""
        caches = self._req_caches.get(req_id)
        if caches is None or not ranges:
            return None
        cache = caches[0]
        pos = np.concatenate([np.arange(lo, hi, dtype=np.int64) for lo, hi in ranges])
        pos = np.unique(pos[pos < cache.offset])
        if pos.size == 0:
            return None
        idx = mx.array(pos.astype(np.int32))
        layers = [
            (lc.keys.buf[idx], lc.values.buf[idx], lc.index_keys.buf[idx])
            for lc in cache.layers
        ]
        mx.eval(*[a for rows in layers for a in rows])
        return RetainedRows(positions=pos, layers=layers)

    @staticmethod
    def _copy_rows(cache, rows: RetainedRows, src_positions, dst_start: int) -> None:
        """Write retained rows at ``dst_start``; stale block summaries get rebuilt."""
        idx = rows.index(src_positions)
        for lc, (k, v, ik) in zip(cache.layers, rows.layers):
            lc.keys.write(k[idx], dst_start)
            lc.values.write(v[idx], dst_start)
            lc.index_keys.write(ik[idx], dst_start)
            lc.kbar_tokens = min(lc.kbar_tokens, dst_start)

    def _seed_reused_prefix(
        self, cache, seed: MlxPrefixSeed, full_token_ids: list[int]
    ) -> None:
        """Copy the previous window's tail as this window's prompt head."""
        self._copy_rows(cache, seed.rows, seed.src_positions, 0)
        # Same page state the CUDA path writes for the copied slots.
        cache.page_state.step(list(full_token_ids[: seed.copy_len]), cache.gen_start)
        cache.offset = seed.copy_len

    def _apply_splice(self, req_id: str, late: bool = False) -> bool:
        """Append the spliced text tail and forced marker; True if one was queued.

        ``late``: the prefill's sample was already committed to the guard.
        """
        sp = self._pending_splice.pop(req_id, None)
        if sp is None:
            return False
        cache = self._req_caches[req_id][0]
        if cache.offset != sp.seq_len:
            raise RuntimeError(
                f"request {req_id}: splice at {sp.seq_len}, cache holds {cache.offset}"
            )
        n = len(sp.positions)
        if n:
            self._copy_rows(cache, sp.rows, sp.positions, sp.seq_len)
            cache.offset += n
        ps = cache.page_state
        ps.slot_img_page = np.concatenate([ps.slot_img_page, np.full(n, -1, np.int32)])
        # Decode resumes past the context page: its text is spliced, not decoded.
        ps.mc += CONTEXT_PAGES
        cache.lazy = None
        # The prefill's own sample is dropped; decode continues from the marker.
        toks = self._req_token_ids[req_id]
        self._req_token_ids[req_id] = toks[:-1] + list(sp.forced)
        guard = self._req_guard.get(req_id)
        if guard is not None:
            if late:
                guard = _MlxUocrGuardConfig(guard.req, self.args.vocab_size, toks[:-1])
                self._req_guard[req_id] = guard
            guard.commit_tokens(list(sp.forced))
        return True

    # -- multimodal inputs ---------------------------------------------------

    @property
    def vision(self) -> UocrV2VisionTower:
        """Vision tower, built on first local encode (fallback path)."""
        if self._vision is None:
            logger.info("Building the MLX vision tower in the scheduler (local encode)")
            self._vision = UocrV2VisionTower(self._vision_dir, self._vision_hf_config)
        return self._vision

    def _encode_request_images(
        self, req: Any, full_token_ids: list[int], from_offsets: bool = False
    ) -> Optional[_ImageFeatures]:
        """Run the vision tower once per request; None for text-only requests.

        A reused window's prompt head holds image tokens whose items were
        dropped (their KV is copied), so its positions come from the items.
        """
        mm = req.multimodal_inputs if req is not None else None
        items = [it for it in mm.mm_items if it.is_image()] if mm is not None else []
        if not items:
            return None
        image_id = self.args.dash.image_token_id
        if from_offsets:
            positions = np.array(
                [p for it in items for s, e in it.offsets for p in range(s, e + 1)],
                dtype=np.int64,
            )
        else:
            positions = np.array(
                [
                    i
                    for i, t in enumerate(full_token_ids)
                    if is_image_token(t, image_id)
                ],
                dtype=np.int64,
            )
        start = time.perf_counter()
        feats, n_pre = self._item_features(items)
        if feats.shape[0] != positions.shape[0]:
            raise RuntimeError(
                f"request {req.rid}: {feats.shape[0]} image features for "
                f"{positions.shape[0]} image tokens"
            )
        out = _ImageFeatures(positions, feats)
        logger.info(
            "vision features: %d images (%d precomputed in tokenizer) -> %d tokens in %.2fs",
            len(items),
            n_pre,
            feats.shape[0],
            time.perf_counter() - start,
        )
        return out

    def _item_features(self, items: list[Any]) -> tuple[mx.array, int]:
        """Features of ``items`` in order -> (mx.array [T, hidden], n_precomputed).

        Items encoded in the tokenizer process carry ``precomputed_embeddings``;
        any that do not are encoded here on the local vision tower.
        """
        pre = [it.precomputed_embeddings for it in items]
        n_pre = sum(p is not None for p in pre)
        if n_pre == len(items):
            return mx.concatenate(
                [UocrV2VisionTower.to_mlx(p) for p in pre], axis=0
            ), n_pre
        if n_pre == 0:
            return UocrV2VisionTower.to_mlx(self.vision.encode_items(items)), 0
        missing = [it for it, p in zip(items, pre) if p is None]
        enc = iter(
            mx.split(
                UocrV2VisionTower.to_mlx(self.vision.encode_items(missing)),
                np.cumsum([sum(e - s + 1 for s, e in it.offsets) for it in missing])[
                    :-1
                ].tolist(),
            )
            if len(missing) > 1
            else [UocrV2VisionTower.to_mlx(self.vision.encode_items(missing))]
        )
        parts = [
            UocrV2VisionTower.to_mlx(p) if p is not None else next(enc) for p in pre
        ]
        return mx.concatenate(parts, axis=0), n_pre

    def _chunk_embeddings(
        self, req_id: str, token_ids: list[int], start: int
    ) -> Optional[mx.array]:
        """Text embeddings with this chunk's image positions replaced."""
        images = self._req_images.get(req_id)
        if images is None:
            return None
        end = start + len(token_ids)
        lo, hi = np.searchsorted(images.positions, [start, end])
        if lo == hi:
            return None
        emb = self.model.embed(token_ids)
        local = mx.array((images.positions[lo:hi] - start).astype(np.int32))
        emb[local] = images.features[int(lo) : int(hi)].astype(emb.dtype)
        return emb

    # -- forward -------------------------------------------------------------

    def _forward_chunk(
        self,
        req_id: str,
        cache: Any,
        token_ids: list[int],
        needs_logits: bool,
        logit_edit_row: mx.array | None,
        logprob_spec: MlxLogprobSpec | None,
        full_seq_for_guard: list[int] | None = None,
    ):
        """One prefill/extend chunk -> (lazy next token [1], lazy logprobs)."""
        emb = self._chunk_embeddings(req_id, token_ids, cache.offset)
        hidden = self.model.hidden_states(token_ids, cache, input_embeddings=emb)
        if not needs_logits:
            # Discarded output of a non-final chunk: skip the vocab head.
            return self._dummy_next_token(hidden[None]), None
        logits = self.model.lm_head(hidden[-1:]).astype(mx.float32)
        edits = logit_edit_row[None, :] if logit_edit_row is not None else None
        if edits is not None:
            logits = logits + edits
            edits = None
        guard = self._req_guard.get(req_id)
        if guard is not None and full_seq_for_guard is not None:
            logits = guard.apply_fresh(logits[0], full_seq_for_guard)[None, :]
        return self._select_tokens_with_logprobs(
            logits, [req_id], [[cache]], edits, logprob_spec
        )

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
    ) -> MlxPendingPrefill:
        """First prefill chunk: create the request cache and encode its images."""
        # A follow-up document window starts on the previous window's copied tail.
        doc_reuse = self.doc_reuse_backend.doc_reuse
        seed = (
            doc_reuse.take_prefix(req_id)
            if prefix_slot_ids and doc_reuse is not None
            else None
        )
        if prefix_slot_ids and (seed is None or seed.copy_len != len(prefix_slot_ids)):
            raise RuntimeError("Unlimited-OCR v2 on MLX has no prefix cache")
        if self._enable_sampling:
            self._req_sampling[req_id] = (
                MlxSamplingParams.from_req(
                    req, deterministic_seeding=self._deterministic_seeding
                )
                if req is not None
                else GREEDY_PARAMS
            )
        # A retracted request re-prefills prompt + output; generated tokens
        # keep their response-page numbering.
        prompt_ids = list(req.origin_input_ids if req is not None else full_token_ids)
        prompt_len = len(prompt_ids)
        self._init_req_guard(req_id, req, prompt_ids)
        cache = self.model.make_cache(
            prompt_len,
            self._context_len,
            page_base=page_base(req) if req is not None else 0,
        )
        if seed is not None:
            self._seed_reused_prefix(cache, seed, full_token_ids)
        # fill ids stop at this chunk; image positions span the whole prompt.
        all_ids = list(
            req.full_untruncated_fill_ids if req is not None else full_token_ids
        )
        self._req_images[req_id] = self._encode_request_images(
            req, all_ids, from_offsets=seed is not None
        )
        lazy_token, lazy_logprobs = self._forward_chunk(
            req_id,
            cache,
            list(new_token_ids),
            needs_logits,
            logit_edit_row,
            logprob_spec,
            full_seq_for_guard=list(full_token_ids) if needs_logits else None,
        )
        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=[cache],
            req_id=req_id,
            full_token_ids=list(full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=lazy_logprobs,
        )

    def prefill_finalize(self, pending: MlxPendingPrefill) -> int:
        """Sync the prefill token and commit it to the logit guard."""
        next_token = super().prefill_finalize(pending)
        if self._apply_splice(pending.req_id):
            return next_token
        guard = self._req_guard.get(pending.req_id)
        if guard is not None:
            # Only the final prompt chunk produces an output token.
            if len(pending.full_token_ids) >= guard.fill_len:
                guard.commit_tokens([next_token])
        return next_token

    def extend_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        new_slot_ids: list[int],
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
    ) -> MlxPendingExtend:
        """Chunked-prefill continuation on the request's own cache."""
        cache = self._req_caches[req_id]
        full_seq = None
        if needs_logits and self._req_guard.get(req_id) is not None:
            prev = self._req_token_ids.get(req_id, [])
            full_seq = (prev[:-1] if prev else []) + list(new_token_ids)
        lazy_token, lazy_logprobs = self._forward_chunk(
            req_id,
            cache[0],
            list(new_token_ids),
            needs_logits,
            logit_edit_row,
            logprob_spec,
            full_seq_for_guard=full_seq,
        )
        return MlxPendingExtend(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            new_token_ids=list(new_token_ids),
            new_synced_offset=0,
            lazy_logprobs=lazy_logprobs,
        )

    def extend_finalize(self, pending: MlxPendingExtend) -> int:
        """Sync the extend token and commit it to the logit guard."""
        next_token = super().extend_finalize(pending)
        if self._apply_splice(pending.req_id):
            return next_token
        guard = self._req_guard.get(pending.req_id)
        if guard is not None:
            if len(self._req_token_ids[pending.req_id]) > guard.fill_len:
                guard.commit_tokens([next_token])
        return next_token

    def _decode_logits(self, req_ids: list[str], tokens: list[int]) -> mx.array:
        """One token per request in one batched step -> [B, vocab] fp32."""
        caches = [self._req_caches[rid][0] for rid in req_ids]
        return self.model.decode_batch(list(tokens), caches)

    def _apply_native_guards_fresh(
        self, last_logits: mx.array, req_ids: list[str]
    ) -> mx.array:
        if not any(self._req_guard.get(rid) is not None for rid in req_ids):
            return last_logits
        rows = []
        for b, rid in enumerate(req_ids):
            guard = self._req_guard.get(rid)
            row = last_logits[b]
            if guard is not None:
                row = guard.apply_fresh(row, self._req_token_ids[rid])
            rows.append(row)
        return mx.stack(rows, axis=0)

    def _apply_native_guards_chained(
        self, last_logits: mx.array, req_ids: list[str], prev_lazy_tokens: mx.array
    ) -> mx.array:
        if not any(self._req_guard.get(rid) is not None for rid in req_ids):
            return last_logits
        rows = []
        for b, rid in enumerate(req_ids):
            guard = self._req_guard.get(rid)
            row = last_logits[b]
            if guard is not None:
                row = guard.apply_chained(
                    row, prev_lazy_tokens[b : b + 1], self._req_token_ids[rid]
                )
            rows.append(row)
        return mx.stack(rows, axis=0)

    def decode_batch_start(
        self,
        req_ids: list[str],
        edit_rows: mx.array | None = None,
        logprob_spec: MlxLogprobSpec | None = None,
        logits_hook=None,
    ) -> MlxPendingDecode:
        """One batched decode step from the committed last tokens."""
        caches = [self._req_caches[rid] for rid in req_ids]
        tokens = [self._req_token_ids[rid][-1] for rid in req_ids]
        last_logits = self._decode_logits(list(req_ids), tokens)

        if edit_rows is not None:
            last_logits = last_logits.astype(mx.float32) + edit_rows
        if logits_hook is not None:
            last_logits = self._run_logits_hook(last_logits, logits_hook)
        else:
            last_logits = self._apply_native_guards_fresh(last_logits, req_ids)

        lazy_tokens, lazy_logprobs = self._select_tokens_with_logprobs(
            last_logits, list(req_ids), caches, None, logprob_spec
        )
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=list(req_ids),
            caches=caches,
            lazy_logprobs=lazy_logprobs,
            logprob_spec=logprob_spec,
            edit_rows=edit_rows,
        )

    def decode_batch_start_chained(self, prev: MlxPendingDecode) -> MlxPendingDecode:
        """Next decode step on ``prev``'s lazy tokens, without a host sync."""
        caches = [c[0] for c in prev.caches]
        if self.model.lazy_decode_ok(caches):
            last_logits = self.model.decode_batch(prev.lazy_tokens, caches)
            if prev.edit_rows is not None:
                last_logits = last_logits.astype(mx.float32) + prev.edit_rows
            last_logits = self._apply_native_guards_chained(
                last_logits, prev.req_ids, prev.lazy_tokens
            )
        else:
            raw = prev.lazy_tokens.tolist()
            tokens = [int(t) for t in (raw if isinstance(raw, list) else [raw])]
            last_logits = self._decode_logits(prev.req_ids, tokens)
            if prev.edit_rows is not None:
                last_logits = last_logits.astype(mx.float32) + prev.edit_rows
            if any(self._req_guard.get(rid) is not None for rid in prev.req_ids):
                rows = []
                for b, (rid, tok) in enumerate(zip(prev.req_ids, tokens)):
                    guard = self._req_guard.get(rid)
                    row = last_logits[b]
                    if guard is not None:
                        row = guard.apply_chained(
                            row,
                            mx.array([tok], dtype=mx.int32),
                            self._req_token_ids[rid],
                        )
                    rows.append(row)
                last_logits = mx.stack(rows, axis=0)
        lazy_tokens, lazy_logprobs = self._select_tokens_with_logprobs(
            last_logits, prev.req_ids, prev.caches, None, prev.logprob_spec
        )
        return MlxPendingDecode(
            lazy_tokens=lazy_tokens,
            req_ids=prev.req_ids,
            caches=prev.caches,
            lazy_logprobs=lazy_logprobs,
            logprob_spec=prev.logprob_spec,
            edit_rows=prev.edit_rows,
        )

    def decode_batch_finalize(self, pending: MlxPendingDecode) -> list[int]:
        """Sync decode tokens and commit them to page state and logit guard."""
        next_tokens = super().decode_batch_finalize(pending)
        for rid, tok in zip(pending.req_ids, next_tokens):
            cache = self._req_caches.get(rid)
            # A chained successor already consumed this token on the device.
            if cache is not None and cache[0].uncommitted:
                self.model.commit(cache[0], [tok])
            guard = self._req_guard.get(rid)
            if guard is not None:
                guard.commit_tokens([tok])
        return next_tokens
