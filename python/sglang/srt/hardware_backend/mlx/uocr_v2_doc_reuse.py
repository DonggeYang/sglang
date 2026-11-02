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
"""Unlimited-OCR v2 document-window KV reuse on MLX.

The scheduler keeps its bookkeeping on the stub pools, but each request's KV
lives in its own MLX cache (``uocr_v2_runner.py``). A finished window copies
the rows its follow-up reads out of that cache; the follow-up's cache starts
from them, and the overlap text is spliced in after its prefill. No device
edit is queued for the attention backend.
"""

from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import msgspec
import numpy as np

from sglang.srt.models.unlimited_ocr_v2_support.document_reuse import (
    DocEntry,
    DocReuse,
    TailCopyPlan,
    TextPlan,
    continuation_carry,
    hide_forced_text,
    req_state,
    tail_ranges,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.hardware_backend.mlx.uocr_v2_runner import (
        MlxDocReuseBackend,
        RetainedRows,
    )
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class MlxPrefixSeed(msgspec.Struct, frozen=True):
    """Retained rows a follow-up window's cache starts from."""

    rows: RetainedRows
    # Positions in ``rows`` copied to the new prompt head, in order.
    src_positions: List[int]
    copy_len: int


class MlxTextPlan(TextPlan, frozen=True):
    """``TextPlan`` with the retained rows the spliced text is read from."""

    rows: Optional[RetainedRows] = None
    text_positions: List[int] = []


class MlxDocReuse(DocReuse):
    """``DocReuse`` over the MLX runner's per-request caches."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # id(retained entry) -> its copied rows.
        self._rows: Dict[int, RetainedRows] = {}
        # rid -> the rows its first prefill seeds its cache with.
        self._seeds: Dict[str, MlxPrefixSeed] = {}

    @staticmethod
    def backend_for(attn_backend: MlxDocReuseBackend) -> MlxDocReuseBackend:
        return attn_backend

    def take_prefix(self, rid: str) -> Optional[MlxPrefixSeed]:
        """The rows to seed ``rid``'s new cache with, handed out once."""
        return self._seeds.pop(rid, None)

    def reset(self) -> None:
        super().reset()
        self._ops.clear()
        self._rows.clear()
        self._seeds.clear()

    def on_req_dropped(self, req: Req) -> None:
        super().on_req_dropped(req)
        self._seeds.pop(req.rid, None)

    def _free_entry(self, entry: DocEntry) -> None:
        self._rows.pop(id(entry), None)
        super()._free_entry(entry)

    # -- Finished windows --

    def _committed_len(self, req: Req) -> int:
        # Chained decode steps skip the stub pool's bookkeeping.
        return max(super()._committed_len(req), self._backend.committed_len(req.rid))

    def _keep_tails(self, entry: DocEntry, req: Req, committed: int) -> bool:
        """Copy the rows the follow-up reads out of the finished request's cache.

        The stub row keeps the same ranges, so retained windows hold what they
        hold on CUDA and pool pressure evicts them the same way.
        """
        b = self._backend
        kept = tail_ranges(
            entry.input_ids,
            entry.output_ids,
            True,
            b.page_state.page_token_id,
            b.page_state.image_token_id,
            # The tail starts on a DASH block, whatever the stub pool's page size.
            b.block_size,
            committed,
            lookahead=entry.lookahead,
        )
        rows = b.snapshot_rows(req.rid, kept) if kept else None
        kv = entry.holder.kv
        allocated = kv.kv_allocated_len
        held = (
            [(lo, min(hi, allocated)) for lo, hi in kept if lo < allocated]
            if rows is not None
            else []
        )
        gaps, pos = [], 0
        for lo, hi in held:
            if lo > pos:
                gaps.append((pos, lo))
            pos = hi
        if pos < allocated:
            gaps.append((pos, allocated))
        if gaps:
            self._cache.free_kv_row(kv, gaps)
        entry.kept = held
        if rows is None:
            return False
        self._rows[id(entry)] = rows
        logger.info(
            "[Unlimited-OCR v2] retained %d of %d window tokens (marker + tails)",
            rows.positions.shape[0],
            committed,
        )
        return True

    # -- Matching and the prefix copy --

    def _reusable(self, entry: DocEntry) -> bool:
        return id(entry) in self._rows

    def _covers(
        self, entry: DocEntry, plan: TailCopyPlan, text: Tuple[int, List[int]]
    ) -> bool:
        start = len(entry.input_ids) + text[0]
        needed = list(plan.src_positions) + list(range(start, start + len(text[1])))
        return bool(np.isin(needed, self._rows[id(entry)].positions).all())

    def _alloc_prefix_slots(self, copy_len: int, bs: int) -> Optional[torch.Tensor]:
        # The stub pool only accounts for tokens; it has no pages to fill.
        return super()._alloc_prefix_slots(copy_len, 1)

    def _queue_prefix_copy(
        self,
        req,
        *,
        entry: DocEntry,
        plan: TailCopyPlan,
        dst: torch.Tensor,
        text: Tuple[int, List[int]],
        forced: List[int],
    ) -> MlxTextPlan:
        """Hand the runner the rows to seed the follow-up's cache with.

        The runner copies them when it creates the cache at the first prefill
        and replays the page state over the copied prompt head.
        """
        b = self._backend
        rows = self._rows[id(entry)]
        self._seeds[req.rid] = MlxPrefixSeed(
            rows=rows, src_positions=list(plan.src_positions), copy_len=plan.copy_len
        )
        gen_coord, gen_iso = continuation_carry(
            plan.new_ids,
            b.page_state.page_token_id,
            b.page_state.image_token_id,
            b.page_state.per_page_image_tokens,
            b.coord_state.max_pages,
        )
        start = len(entry.input_ids) + text[0]
        positions = list(range(start, start + len(forced) - 1))
        return MlxTextPlan(
            forced=forced,
            src_slots=dst[:0],
            iso=gen_iso,
            coord=gen_coord,
            first_block=dst[:1] // b.block_size,
            rows=rows,
            text_positions=positions,
        )

    # -- After the follow-up's prefill --

    def _splice_text(self, req, seq_len: int, plan: MlxTextPlan) -> int:
        """Record the text tail in the stub row; the runner appends its rows.

        The runner feeds decode from its own token list, and its copies hold
        their own references to the rows, so the source is released at once.
        """
        row = req.kv.req_pool_idx
        req_to_token = self._cache.req_to_token_pool.req_to_token
        n = len(plan.text_positions)
        if n > 0:
            dst = self._cache.token_to_kv_pool_allocator.alloc(n)
            if dst is None:
                logger.info(
                    "[Unlimited-OCR v2] %s: no KV for text splice; marker only", req.rid
                )
                n = 0
            else:
                req_to_token[row, seq_len : seq_len + n] = dst.to(req_to_token.dtype)
        forced = plan.forced if n > 0 else plan.forced[-1:]
        self._backend.queue_splice(
            req.rid, plan.rows, plan.text_positions[:n], forced, seq_len
        )
        req.output_ids.extend(array("q", forced))
        st = req_state(req)
        st.forced_len = len(forced)
        # The copied text sits in output_ids, where the length check counts it.
        if req.sampling_params.max_new_tokens is not None:
            req.sampling_params.max_new_tokens += len(forced)
        st.skip_prefill_append = True
        hide_forced_text(req)
        req.kv.allocate_decode(n)
        self._release_source(req)
        return n
