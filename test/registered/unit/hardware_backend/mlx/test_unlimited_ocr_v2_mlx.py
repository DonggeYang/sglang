"""Unlimited-OCR v2 MLX port vs the PyTorch reference math of the CUDA backend.

Checks the pieces that replace Triton kernels (stage-0 summaries, entmax router,
page2d RoPE tables) and the CPU page state against the torch specs in
``layers/attention/unlimited_dash_ops`` and ``models/unlimited_ocr_v2_support/document_reuse``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")

# Full fp32 matmuls, as load_language_model sets up (read at the first matmul).
os.environ.setdefault("MLX_ENABLE_TF32", "0")
_HAS_MLX = importlib.util.find_spec("mlx") is not None

if _HAS_MLX:
    import mlx.core as mx
    import torch

    from sglang.srt.hardware_backend.mlx.models import unlimited_ocr_v2_mlx as U
    from sglang.srt.hardware_backend.mlx.uocr_v2_doc_reuse import MlxPrefixSeed
    from sglang.srt.layers.attention.unlimited_dash_ops import dash_stage0
    from sglang.srt.layers.attention.unlimited_dash_ops.page2d_rope import PageRope2D
    from sglang.srt.models.unlimited_ocr_v2_support.document_reuse import (
        replay_coords,
        replay_page_isolation,
    )
    from sglang.test.kernels.unlimited_ocr_dash import (
        ACT_CLAMP,
        SCORE_NEG,
        entmax_route_reference,
    )

PAGE, IMG = 7, 9

_SPARSE_CFG = {
    "dash_stage0": "head_cls",
    "sparse_main_rope_mode": "page2d",
    "sparse_rope_mode": "page2d",
    "sparse_block_size": 4,
    "sparse_local_block": 2,
    "sparse_index_dim": 8,
    "sparse_topk_blocks": 6,
    "dash_topk_blocks": 3,
    "dash_infer_topm": 5,
    "dash_scaling_factor": 8.0,
    "dash_prior_floor": 1e-4,
    "dash_sigma": 1.0,
    "dash_image_only": True,
    "image_bucket_constraint": True,
    "image_token_id": IMG,
    "page_token_id": PAGE,
    "per_page_image_tokens": 3,
    "sparse_rope_tok_theta": 10000.0,
    "sparse_rope_page_theta": 256.0,
    "sparse_rope_page_pairs": 1,
    "sparse_rope_page_mod": 2,
    "sparse_rope_tok_freq_mode": "resample",
    "sparse_main_rope_tok_freq_mode": "resample",
}


def _np(x) -> np.ndarray:
    return np.array(x.astype(mx.float32)) if isinstance(x, mx.array) else x.numpy()


def _hf_config(args, quantization: dict) -> dict:
    """A config.json for a tiny checkpoint of ``args``."""
    fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "first_k_dense_replace",
        "moe_layer_freq",
        "norm_topk_prob",
        "routed_scaling_factor",
        "vocab_size",
        "rms_norm_eps",
        "eos_token_id",
    )
    return {
        **{name: getattr(args, name) for name in fields},
        "sparse_attention_config": {**_SPARSE_CFG, "training_stage": "dash"},
        "quantization": quantization,
    }


def _attention(dash) -> U.DashAttention:
    """A tiny layer-0 DashAttention; the router/summary math reads ``dash``."""
    args = U.ModelArgs(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        moe_intermediate_size=8,
        n_routed_experts=0,
        n_shared_experts=0,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        vocab_size=32,
        rms_norm_eps=1e-6,
        eos_token_id=1,
        dash=dash,
    )
    return U.DashAttention(args, 0)


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestDashMath(CustomTestCase):
    def setUp(self):
        self.dash = U.DashConfig.from_sparse_config(_SPARSE_CFG, num_layers=1)
        self.rng = np.random.default_rng(0)

    def test_block_summaries_match_stage0(self):
        bs, d, total = self.dash.block_size, self.dash.index_dim, 14
        keys = self.rng.normal(size=(total, d)).astype(np.float32)
        head_cls = self.rng.normal(size=(d,)).astype(np.float32)
        attn = _attention(self.dash)
        attn.head_cls = mx.array(head_cls)
        nb = math.ceil(total / bs)
        padded = np.zeros((nb * bs, d), np.float32)
        padded[:total] = keys
        valid = torch.arange(nb * bs).view(nb, bs) < total
        ref = dash_stage0._summary_keys(
            torch.from_numpy(padded).view(nb, bs, d),
            valid,
            torch.from_numpy(head_cls),
            1 / math.sqrt(d),
        ).numpy()
        for first in (0, 2):
            got = _np(attn._block_summaries(mx.array(keys), first, total))
            np.testing.assert_allclose(got, ref[first:], atol=1e-5)

    def test_route_matches_reference(self):
        d, nb, T = self.dash.index_dim, 10, 12
        # A top-M window wider than the candidates: the reference has none.
        dash = U.DashConfig.from_sparse_config(
            {**_SPARSE_CFG, "dash_infer_topm": 16}, num_layers=1
        )
        idx_q = self.rng.normal(size=(T, d)).astype(np.float32)
        kbar = self.rng.normal(size=(nb, d)).astype(np.float32)
        q_pos = np.arange(28, 28 + T)
        allow = self.rng.random((T, nb)) > 0.4
        allow[0] = False  # a row whose image filter empties the candidates
        attn = _attention(dash)
        order, valid, slot_bias = attn._route(
            mx.array(idx_q), mx.array(kbar), mx.array(q_pos), mx.array(allow)
        )
        bias, selected = attn._densify(order, valid, slot_bias, nb)

        z = np.clip(idx_q @ kbar.T / math.sqrt(d), -ACT_CLAMP, ACT_CLAMP)
        remote = np.arange(nb)[None] <= (
            q_pos[:, None] // dash.block_size - dash.local_blocks
        )
        scores = np.where(remote, z, SCORE_NEG).astype(np.float32)
        idx, ref_bias = entmax_route_reference(
            scores=torch.from_numpy(scores),
            image_allow=torch.from_numpy(allow),
            idx_scale=dash.idx_scale[0],
            prior_floor=dash.prior_floor[0],
            sigma=dash.sigma,
            budget=dash.budget_remote,
            out_slots=nb,
        )
        want_sel = np.zeros((T, nb), bool)
        want_bias = np.zeros((T, nb), np.float32)
        for t in range(T):
            for j, b in zip(idx[t].tolist(), ref_bias[t].tolist()):
                if j >= 0:
                    want_sel[t, j], want_bias[t, j] = True, b
        np.testing.assert_array_equal(np.array(selected), want_sel)
        np.testing.assert_allclose(_np(bias), want_bias, atol=1e-5)

    def test_rope_tables_match_torch(self):
        tok = self.rng.integers(0, 3000, size=32)
        page = self.rng.integers(0, 9, size=32)
        ref = PageRope2D(16, 10000.0, 256.0, 1, "resample", page_mod=2).cos_sin(
            torch.from_numpy(tok), torch.from_numpy(page), torch.float32
        )
        got = U.PageRope2D(16, self.dash, "resample").cos_sin(tok, page, mx.float32)
        for g, r in zip(got, ref):
            np.testing.assert_allclose(_np(g), r.numpy(), atol=1e-5)


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestPageState(CustomTestCase):
    # bos, two prompt pages (marker + image tokens), instruction, then a
    # response with its own markers (generated from gen_start on).
    PROMPT = [0, PAGE, IMG, IMG, IMG, IMG, PAGE, IMG, IMG, 5, 6]
    RESPONSE = [PAGE, 11, 12, 13, PAGE, 14, 15, PAGE, 16]

    def _expected(self, ids, gen_start):
        img_page, _, _, _ = replay_page_isolation(ids, PAGE, IMG, per_page=3)
        pages, _ = replay_coords(ids, gen_start, PAGE, IMG, max_pages=16)
        cnt, tok = {}, []
        for p in pages:
            tok.append(cnt.get(p, 0))
            cnt[p] = tok[-1] + 1
        return img_page, pages, tok

    def test_matches_sequential_replay_in_any_chunking(self):
        dash = U.DashConfig.from_sparse_config(_SPARSE_CFG, num_layers=1)
        ids = self.PROMPT + self.RESPONSE
        gen_start = len(self.PROMPT)
        img_page, pages, tok = self._expected(ids, gen_start)
        for cuts in (
            [len(ids)],
            [3, 8, len(ids)],
            [gen_start] + list(range(gen_start + 1, len(ids) + 1)),
        ):
            state, start, got_page, got_tok = U.PageState(dash, 16), 0, [], []
            for end in cuts:
                _, t, p = state.step(ids[start:end], gen_start)
                got_tok += t.tolist()
                got_page += p.tolist()
                start = end
            self.assertEqual(state.slot_img_page.tolist(), img_page)
            self.assertEqual(got_page, pages)
            self.assertEqual(got_tok, tok)

    def test_query_pages(self):
        dash = U.DashConfig.from_sparse_config(_SPARSE_CFG, num_layers=1)
        ids = self.PROMPT + self.RESPONSE
        q_page, _, _ = U.PageState(dash, 16).step(ids, len(self.PROMPT))
        # bos: -1; a prompt marker opens its page; images: their page; the
        # instruction after the last image page: -1; response pages restart at
        # 0 and, unlike the page2d coordinates, are not clamped to pp.
        self.assertEqual(
            q_page.tolist(),
            [-1, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2],
        )

    def test_lazy_page_state_matches_host(self):
        dash = U.DashConfig.from_sparse_config(_SPARSE_CFG, num_layers=1)
        gen_start = len(self.PROMPT)
        host = U.PageState(dash, 16)
        host.step(self.PROMPT, gen_start)
        lazy = U.LazyPageState(host, dash, max_blocks=64)
        pos = gen_start
        for t in self.RESPONSE:
            nb = pos // dash.block_size + 1
            q_l, tok_l, page_l, allow_l = lazy.step(mx.array([t], dtype=mx.int32), nb)
            q_h, tok_h, page_h = host.step([t], gen_start)
            allow_h = host.block_allow(q_h, nb, dash.block_size)[0]
            self.assertEqual(int(q_l.item()), int(q_h[0]))
            self.assertEqual(int(tok_l.item()), int(tok_h[0]))
            self.assertEqual(int(page_l.item()), int(page_h[0]))
            self.assertEqual(np.array(allow_l).tolist(), allow_h.tolist())
            pos += 1


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestDecodePaths(CustomTestCase):
    """The fused one-query decode kernels and the batched decode step must
    compute what the dense prefill path computes."""

    def setUp(self):
        mx.random.seed(0)
        dash = U.DashConfig.from_sparse_config(_SPARSE_CFG, num_layers=2)
        args = U.ModelArgs(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=48,
            moe_intermediate_size=16,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
            vocab_size=40,
            rms_norm_eps=1e-6,
            eos_token_id=1,
            dash=dash,
        )
        self.model = U.UnlimitedOCRV2LanguageModel(args)
        for layer in self.model.layers:
            layer.self_attn.head_cls = mx.random.normal((dash.index_dim,))
        mx.eval(self.model.parameters())
        rng = np.random.default_rng(0)
        self.prompt = (
            [0, PAGE]
            + [IMG] * 20
            + [PAGE]
            + [IMG] * 20
            + rng.integers(10, 40, 6).tolist()
        )
        self.steps = rng.integers(10, 40, 30).tolist()

    def _dense_logits(self, ids, prompt_len):
        cache = self.model.make_cache(prompt_len, 4096)
        return self.model(ids, cache)

    def test_decode_matches_dense_prefill(self):
        cache = self.model.make_cache(len(self.prompt), 4096)
        self.model(self.prompt, cache)
        seen = list(self.prompt)
        for t in self.steps:
            seen.append(t)
            got = self.model([t], cache)
            want = self._dense_logits(seen, len(self.prompt))
            # Same routing on both sides; the dense oracle rounds q/k/v and the
            # bias to fp16 while the decode kernels stay fp32 (max error here
            # ~1.3e-3). A dropped remote set moves these logits by ~2.
            np.testing.assert_allclose(_np(got), _np(want), atol=5e-3, rtol=0)

    def test_batched_decode_matches_single(self):
        prompts = [self.prompt, self.prompt[:-3] + [12, 13]]
        singles = [self.model.make_cache(len(p), 4096) for p in prompts]
        batched = [self.model.make_cache(len(p), 4096) for p in prompts]
        for p, c1, c2 in zip(prompts, singles, batched):
            self.model(p, c1)
            self.model(p, c2)
        for t in self.steps[:10]:
            toks = [t, (t + 7) % 40]
            want = mx.concatenate([self.model([x], c) for x, c in zip(toks, singles)])
            got = self.model.decode_batch(toks, batched)
            np.testing.assert_allclose(_np(got), _np(want), atol=1e-3, rtol=0)

    def test_batched_kernels_span_launch_groups(self):
        """24 requests (two 12-request launches), host and lazy metadata, with
        distinct text tails and contexts from under 8 blocks up."""
        from sglang.srt.hardware_backend.mlx.models import unlimited_ocr_v2_kernels as K

        rng = np.random.default_rng(1)
        cut = [8 + (40 * i) // 23 for i in range(24)]
        prompts = [self.prompt[: n - 2] + rng.integers(10, 40, 2).tolist() for n in cut]
        self.assertEqual(K._groups(len(prompts)), [(0, 12), (12, 24)])
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                singles = [self.model.make_cache(len(p), 4096) for p in prompts]
                batched = [self.model.make_cache(len(p), 4096) for p in prompts]
                for p, c1, c2 in zip(prompts, singles, batched):
                    self.model(p, c1)
                    self.model(p, c2)
                for t in self.steps[:6]:
                    # Text tokens only (10..39): no page marker or eos.
                    toks = [10 + (t + 3 * b) % 30 for b in range(len(prompts))]
                    want = mx.concatenate(
                        [self.model([x], c) for x, c in zip(toks, singles)]
                    )
                    ids = mx.array(toks, dtype=mx.int32) if lazy else toks
                    got = self.model.decode_batch(ids, batched)
                    if lazy:
                        for c, x in zip(batched, toks):
                            self.model.commit(c, [x])
                    np.testing.assert_allclose(_np(got), _np(want), atol=1e-3, rtol=0)

    def test_launch_groups_split_evenly(self):
        from sglang.srt.hardware_backend.mlx.models import unlimited_ocr_v2_kernels as K

        for n in (1, 12, 13, 25):
            groups = K._groups(n)
            sizes = [hi - lo for lo, hi in groups]
            self.assertEqual(groups[0][0], 0)
            self.assertEqual(groups[-1][1], n)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(groups, groups[1:])))
            self.assertLessEqual(max(sizes), K.MAX_GROUP)
            self.assertEqual(len(groups), (n + K.MAX_GROUP - 1) // K.MAX_GROUP)

    def test_mixed_fused_and_unfused_batch(self):
        """A batch with a request past MAX_ROUTE_BLOCKS takes the per-request
        path, whose gathered decode attention must match the fused kernels."""
        from unittest import mock

        prompts = [self.prompt[:20], self.prompt]  # 5 and 13 blocks of 4
        fused = [self.model.make_cache(len(p), 4096) for p in prompts]
        mixed = [self.model.make_cache(len(p), 4096) for p in prompts]
        for p, c1, c2 in zip(prompts, fused, mixed):
            self.model(p, c1)
            self.model(p, c2)
        for t in self.steps[:6]:
            toks = [t, 10 + t % 30]
            want = self.model.decode_batch(toks, fused)
            with mock.patch.object(U, "MAX_ROUTE_BLOCKS", 10):
                got = self.model.decode_batch(toks, mixed)
            np.testing.assert_allclose(_np(got), _np(want), atol=1e-3, rtol=0)

    def test_lazy_decode_matches_host_decode(self):
        """Device page state on unevaluated tokens == host page state."""
        steps = [PAGE, 11, 12, PAGE, 13, 14, PAGE, 15] + self.steps[:8]
        host = self.model.make_cache(len(self.prompt), 4096)
        lazy = self.model.make_cache(len(self.prompt), 4096)
        self.model(self.prompt, host)
        self.model(self.prompt, lazy)
        for t in steps:
            want = self.model([t], host)
            got = self.model.decode_batch(mx.array([t], dtype=mx.int32), [lazy])
            self.model.commit(lazy, [t])
            np.testing.assert_allclose(_np(got), _np(want), atol=1e-3, rtol=0)
        self.assertEqual(lazy.page_state.cnt, host.page_state.cnt)

    def test_fused_projections_and_prequantized_load(self):
        """Fusing Q/K/V and index_q/index_k (both bf16 and q8) preserves logits,
        and saving + reloading a pre-quantized checkpoint yields identical output."""
        import json
        import tempfile

        from mlx.utils import tree_flatten, tree_map

        c_unfused = self.model.make_cache(len(self.prompt), 4096)
        want_bf16 = _np(self.model(self.prompt, c_unfused))
        self.model.fuse_projections()
        c_fused = self.model.make_cache(len(self.prompt), 4096)
        got_bf16 = _np(self.model(self.prompt, c_fused))
        np.testing.assert_allclose(got_bf16, want_bf16, atol=1e-4, rtol=0)

        # Build a tiny pre-quantized checkpoint in a temp dir and reload via load_language_model
        dash = U.DashConfig.from_sparse_config(_SPARSE_CFG, num_layers=2)
        args = U.ModelArgs(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=64,
            moe_intermediate_size=32,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
            vocab_size=40,
            rms_norm_eps=1e-6,
            eos_token_id=1,
            dash=dash,
        )
        q_model = U.UnlimitedOCRV2LanguageModel(args)
        for layer in q_model.layers:
            layer.self_attn.head_cls = mx.random.normal((dash.index_dim,))
        q_model.update(tree_map(lambda p: p.astype(mx.bfloat16), q_model.parameters()))
        mx.eval(q_model.parameters())
        U.quantize_language_model(q_model, bits=8, group_size=32, mode="full")
        c_q = q_model.make_cache(len(self.prompt), 4096)
        want_q = _np(q_model(self.prompt, c_q))

        with tempfile.TemporaryDirectory() as tmpdir:
            mx.save_safetensors(
                os.path.join(tmpdir, "model.safetensors"),
                dict(tree_flatten(q_model.parameters())),
            )
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(
                    _hf_config(args, {"bits": 8, "group_size": 32, "mode": "full"}), f
                )
            reloaded, _, _ = U.load_language_model(tmpdir)
            c_rel = reloaded.make_cache(len(self.prompt), 4096)
            got_q = _np(reloaded(self.prompt, c_rel))
            np.testing.assert_allclose(got_q, want_q, atol=1e-5, rtol=0)

    def test_mlx_vlm_checkpoint_loads(self):
        """An mlx-vlm export (``language_model.`` keys, unfused Q/K/V, named
        ``dash_router``, affine mode) loads to the model it was saved from."""
        for bits in (4, 6):
            with self.subTest(bits=bits):
                self._check_mlx_vlm_load(bits)

    def _check_mlx_vlm_load(self, bits: int):
        import json
        import tempfile

        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_map

        # Its own weights, whatever the other tests drew from the generator.
        mx.random.seed(0)
        args = U.ModelArgs(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=64,
            moe_intermediate_size=32,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
            vocab_size=40,
            rms_norm_eps=1e-6,
            eos_token_id=1,
            dash=self.model.args.dash,
        )
        model = U.UnlimitedOCRV2LanguageModel(args)
        for layer in model.layers:
            layer.self_attn.head_cls = mx.random.normal((args.dash.index_dim,))
        # The loader keeps unquantized weights in bf16.
        model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
        nn.quantize(
            model,
            group_size=32,
            bits=bits,
            class_predicate=lambda p, m: U._quantize_predicate(p, m),
        )
        mx.eval(model.parameters())
        want = _np(model(self.prompt, model.make_cache(len(self.prompt), 4096)))

        renames = (
            (".self_attn.index_q_proj.", ".self_attn.dash_router.q_proj."),
            (".self_attn.index_k_proj.", ".self_attn.dash_router.k_proj."),
            (".self_attn.index_q_norm.", ".self_attn.dash_router.q_norm."),
            (".self_attn.index_k_norm.", ".self_attn.dash_router.k_norm."),
            (".self_attn.head_cls", ".self_attn.dash_router.head_cls"),
        )
        state = {}
        for key, value in tree_flatten(model.parameters()):
            for src, dst in renames:
                key = key.replace(src, dst)
            root = (
                "language_model."
                if key.startswith("lm_head")
                else "language_model.model."
            )
            state[root + key] = value
        self.assertTrue(any(".self_attn.q_proj.scales" in k for k in state))

        with tempfile.TemporaryDirectory() as tmpdir:
            mx.save_safetensors(os.path.join(tmpdir, "model.safetensors"), state)
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(
                    _hf_config(
                        args, {"bits": bits, "group_size": 32, "mode": "affine"}
                    ),
                    f,
                )
            reloaded, _, _ = U.load_language_model(tmpdir)
            got = _np(
                reloaded(self.prompt, reloaded.make_cache(len(self.prompt), 4096))
            )
        np.testing.assert_allclose(got, want, atol=1e-4, rtol=0)

    def test_checkpoint_keys_map_mlx_vlm_layout(self):
        cases = {
            "language_model.model.layers.0.mlp.gate.weight": "model.layers.0.mlp.gate.weight",
            "language_model.lm_head.scales": "lm_head.scales",
            "vision_model.transformer.layers.0.mlp.fc1.bias": "model.vision_model.transformer.layers.0.mlp.fc1.bias",
            "sam_model.neck.0.weight": "model.sam_model.neck.0.weight",
            "projector.layers.biases": "model.projector.layers.biases",
            "image_newline": "model.image_newline",
            "view_separator": "model.view_seperator",
            # HF and pre-quantized checkpoints keep their keys.
            "model.layers.0.self_attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.qkv_proj.scales": "layers.0.self_attn.qkv_proj.scales",
            "model.view_seperator": "model.view_seperator",
        }
        for key, want in cases.items():
            self.assertEqual(U.checkpoint_key(key), want)

    def test_cache_state_arrays_leaves_no_garbage(self):
        """Collecting the arrays to evaluate creates no reference cycle.

        A cycle would keep every request cache's buffers alive until Python's
        cyclic GC runs: memory grew ~3 MB per page over a 598-page document.
        """
        import gc

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        cache = self.model.make_cache(len(self.prompt), 4096)
        self.model(self.prompt, cache)
        gc.collect()
        gc.disable()
        try:
            arrays = MlxModelRunner.cache_state_arrays([[cache]])
            self.assertEqual(len(arrays), len(cache.state))
            del arrays
            self.assertEqual(gc.collect(), 0)
        finally:
            gc.enable()

    def test_window_page_base_shifts_rope_pages(self):
        """A window's page base moves pages > 0 by ``base % page_mod`` (CUDA parity)."""
        plain = self.model.make_cache(len(self.prompt), 4096)
        odd = self.model.make_cache(len(self.prompt), 4096, page_base=3)
        even = self.model.make_cache(len(self.prompt), 4096, page_base=4)
        self.assertEqual((plain.page_shift, odd.page_shift, even.page_shift), (0, 1, 0))
        want = _np(self.model(self.prompt, plain))
        np.testing.assert_array_equal(_np(self.model(self.prompt, even)), want)
        self.assertFalse(np.allclose(_np(self.model(self.prompt, odd)), want))

    def test_window_reuse_copies_tail_and_splices_text(self):
        """Retained rows land in the follow-up's cache; the splice resumes past them."""
        from sglang.srt.hardware_backend.mlx import uocr_v2_runner as R

        runner = object.__new__(R.MlxUocrV2ModelRunner)
        runner.model, runner.args, runner._context_len = (
            self.model,
            self.model.args,
            4096,
        )
        runner._req_caches, runner._req_token_ids = {}, {}
        runner._req_guard, runner._pending_splice = {}, {}

        old = self.model.make_cache(len(self.prompt), 4096)
        self.model(self.prompt, old)
        for t in self.steps[:6]:
            self.model([t], old)
        runner._req_caches["old"] = [old]
        ranges = [(1, 2), (20, old.offset)]
        rows = runner.snapshot_rows("old", ranges)
        self.assertEqual(rows.positions.tolist(), [1] + list(range(20, old.offset)))

        # Rewritten prompt: the old marker + image tail as its head, then a new page.
        src = [1] + list(range(20, len(self.prompt) - 6))
        head = [PAGE] + self.prompt[20 : len(self.prompt) - 6]
        new_prompt = head + [PAGE] + [IMG] * 5
        new = self.model.make_cache(len(new_prompt), 4096)
        runner._seed_reused_prefix(
            new,
            MlxPrefixSeed(rows=rows, src_positions=src, copy_len=len(head)),
            new_prompt,
        )
        self.assertEqual(new.offset, len(head))
        for lo, ln in zip(old.layers, new.layers):
            np.testing.assert_array_equal(
                _np(ln.keys.buf[: len(head)]), _np(lo.keys.buf[mx.array(src)])
            )
            self.assertEqual(ln.kbar_tokens, 0)
        replay = U.PageState(self.model.args.dash, new.page_state.max_pages)
        replay.step(head, len(new_prompt))
        self.assertEqual(
            new.page_state.slot_img_page.tolist(), replay.slot_img_page.tolist()
        )

        self.model(new_prompt[len(head) :], new)
        runner._req_caches["new"] = [new]
        runner._req_token_ids["new"] = list(new_prompt) + [11]
        text = list(range(len(self.prompt), old.offset))
        forced = self.steps[:5] + [PAGE]
        marks = new.page_state.mc
        backend = R.MlxDocReuseBackend(runner)
        # The prefill is already finalized here, so the splice applies at once.
        backend.queue_splice("new", rows, text, forced, len(new_prompt))
        self.assertEqual(new.offset, len(new_prompt) + len(text))
        self.assertEqual(runner._req_token_ids["new"], list(new_prompt) + forced)
        self.assertEqual(new.page_state.mc, marks + 1)
        self.assertEqual(new.page_state.num_tokens, new.offset)
        self.assertFalse(runner._pending_splice)
        # Decoding the forced marker works on the spliced cache.
        self.assertEqual(self.model([PAGE], new).shape, (1, self.model.args.vocab_size))

    def test_retained_window_holds_its_tails_once(self):
        """A retained MLX window copies its tail rows and keeps the same ranges
        of its stub row (so pool pressure can evict it); releasing it returns
        those ranges and the request row exactly once."""
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.mlx import uocr_v2_runner as R
        from sglang.srt.hardware_backend.mlx.uocr_v2_doc_reuse import MlxDocReuse
        from sglang.srt.managers.schedule_batch import ReqKvInfo
        from sglang.srt.models.unlimited_ocr_v2_support import document_reuse as D
        from sglang.srt.models.unlimited_ocr_v2_support.retained_window import (
            RetainedWindow,
        )

        runner = object.__new__(R.MlxUocrV2ModelRunner)
        runner.model, runner.args, runner._context_len = (
            self.model,
            self.model.args,
            4096,
        )
        runner._req_token_ids = {}
        old = self.model.make_cache(len(self.prompt), 4096)
        self.model(self.prompt, old)
        out = [PAGE] + self.steps[:6]
        for t in out:
            self.model([t], old)
        runner._req_caches = {"old": [old]}

        freed, released = [], []
        cache = SimpleNamespace(
            page_size=1,
            free_kv_row=lambda kv, ranges: freed.extend(ranges),
            req_to_token_pool=SimpleNamespace(free=released.append),
        )
        reuse = MlxDocReuse(
            cache=cache,
            backend=R.MlxDocReuseBackend(runner),
            on_window_finished=lambda req: None,
        )
        holder = RetainedWindow(
            kv=ReqKvInfo(req_pool_idx=3, kv_allocated_len=old.offset)
        )
        entry = D.DocEntry(
            holder=holder, input_ids=list(self.prompt), output_ids=out, next_page=2
        )
        req = SimpleNamespace(rid="old")
        self.assertTrue(reuse._keep_tails(entry, req, old.offset))
        rows = reuse._rows[id(entry)]
        kept = [int(p) for p in rows.positions]
        held = [p for lo, hi in entry.kept for p in range(lo, hi)]
        self.assertEqual(held, kept)
        gaps = sorted(p for lo, hi in freed for p in range(lo, hi))
        self.assertEqual(sorted(gaps + held), list(range(old.offset)))
        self.assertTrue(reuse._reusable(entry))

        freed.clear()
        reuse._free_entry(entry)
        reuse._free_entry(entry)
        self.assertEqual(sorted(p for lo, hi in freed for p in range(lo, hi)), held)
        self.assertEqual(released, [holder])
        self.assertFalse(reuse._reusable(entry))

    def test_window_reuse_through_prefill_in_either_splice_order(self):
        """prefill_start -> queue_splice -> prefill_finalize (overlap scheduler)
        ends in the same state as queueing the splice after finalize."""
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.mlx import uocr_v2_runner as R

        old = self.model.make_cache(len(self.prompt), 4096)
        self.model(self.prompt, old)
        for t in self.steps[:6]:
            self.model([t], old)
        ranges = [(1, 2), (20, old.offset)]
        src = [1] + list(range(20, len(self.prompt) - 6))
        head = [PAGE] + self.prompt[20 : len(self.prompt) - 6]
        new_prompt = head + [PAGE] + [IMG] * 5
        text = list(range(len(self.prompt), old.offset))
        forced = self.steps[:5] + [PAGE]

        def run(splice_before_finalize):
            runner = object.__new__(R.MlxUocrV2ModelRunner)
            runner.model, runner.args, runner._context_len = (
                self.model,
                self.model.args,
                4096,
            )
            runner._enable_sampling = False
            for name in (
                "_req_caches",
                "_req_token_ids",
                "_req_guard",
                "_pending_splice",
                "_req_sampling",
                "_req_images",
                "_req_pool_idx",
                "_req_synced_offset",
            ):
                setattr(runner, name, {})
            runner._req_caches["old"] = [old]
            rows = runner.snapshot_rows("old", ranges)
            backend = R.MlxDocReuseBackend(runner)
            runner.doc_reuse_backend = backend
            seeds = {
                "new": MlxPrefixSeed(rows=rows, src_positions=src, copy_len=len(head))
            }
            backend.attach_doc_reuse(SimpleNamespace(take_prefix=seeds.pop))
            req = SimpleNamespace(
                rid="new",
                uocr_doc=None,
                sampling_params=SimpleNamespace(custom_params=None),
                origin_input_ids=new_prompt,
                multimodal_inputs=None,
                full_untruncated_fill_ids=new_prompt,
                custom_logit_processor=None,
            )
            pending = runner.prefill_start(
                "new",
                new_prompt[len(head) :],
                new_prompt,
                list(range(len(head))),
                [],
                0,
                req=req,
            )
            self.assertFalse(seeds)
            mx.eval(pending.lazy_token)
            if splice_before_finalize:
                backend.queue_splice("new", rows, text, forced, len(new_prompt))
                self.assertIn("new", runner._pending_splice)
            runner.prefill_finalize(pending)
            if not splice_before_finalize:
                backend.queue_splice("new", rows, text, forced, len(new_prompt))
            self.assertFalse(runner._pending_splice)
            cache = runner._req_caches["new"][0]
            self.assertEqual(cache.offset, len(new_prompt) + len(text))
            self.assertEqual(cache.page_state.num_tokens, cache.offset)
            self.assertEqual(runner._req_token_ids["new"], list(new_prompt) + forced)
            return cache.page_state.mc, _np(self.model([PAGE], cache))

        overlap_mc, overlap_logits = run(True)
        plain_mc, plain_logits = run(False)
        self.assertEqual(overlap_mc, plain_mc)
        np.testing.assert_array_equal(overlap_logits, plain_logits)

    def test_item_features_uses_precomputed_embeddings(self):
        """Features encoded in the tokenizer process are used without the tower."""
        from sglang.srt.hardware_backend.mlx import uocr_v2_runner as R

        runner = object.__new__(R.MlxUocrV2ModelRunner)
        # No vision tower: touching it would raise, proving it is not used.
        runner._vision = None
        runner._vision_dir = None
        runner._vision_hf_config = None

        class _Item:
            def __init__(self, emb, n):
                self.precomputed_embeddings = emb
                self.offsets = [(0, n - 1)]

        e0 = (torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8) / 7).to(
            torch.bfloat16
        )
        e1 = torch.ones((2, 8), dtype=torch.bfloat16)
        feats, n_pre = runner._item_features([_Item(e0, 4), _Item(e1, 2)])
        self.assertEqual(n_pre, 2)
        self.assertEqual(feats.shape, (6, 8))
        np.testing.assert_array_equal(_np(feats[:4]), e0.float().numpy())
        np.testing.assert_array_equal(_np(feats[4:]), e1.float().numpy())


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestSortedMoE(CustomTestCase):
    """The sorted expert gather against mlx-lm's ``SwitchGLU`` (the tiny
    ``TestDecodePaths`` model is unquantized and never takes this branch)."""

    def test_matches_switch_glu_bit_for_bit(self):
        import mlx.nn as nn
        from mlx_lm.models.switch_layers import SwitchGLU

        for bits in (4, 6, 8):
            mx.random.seed(bits)
            moe = U.SortedSwitchGLU(64, 64, 8)
            nn.quantize(moe, group_size=32, bits=bits)
            ref = SwitchGLU(64, 64, 8)
            nn.quantize(ref, group_size=32, bits=bits)
            ref.update(moe.parameters())
            # (8, 6) is a decode batch of 8 under top-6: 48 pairs, below the
            # 64 at which mlx-lm starts sorting.
            for tokens, k in ((1, 6), (3, 2), (8, 6)):
                x = mx.random.normal((tokens, 64)).astype(mx.bfloat16)
                inds = mx.random.randint(0, 8, (tokens, k))
                with self.subTest(bits=bits, tokens=tokens, k=k):
                    # Same products; the sorted gather_qmm kernel may sum
                    # in another order (last-bit fp32 differences).
                    np.testing.assert_allclose(
                        _np(moe(x, inds)), _np(ref(x, inds)), rtol=1e-5, atol=1e-6
                    )


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestLogitGuard(CustomTestCase):
    """The in-graph MLX guard bans and forces what the torch
    ``DeepseekOCRNoRepeatNGramLogitProcessor`` does for the same history."""

    VOCAB, EOS = 40, 1
    PROMPT = [0, PAGE, IMG, IMG, PAGE, IMG, IMG, 5, 6, 12, 13, 14]

    def _masks(self, params: dict, output: list[int]):
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.mlx import uocr_v2_runner as R
        from sglang.srt.sampling.custom_logit_processor import (
            DeepseekOCRNoRepeatNGramLogitProcessor,
        )

        def req():
            return SimpleNamespace(
                origin_input_ids=list(self.PROMPT),
                output_ids=list(output),
                uocr_doc=None,
                page_guard_state=None,
                sampling_params=SimpleNamespace(custom_params=dict(params)),
            )

        logits = torch.zeros(1, self.VOCAB)
        DeepseekOCRNoRepeatNGramLogitProcessor()(logits, [{**params, "__req__": req()}])
        guard = R._MlxUocrGuardConfig(req(), self.VOCAB, list(self.PROMPT))
        guard.commit_tokens(list(output))
        row = guard.apply_fresh(
            mx.zeros((self.VOCAB,)), list(self.PROMPT) + list(output)
        )
        return np.isinf(_np(row)), torch.isinf(logits[0]).numpy()

    def test_matches_torch_processor(self):
        rng = np.random.default_rng(0)
        configs = [
            {"ngram_size": 3, "window_size": 16},
            {"ngram_size": 1, "window_size": 8, "ngram_output_only": True},
            {"page_token_id": PAGE, "eos_token_id": self.EOS, "max_page_tokens": 6},
            {
                "ngram_size": 2,
                "window_size": 12,
                "page_token_id": PAGE,
                "eos_token_id": self.EOS,
                "rep_min_pattern_size": 1,
                "rep_max_pattern_size": 3,
                "rep_min_count": 3,
            },
        ]
        for params in configs:
            for trial in range(25):
                # Small alphabets repeat n-grams, markers move the page count.
                out = rng.choice([PAGE, 10, 11, 12, 13], size=rng.integers(0, 30))
                with self.subTest(params=params, trial=trial):
                    got, want = self._masks(params, out.tolist())
                    np.testing.assert_array_equal(got, want)


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestSamEncoder(CustomTestCase):
    """MLX SAM ViT-B == the torch ``ImageEncoderViT`` (fp32, random weights).

    256 px exercises window padding (16 -> 28), the abs-pos resize (64 -> 16)
    and the rel-pos resize of the global blocks (127 -> 31).
    """

    def test_matches_torch_encoder(self):
        from sglang.srt.hardware_backend.mlx.models import unlimited_ocr_v2_sam as S
        from sglang.srt.models.deepseek_ocr import build_sam_vit_b

        torch.manual_seed(0)
        ref = build_sam_vit_b().float().eval()
        _randomize(ref)  # rel_pos tables are zero-initialised
        for sub in ref.modules():  # CUDA-only JIT kernels
            for flag in ("use_rel_pos_bias_kernel", "use_fused_window_unpartition_add"):
                if getattr(sub, flag, False):
                    setattr(sub, flag, False)
        state = {
            S._PREFIX + k: mx.array(v.detach().numpy())
            for k, v in ref.state_dict().items()
        }
        enc = S.SamImageEncoder()
        enc.load_weights(S.sam_weights_from_state(state, enc), strict=True)

        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            want = ref(x).permute(0, 2, 3, 1).numpy()
        got = _np(enc(S.nchw_to_mlx(x, dtype=None)))
        self.assertEqual(got.shape, (2, 4, 4, 1024))
        rel = np.linalg.norm(got - want) / np.linalg.norm(want)
        self.assertLess(rel, 1e-4)

        # mlx-vlm exports store the convs as OHWI already.
        ohwi = {
            k: v.transpose(0, 2, 3, 1)
            if v.ndim == 4 and k != S._PREFIX + "pos_embed"
            else v
            for k, v in state.items()
        }
        enc_ohwi = S.SamImageEncoder()
        enc_ohwi.load_weights(S.sam_weights_from_state(ohwi, enc_ohwi), strict=True)
        np.testing.assert_array_equal(_np(enc_ohwi(S.nchw_to_mlx(x, dtype=None))), got)


def _randomize(module: torch.nn.Module) -> None:
    for name, p in module.named_parameters():
        if name.endswith("weight") and p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.02)
        else:  # biases, norms, position tables
            torch.nn.init.normal_(p, mean=1.0 if "norm" in name else 0.0, std=0.1)


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestClipAndLayout(CustomTestCase):
    """MLX CLIP-L / projector / token layout == the torch vision code."""

    def test_quantized_projector_loads(self):
        """mlx-vlm quantizes the projector like the language tower."""
        import json
        import tempfile

        import mlx.nn as nn

        from sglang.srt.hardware_backend.mlx.models import unlimited_ocr_v2_clip as C

        mx.random.seed(0)
        proj = nn.Linear(128, 64, bias=True)
        qproj = nn.QuantizedLinear.from_linear(proj, group_size=64, bits=4)
        x = mx.random.normal((3, 128))
        with tempfile.TemporaryDirectory() as tmpdir:
            mx.save_safetensors(
                os.path.join(tmpdir, "model.safetensors"),
                {f"projector.layers.{k}": v for k, v in qproj.parameters().items()},
            )
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump({"quantization": {"bits": 4, "group_size": 64}}, f)
            loaded = C._load_projector(tmpdir, mx.float32)
        self.assertIsInstance(loaded, nn.QuantizedLinear)
        np.testing.assert_allclose(_np(loaded(x)), _np(qproj(x)), atol=1e-6)

    def test_clip_matches_torch(self):
        from sglang.srt.hardware_backend.mlx.models import unlimited_ocr_v2_clip as C
        from sglang.srt.models.deepseek_ocr import build_clip_l

        torch.manual_seed(0)
        ref = build_clip_l().float().eval()
        _randomize(ref)
        state = {k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()}
        clip = C.ClipVisionModel()
        clip.load_weights(C.clip_weights_from_state(state), strict=True)
        # A 10x10 grid (640 px tile) exercises the 16x16 -> 10x10 resize.
        patches = torch.randn(2, 1024, 10, 10)
        with torch.no_grad():
            want = ref(patches, patches).numpy()
        got = _np(
            clip(mx.array(patches.permute(0, 2, 3, 1).reshape(2, 100, 1024).numpy()))
        )
        self.assertEqual(got.shape, (2, 101, 1024))
        rel = np.linalg.norm(got - want) / np.linalg.norm(want)
        self.assertLess(rel, 1e-4)

    def test_token_layout_matches_torch(self):
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_vision import (
            _MlxVisionTower,
        )
        from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM as Ref

        newline = np.arange(8, dtype=np.float32) + 100
        tower = _MlxVisionTower.__new__(_MlxVisionTower)
        tower.image_newline = mx.array(newline)
        ref = SimpleNamespace(image_newline=torch.from_numpy(newline))
        local = np.random.default_rng(0).standard_normal((6, 9, 8)).astype(np.float32)
        want = Ref._format_ocr1_local_features(
            ref, torch.from_numpy(local), torch.tensor([3, 2])
        )
        got = tower._format_local(mx.array(local), 3, 2)
        np.testing.assert_array_equal(_np(got), want.numpy())
        glob_ = local[:1, :, :]
        want = Ref._format_ocr1_global_features(ref, torch.from_numpy(glob_))
        np.testing.assert_array_equal(
            _np(tower._format_global(mx.array(glob_))), want.numpy()
        )

    def test_gpu_preprocess_matches_pil(self):
        from PIL import Image

        from sglang.srt.configs.deepseek_ocr import (
            ImageTransform,
            pad_image,
            resize_image,
        )
        from sglang.srt.configs.unlimited_ocr import dynamic_preprocess
        from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_preprocess import (
            preprocess_page_mlx,
        )
        from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_sam import (
            nchw_to_mlx,
        )

        rng = np.random.default_rng(7)
        tfm = ImageTransform()
        for w, h in ((1000, 760), (480, 620), (640, 640)):
            arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            pil = Image.fromarray(arr)
            raw = mx.array(arr)
            for base_s, img_s, crop in (
                (1024, 640, True),
                (1024, 1024, False),
                (512, 512, False),
            ):
                im = pil
                if w <= 640 and h <= 640 or not crop:
                    crops_raw, ratio = [], (1, 1)
                else:
                    crops_raw, ratio = dynamic_preprocess(im, image_size=img_s)
                if img_s <= 640 and not crop:
                    im = resize_image(im, (img_s, img_s))
                pad_s = base_s if crop else img_s
                gv = pad_image(im, (pad_s, pad_s), (127, 127, 127))
                ref_g = nchw_to_mlx(
                    tfm(gv).unsqueeze(0).to(torch.bfloat16), mx.bfloat16
                )
                got_g, got_c, got_ratio = preprocess_page_mlx(
                    raw, cropping=crop, base_size=base_s, image_size=img_s
                )
                self.assertEqual(tuple(got_ratio), tuple(ratio))
                np.testing.assert_array_equal(_np(got_g), _np(ref_g))
                if ratio[0] > 1 or ratio[1] > 1:
                    ref_c = nchw_to_mlx(
                        torch.stack([tfm(c) for c in crops_raw]).to(torch.bfloat16),
                        mx.bfloat16,
                    )
                    np.testing.assert_array_equal(_np(got_c), _np(ref_c))
                else:
                    self.assertIsNone(got_c)


if __name__ == "__main__":
    unittest.main()
