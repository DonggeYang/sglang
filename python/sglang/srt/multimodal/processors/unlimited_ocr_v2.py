"""Unlimited-OCR v2 processor with DASH image-mode semantics."""

import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Optional

import numpy as np
import torch
from PIL import Image

from sglang.srt.configs.deepseek_ocr import get_image_size
from sglang.srt.configs.unlimited_ocr import merge_mm_processor_kwargs
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.models.unlimited_ocr_v2 import UnlimitedOCRV2ForCausalLM
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.unlimited_ocr import (
    _DEFAULT_MODE,
    _IMAGE_MODE_PRESETS,
    UnlimitedOCRProcessor,
)

logger = logging.getLogger(__name__)

# Alias for mm_processor_kwargs={"crop_mode": bool} (vLLM-compatible spelling):
# True selects gundam, False selects base.
_CROP_MODE_ALIAS = {True: "gundam", False: "base"}


def _resolve_mode_v2(images_config) -> dict:
    """Resolve v2 image mode, including vLLM crop-mode aliases."""
    mode = _DEFAULT_MODE
    if images_config:
        mode = images_config.get("image_mode") or _DEFAULT_MODE
        crop = images_config.get("crop_mode")
        if crop is not None and "image_mode" not in images_config:
            mode = _CROP_MODE_ALIAS[bool(crop)]
    key = str(mode).strip().lower()
    if key == "auto":
        key = _DEFAULT_MODE
    preset = _IMAGE_MODE_PRESETS.get(key)
    if preset is None:
        raise ValueError(
            f"Unknown image_mode '{mode}'. "
            f"Supported: auto, {', '.join(_IMAGE_MODE_PRESETS)}"
        )
    return dict(zip(("base_size", "image_size", "crop_mode"), preset))


def _release_slice_storage(value):
    """Clone a tensor view that keeps a larger parent storage alive."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.untyped_storage().nbytes() <= value.numel() * value.element_size():
        return value
    return value.clone()


def _is_raw_page(image: Any) -> bool:
    """A decoded page the Metal preprocess reads as is: PIL or 3D ``uint8``."""
    if isinstance(image, Image.Image):
        return True
    return (
        isinstance(image, torch.Tensor)
        and image.ndim == 3
        and image.dtype == torch.uint8
    )


def _raw_hwc_uint8(image: Any) -> torch.Tensor:
    """An RGB page as a ``uint8 [H, W, 3]`` CPU tensor, not resized or cropped."""
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
    t = image.detach().cpu()
    if t.shape[0] == 3 and t.shape[-1] != 3:
        t = t.permute(1, 2, 0)
    return t.contiguous()


class UnlimitedOCRV2ProcessorOutput(MultimodalProcessorOutput):
    """Processor output whose crop tiles go over SHM, not inline in the request."""

    shm_model_specific_data: ClassVar[bool] = True


class UnlimitedOCRV2Processor(UnlimitedOCRProcessor):
    """Multimodal processor for UNLIMITED-OCR v2 (DASH) checkpoints."""

    models: ClassVar = [UnlimitedOCRV2ForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # MLX: pages are encoded here, on one thread that owns the vision tower,
        # so a window's vision encode does not stall the scheduler's decode.
        self._mlx_vision = None
        self._mlx_vision_executor = None
        # The scheduler's own M-RoPE helper instance (skip_mm_pool) never encodes.
        if use_mlx() and not kwargs.get("skip_mm_pool", False):
            self._mlx_vision_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="sglang-uocr-mlx-vision"
            )
            # Built while the scheduler loads the model, not at the first request.
            self._mlx_vision_executor.submit(self._mlx_vision_tower)

    def _mlx_vision_tower(self):
        """The MLX vision tower, built on first use (vision thread only)."""
        if self._mlx_vision is None:
            from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_vision import (
                UocrV2VisionTower,
            )
            from sglang.srt.hardware_backend.mlx.remote_code_gate import (
                resolve_model_directory,
            )

            model_dir = resolve_model_directory(
                self.server_args.model_path, revision=self.server_args.revision
            )
            self._mlx_vision = UocrV2VisionTower(str(model_dir), {})
        return self._mlx_vision

    def _process_mlx_raw(
        self, base_output, *, base_size: int, image_size: int, crop_mode: bool
    ) -> Optional[tuple[list[MultimodalDataItem], torch.Tensor]]:
        """Tokenize like ``format_messages_v2`` but keep each page as raw ``uint8``
        for the Metal preprocess; None when an input is not a plain image."""
        from sglang.srt.hardware_backend.mlx.models.unlimited_ocr_v2_preprocess import (
            select_crop_grid,
        )

        loaded = base_output.organize_results()
        if not loaded or not all(
            mod == Modality.IMAGE and _is_raw_page(item) for mod, item in loaded
        ):
            return None
        proc = self._processor
        images = [item for _, item in loaded]
        text_splits = base_output.input_text.split(proc.image_token)
        if len(text_splits) != len(images) + 1:
            raise ValueError(
                f"{len(text_splits) - 1} image tokens for {len(images)} images"
            )
        num_queries = math.ceil((image_size // proc.patch_size) / proc.downsample_ratio)
        num_queries_base = math.ceil(
            (base_size // proc.patch_size) / proc.downsample_ratio
        )

        tokens: list[int] = [proc.bos_id]
        mm_items: list[MultimodalDataItem] = []
        for text, image in zip(text_splits, images):
            tokens.extend(proc.encode(text, bos=False, eos=False))
            tiles_w, tiles_h = select_crop_grid(
                *get_image_size(image), cropping=crop_mode, image_size=image_size
            )
            has_local = tiles_w > 1 or tiles_h > 1
            if crop_mode:
                n_image = (num_queries_base + 1) * num_queries_base + 1
                if has_local:
                    n_image += (num_queries * tiles_w + 1) * (num_queries * tiles_h)
            else:
                n_image = (num_queries + 1) * num_queries + 1
            start = len(tokens)
            tokens.extend([proc.image_token_id] * n_image)
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=_raw_hwc_uint8(image),
                    offsets=[(start, len(tokens) - 1)],
                    model_specific_data={
                        "images_spatial_crop": torch.tensor(
                            [[[tiles_w, tiles_h]]], dtype=torch.long
                        ),
                        "has_local_crops": [has_local],
                        "uocr_mlx_raw_mode": (base_size, image_size, crop_mode),
                    },
                )
            )
        tokens.extend(proc.encode(text_splits[-1], bos=False, eos=False))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        input_ids[input_ids < 0] = proc.pad_id
        return mm_items, input_ids

    def _encode_on_mlx(self, mm_items) -> list[torch.Tensor]:
        """bf16 CPU features per item; runs on the vision thread."""
        import mlx.core as mx

        feats = np.array(
            self._mlx_vision_tower().encode_items(mm_items).astype(mx.float32)
        )
        out, pos = [], 0
        for item in mm_items:
            n = sum(e - s + 1 for s, e in item.offsets)
            out.append(torch.from_numpy(feats[pos : pos + n]).to(torch.bfloat16))
            pos += n
        return out

    async def _attach_mlx_embeddings(self, mm_items) -> None:
        """Ship encoded pages with the request; the scheduler encodes on failure."""
        loop = asyncio.get_running_loop()
        try:
            embeddings = await loop.run_in_executor(
                self._mlx_vision_executor, self._encode_on_mlx, list(mm_items)
            )
        except Exception:
            logger.exception(
                "Unlimited-OCR v2: MLX vision encode in the tokenizer failed; "
                "the scheduler encodes these pages at prefill"
            )
            return
        for item, embedding in zip(mm_items, embeddings):
            item.precomputed_embeddings = embedding
            item.feature = None

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: list[MultimodalDataItem],
        *,
        images: Optional[list[Any]],
    ) -> list[MultimodalDataItem]:
        mm_items = super()._postprocess_mm_items_before_transport(
            mm_items, images=images
        )
        # Each page's tensors are views of the [pages, ...] bundle, and pickling a
        # view serializes the whole bundle: N pages would ship N^2 pages of crops.
        for item in mm_items:
            item.model_specific_data = {
                key: _release_slice_storage(value)
                for key, value in item.model_specific_data.items()
            }
        return mm_items

    async def process_mm_data_async(
        self, image_data: list[str | bytes], input_text, *args, **kwargs
    ):
        """Process multimodal data asynchronously (v2 mode resolution)."""
        request_obj = kwargs.get("request_obj")
        images_config = (
            merge_mm_processor_kwargs(
                request_obj.images_config, request_obj.mm_processor_kwargs
            )
            if request_obj is not None
            else None
        )
        processor_kwargs = _resolve_mode_v2(images_config)

        prefix = images_config.get("prefix", "") if images_config else ""

        base_output = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )
        raw = (
            self._process_mlx_raw(base_output, **processor_kwargs)
            if self._mlx_vision_executor is not None
            else None
        )
        if raw is not None:
            mm_items, input_ids = raw
        else:
            mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
                base_output, self.mm_tokens, **processor_kwargs, v2_layout=True
            )

        if prefix:
            prefix_ids = self._tokenizer.encode(prefix, add_special_tokens=False)
            input_ids = torch.cat(
                [input_ids, torch.tensor(prefix_ids, dtype=input_ids.dtype)]
            )

        self._mix_config_into_hash(mm_items, processor_kwargs)
        if raw is not None and mm_items:
            await self._attach_mlx_embeddings(mm_items)

        return UnlimitedOCRV2ProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.mm_tokens.image_token_id,
        )
