"""Compatibility for Cosmos3-Edge on vLLM 0.27 + Transformers 5.15.

Generation needs the image/video compatibility pieces. DFlash deployment also
needs SupportsEagle3 and the 2x auxiliary-layer remap because vLLM splits each
Edge Hugging Face decoder block into separate attention and MLP layers.
"""

from __future__ import annotations

import logging

import transformers.models.qwen3_vl.video_processing_qwen3_vl as qwen3_vl_video
from transformers.image_utils import get_image_size
from transformers.video_processing_utils import BaseVideoProcessor

if not hasattr(qwen3_vl_video, "get_image_size"):
    qwen3_vl_video.get_image_size = get_image_size

from vllm.model_executor.models import interfaces as vllm_interfaces
from vllm.model_executor.models.cosmos3_edge import Cosmos3EdgeForConditionalGeneration
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.transformers_utils.configs.cosmos3_edge import Cosmos3EdgeConfig
from vllm.transformers_utils.processors.cosmos3_edge import Cosmos3EdgeVideoProcessor

if not hasattr(Cosmos3EdgeConfig, "image_token_index"):
    Cosmos3EdgeConfig.image_token_index = property(
        lambda self: getattr(self, "image_token_id", None)
    )


def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
    SupportsEagle3.set_aux_hidden_state_layers(self, layers)


def _get_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
    return SupportsEagle3.get_eagle3_default_aux_hidden_state_layers(self)


Cosmos3EdgeForConditionalGeneration.supports_eagle3 = True
Cosmos3EdgeForConditionalGeneration.set_aux_hidden_state_layers = (
    _set_aux_hidden_state_layers
)
Cosmos3EdgeForConditionalGeneration.get_eagle3_default_aux_hidden_state_layers = (
    _get_default_aux_hidden_state_layers
)

_original_supports_eagle3 = vllm_interfaces.supports_eagle3


def _supports_eagle3(model) -> bool:
    if "Cosmos3Edge" in type(model).__name__:
        return True
    return _original_supports_eagle3(model)


vllm_interfaces.supports_eagle3 = _supports_eagle3

import vllm.v1.worker.gpu_model_runner as gpu_model_runner

gpu_model_runner.supports_eagle3 = _supports_eagle3
_original_get_aux_layers = (
    gpu_model_runner.GPUModelRunner._get_eagle3_aux_layers_from_config
)
_logger = logging.getLogger("vllm")


def _is_edge_config(config) -> bool:
    if config is None:
        return False
    model_type = getattr(config, "model_type", "") or ""
    architectures = getattr(config, "architectures", None) or []
    return model_type == "cosmos3_edge" or any(
        "Cosmos3Edge" in str(architecture) for architecture in architectures
    )


def _runner_uses_edge(runner) -> bool:
    model_config = getattr(runner, "model_config", None)
    if model_config is None:
        vllm_config = getattr(runner, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None)
    return _is_edge_config(getattr(model_config, "hf_config", None))


def _get_aux_layers(self):
    layer_ids = _original_get_aux_layers(self)
    if not layer_ids or not _runner_uses_edge(self):
        return layer_ids
    remapped = tuple(2 * int(layer_id) for layer_id in layer_ids)
    _logger.info("Cosmos3-Edge DFlash aux layers: %s -> %s", layer_ids, remapped)
    return remapped


gpu_model_runner.GPUModelRunner._get_eagle3_aux_layers_from_config = _get_aux_layers

_original_video_resize = qwen3_vl_video.Qwen3VLVideoProcessor.resize


def _edge_video_resize(
    self,
    videos,
    size,
    resample=None,
    factor=None,
    temporal_factor=None,
    **kwargs,
):
    if getattr(size, "height", None) and getattr(size, "width", None):
        return BaseVideoProcessor.resize(
            self, image=videos, size=size, resample=resample, **kwargs
        )
    return _original_video_resize(
        self,
        videos,
        size,
        resample,
        factor,
        temporal_factor,
        **kwargs,
    )


Cosmos3EdgeVideoProcessor.resize = _edge_video_resize
