# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import BoundArguments

import torch
from torch import nn

from .types import LayerReloadingInfo, LayerTensors

__all__ = [
    "get_layer_tensors",
    "get_layer_params_buffers",
    "get_layer_size",
    "get_tensor_load_numel",
    "has_device_tensors",
    "get_info_size",
    "register_fused_buffer",
]


def register_fused_buffer(
    module: nn.Module, name: str, value: torch.Tensor
) -> torch.Tensor:
    """Store a fused weight buffer as a non-persistent registered buffer.

    Drafters stack per-layer weights into fused buffers that are read through
    addresses baked into captured CUDA graphs. Registering the buffer
    non-persistently lets the layerwise reload capture and restore its storage,
    so the address survives a reload and the captured graph keeps reading live
    memory. Non-persistent keeps it out of the state dict: it is derived from
    per-layer weights and is not a checkpoint entry.

    Re-registering a name would replace the storage and defeat that purpose, so
    when ``name`` already holds a materialized tensor of matching shape, dtype
    and device the values are copied into it in place instead. A cold build has
    no prior tensor and registers ``value`` directly; a shape change or a
    meta-derived buffer (seen when the builder runs mid-reload) re-registers.

    Args:
        module: module that owns the fused buffer.
        name: buffer name.
        value: freshly built tensor.

    Returns:
        The tensor now registered at ``module.name``.

    """
    old = module._buffers.get(name)
    if (
        isinstance(old, torch.Tensor)
        and not old.is_meta
        and old.shape == value.shape
        and old.dtype == value.dtype
        and old.device == value.device
    ):
        old.copy_(value)
        return old
    if name in module.__dict__:
        # A plain attribute set before the buffer existed (e.g. the quantized
        # path assigns None in __init__) blocks register_buffer; drop it.
        delattr(module, name)
    module.register_buffer(name, value, persistent=False)
    return value


def get_layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get all parameters and buffers from a module as a dict."""
    params, buffers = get_layer_params_buffers(layer)
    return params | buffers


def get_layer_params_buffers(layer: torch.nn.Module) -> LayerTensors:
    """Get all parameters and buffers of a module as a tuple of dicts."""
    return (
        {name: param for name, param in layer._parameters.items() if param is not None},
        {name: buffer for name, buffer in layer._buffers.items() if buffer is not None},
    )


def get_tensor_load_numel(tensor: torch.Tensor) -> int:
    """Count checkpoint elements, excluding padding declared by the weight creator."""
    return getattr(tensor, "weight_loader_numel", tensor.numel())


def get_layer_size(layer: torch.nn.Module) -> int:
    """Calculate total number of elements across loadable tensors in a layer.

    Excludes SKIP_LOAD_TENSORS (e.g. _expert_map) which are never loaded via
    weight_loader during layerwise reload.
    """
    from .meta import SKIP_LOAD_TENSORS

    return sum(
        get_tensor_load_numel(tensor)
        for name, tensor in get_layer_tensors(layer).items()
        if name not in SKIP_LOAD_TENSORS
    )


def has_device_tensors(bound_args: BoundArguments) -> bool:
    """Return True if the loaded weights exist on an accelerator device.

    Args:
        bound_args: args to load weights

    Returns:
        True if weights are on accelerator device

    """
    return any(
        isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
        for value in bound_args.arguments.values()
    )


def get_info_size(info: LayerReloadingInfo) -> int:
    """Calculate the number of bytes used by loaded weights for a given layer.

    Args:
        info: layerwise info to get size of

    Returns:
        number of bytes used by loaded weights

    """
    return sum(
        value.nbytes
        for _, args in info.loaded_weights
        for value in args.arguments.values()
        if isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
    )
