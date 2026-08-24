"""Checkpoint helpers for Stage-1 LAM training and fine-tuning."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_pretrained_weights(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> int:
    """Load model weights without restoring any Lightning training state.

    Both released LAM artifacts and Lightning training checkpoints store model
    parameters under ``state_dict``. A raw state dictionary is accepted as a
    convenience, but optimizer, scheduler, epoch, loop, and callback state are
    deliberately ignored.

    Returns:
        Number of state-dictionary entries loaded.
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained LAM checkpoint not found: {path}")

    payload: Any = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected checkpoint mapping at {path}, got {type(payload).__name__}"
        )

    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Checkpoint {path} has invalid state_dict type "
            f"{type(state_dict).__name__}"
        )
    if not state_dict:
        raise ValueError(f"Checkpoint {path} contains an empty state_dict")
    non_tensor_keys = [key for key, value in state_dict.items() if not torch.is_tensor(value)]
    if non_tensor_keys:
        raise TypeError(
            f"Checkpoint {path} is not a model state dictionary; non-tensor "
            f"entries include {non_tensor_keys[:5]}"
        )

    # load_state_dict copies tensor values into the already-constructed model.
    # It does not alter requires_grad, so the DINO backbone remains frozen.
    module.load_state_dict(state_dict, strict=strict)
    return len(state_dict)
