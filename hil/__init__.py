"""Standalone Mini-LaWAM human-in-the-loop collection and correction training."""

from .constants import (
    ARM_CORRECTION_INDICES,
    ARM_CORRECTION_MEANING,
    GRIPPER_CLOSE,
    GRIPPER_LABEL_MEANING,
    GRIPPER_OPEN,
)

__all__ = [
    "ARM_CORRECTION_INDICES",
    "ARM_CORRECTION_MEANING",
    "GRIPPER_LABEL_MEANING",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSE",
]
