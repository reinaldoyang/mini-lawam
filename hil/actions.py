"""Canonical action and pose helpers for the HIL collector.

All data written by :mod:`hil.collect_corrections` uses a seven-dimensional
forward-command action::

    [dx, dy, dz, dRx, dRy, dRz, gripper_state]

The six motion values move from the previous commanded TCP pose to the next
commanded TCP pose.  Rotation deltas are composed in SO(3), never obtained by
subtracting UR rotation-vector components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import ACTION_DIM


def _pose6(value: Sequence[float], name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must contain six finite values, got {pose!r}")
    return pose


def rotvec_pose_delta(start_pose: Sequence[float], end_pose: Sequence[float]) -> np.ndarray:
    """Return the base-frame pose delta from ``start_pose`` to ``end_pose``."""
    start = _pose6(start_pose, "start_pose")
    end = _pose6(end_pose, "end_pose")
    relative = Rotation.from_rotvec(end[3:6]) * Rotation.from_rotvec(start[3:6]).inv()
    return np.concatenate((end[:3] - start[:3], relative.as_rotvec())).astype(np.float64)


def action_from_target(
    start_pose: Sequence[float],
    target_pose: Sequence[float],
    gripper_state: float,
) -> np.ndarray:
    """Encode one commanded TCP transition as a canonical 7D action."""
    action = np.empty(ACTION_DIM, dtype=np.float32)
    action[:6] = rotvec_pose_delta(start_pose, target_pose).astype(np.float32)
    action[6] = float(gripper_state)
    if not np.all(np.isfinite(action)):
        raise ValueError(f"non-finite action generated: {action!r}")
    return action


def target_from_action(start_pose: Sequence[float], action: Sequence[float]) -> np.ndarray:
    """Apply one canonical 7D action and return its six-dimensional TCP target."""
    start = _pose6(start_pose, "start_pose")
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (ACTION_DIM,) or not np.all(np.isfinite(value)):
        raise ValueError(f"action must contain {ACTION_DIM} finite values, got {value!r}")
    relative = Rotation.from_rotvec(value[3:6])
    end_rotation = relative * Rotation.from_rotvec(start[3:6])
    return np.concatenate((start[:3] + value[:3], end_rotation.as_rotvec()))


def quat_wxyz_from_rotvec(rotvec: Sequence[float]) -> np.ndarray:
    """Convert a UR rotation vector to dataset quaternion order ``[w,x,y,z]``."""
    xyzw = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def compose_base_z_delta(rotvec: Sequence[float], rz_delta: float) -> np.ndarray:
    """Pre-multiply a TCP orientation by a rotation around robot-base Z."""
    current = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64))
    rz = Rotation.from_rotvec(np.asarray([0.0, 0.0, float(rz_delta)], dtype=np.float64))
    return (rz * current).as_rotvec()


def clamp_target_pose(
    requested_pose: Sequence[float],
    actual_pose: Sequence[float],
    workspace_min: Sequence[float],
    workspace_max: Sequence[float],
    max_target_lead: float,
) -> np.ndarray:
    """Clamp XYZ to the workspace and to a radius around the measured TCP."""
    requested = _pose6(requested_pose, "requested_pose").copy()
    actual = _pose6(actual_pose, "actual_pose")
    minimum = np.asarray(workspace_min, dtype=np.float64)
    maximum = np.asarray(workspace_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
        raise ValueError("workspace bounds must be finite 3-vectors with min < max")
    if np.any(actual[:3] < minimum) or np.any(actual[:3] > maximum):
        raise RuntimeError(
            "measured TCP is outside the configured workspace: "
            f"xyz={actual[:3]!r}, min={minimum!r}, max={maximum!r}"
        )
    requested[:3] = np.clip(requested[:3], minimum, maximum)
    delta = requested[:3] - actual[:3]
    distance = float(np.linalg.norm(delta))
    lead = float(max_target_lead)
    if not np.isfinite(lead) or lead <= 0.0:
        raise ValueError("max_target_lead must be positive and finite")
    if distance > lead and distance > 1e-12:
        requested[:3] = actual[:3] + delta * (lead / distance)
    return requested


def gripper_state(value: float, threshold: float = 0.0) -> float:
    """Map an arbitrary policy gripper value to exact open/close state."""
    return 1.0 if float(value) > float(threshold) else -1.0


@dataclass(frozen=True)
class PolicyTarget:
    target_pose: np.ndarray
    gripper_state: float
    raw_chunk: np.ndarray


def policy_row_to_target(
    *,
    chunk: np.ndarray,
    target_mode: str,
    include_rz: bool,
    enable_rz: bool,
    action_scale: float,
    delta_scale: float,
    actual_pose: Sequence[float],
    command_pose: Sequence[float],
    gripper_threshold: float,
) -> PolicyTarget:
    """Convert the first Mini-LaWAM row into an absolute requested TCP target."""
    rows = np.asarray(chunk, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError(f"policy chunk must have shape [H,D], got {rows.shape}")
    actual = _pose6(actual_pose, "actual_pose")
    command = _pose6(command_pose, "command_pose")
    row = rows[0]
    target = command.copy()

    if target_mode == "joystick":
        expected = 5 if include_rz else 4
        if row.shape != (expected,):
            raise ValueError(f"joystick checkpoint must emit {expected} values, got {row.shape}")
        if not np.isfinite(action_scale) or action_scale < 0.0:
            raise ValueError("action_scale must be finite and non-negative")
        target[:3] += row[:3].astype(np.float64) * float(action_scale)
        if include_rz and enable_rz:
            target[3:6] = compose_base_z_delta(command[3:6], float(row[3]) * float(action_scale))
    elif target_mode in ("abs", "delta"):
        if row.shape[0] < 4:
            raise ValueError(f"{target_mode} checkpoint must emit at least four values")
        requested_xyz = row[:3].astype(np.float64)
        if target_mode == "delta" and float(delta_scale) != 1.0:
            requested_xyz = actual[:3] + float(delta_scale) * (requested_xyz - actual[:3])
        target[:3] = requested_xyz
    else:
        raise ValueError(f"unsupported Mini-LaWAM target_mode={target_mode!r}")

    return PolicyTarget(
        target_pose=target,
        gripper_state=gripper_state(float(row[-1]), gripper_threshold),
        raw_chunk=rows.copy(),
    )
