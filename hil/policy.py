"""Mini-LaWAM adapter for canonical HIL actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .actions import (
    PolicyTarget,
    action_from_target,
    clamp_target_pose,
    policy_row_to_target,
)


@dataclass(frozen=True)
class BasePolicyPrediction:
    action: np.ndarray
    target_pose: np.ndarray
    gripper_state: float
    raw_chunk: np.ndarray
    subgoal_overlay: Optional[np.ndarray] = None


class TemporalEnsembler:
    """ACT-style overlap averaging used by ``rollout_ur7e_vr``.

    Motion channels use the exponentially weighted average of every chunk that
    predicts the current absolute step.  The gripper stays on the newest chunk,
    with the rollout's optional open-only lookahead and release latch.
    """

    def __init__(self, *, enabled: bool, decay: float, gripper_threshold: float, open_lead_steps: int) -> None:
        self.enabled = bool(enabled)
        self.decay = float(decay)
        self.gripper_threshold = float(gripper_threshold)
        self.open_lead_steps = int(open_lead_steps)
        if not np.isfinite(self.decay) or self.decay < 0.0:
            raise ValueError("temporal-ensemble decay must be finite and non-negative")
        if self.open_lead_steps < 0:
            raise ValueError("gripper open lead steps must be non-negative")
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.rows_by_step: dict[int, list[np.ndarray]] = {}
        self.last_gripper_command: Optional[str] = None
        self.release_latched = False

    def _select_gripper(self, chunk: np.ndarray) -> float:
        if self.release_latched:
            return -1.0
        immediate = float(chunk[0, -1])
        if self.last_gripper_command != "close":
            return immediate
        end = min(chunk.shape[0] - 1, self.open_lead_steps)
        window = chunk[: end + 1, -1]
        open_offsets = np.flatnonzero(window <= self.gripper_threshold)
        if open_offsets.size:
            self.release_latched = True
            return float(chunk[int(open_offsets[0]), -1])
        return immediate

    def select(self, chunk: np.ndarray) -> np.ndarray:
        rows = np.asarray(chunk, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 4:
            raise ValueError(f"policy chunk must have shape [H,>=4], got {rows.shape}")
        if not np.all(np.isfinite(rows)):
            raise ValueError("policy chunk contains non-finite values")

        if self.enabled:
            for offset, row in enumerate(rows):
                self.rows_by_step.setdefault(self.step + offset, []).append(row.copy())
            predictions = np.asarray(self.rows_by_step.pop(self.step, [rows[0]]), dtype=np.float32)
            count = predictions.shape[0]
            ages = np.arange(count - 1, -1, -1, dtype=np.float64)
            weights = np.exp(-self.decay * ages)
            weights /= weights.sum()
            selected = (predictions * weights[:, None]).sum(axis=0).astype(np.float32)
        else:
            selected = rows[0].copy()

        selected[-1] = self._select_gripper(rows)
        self.last_gripper_command = "close" if selected[-1] > self.gripper_threshold else "open"
        self.step += 1
        return selected


def make_subgoal_overlay(model_input_rgb: np.ndarray, change_grid: np.ndarray, alpha: float) -> np.ndarray:
    """Render the same DINO feature-change overlay as the normal rollout."""
    import cv2

    base = np.asarray(model_input_rgb, dtype=np.uint8)
    heat = np.nan_to_num(np.asarray(change_grid, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if heat.ndim != 2:
        raise ValueError(f"subgoal change map must be 2D, got {heat.shape}")
    low, high = np.percentile(heat, [5.0, 95.0])
    if high <= low + 1e-8:
        normalized = np.zeros_like(heat)
    else:
        normalized = np.clip((heat - low) / (high - low), 0.0, 1.0)
    resized = cv2.resize(normalized, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_bgr = cv2.applyColorMap(
        np.clip(resized * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heat_rgb = heat_bgr[:, :, ::-1].copy()
    return cv2.addWeighted(base, 1.0 - float(alpha), heat_rgb, float(alpha), 0.0)


class MiniLaWAMBasePolicy:
    """Load Mini-LaWAM and convert its rollout-selected row into a safe 7D action.

    The network remains the repository's normal :class:`MiniLaWAMPolicy`; this
    adapter owns only the HIL-specific conversion to a forward command delta.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        train_frame_hw: tuple[int, int] | None,
        action_scale: float,
        delta_scale: float,
        enable_rz: bool,
        gripper_threshold: float,
        workspace_min: Sequence[float],
        workspace_max: Sequence[float],
        max_target_lead: float,
        temporal_ensemble: bool,
        temporal_ensemble_decay: float,
        gripper_open_lead_steps: int,
        target_ema: float,
        target_deadband: float,
        show_subgoal: bool,
        subgoal_alpha: float,
        subgoal_update_steps: int,
    ) -> None:
        from mini_lawam.rollout import MiniLaWAMPolicy

        self.policy = MiniLaWAMPolicy(
            checkpoint,
            device=device,
            train_frame_hw=train_frame_hw,
        )
        self.checkpoint = str(checkpoint)
        self.target_mode = str(getattr(self.policy.cfg, "target_mode", "abs"))
        self.include_rz = bool(getattr(self.policy.cfg, "include_rz", False))
        self.use_wrist = bool(getattr(self.policy.cfg, "use_wrist", False))
        self.use_state = bool(getattr(self.policy.cfg, "use_state", False))
        if self.target_mode not in ("abs", "delta", "joystick"):
            raise ValueError(f"unsupported checkpoint target_mode={self.target_mode!r}")
        if self.target_mode != "delta" and float(delta_scale) != 1.0:
            raise ValueError("--delta-scale only applies to target_mode='delta'")
        self.action_scale = float(action_scale)
        self.delta_scale = float(delta_scale)
        self.enable_rz = bool(enable_rz)
        self.gripper_threshold = float(gripper_threshold)
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self.max_target_lead = float(max_target_lead)
        self.target_ema = float(target_ema)
        self.target_deadband = float(target_deadband)
        self.show_subgoal = bool(show_subgoal)
        self.subgoal_alpha = float(subgoal_alpha)
        self.subgoal_update_steps = int(subgoal_update_steps)
        self.ensembler = TemporalEnsembler(
            enabled=temporal_ensemble,
            decay=temporal_ensemble_decay,
            gripper_threshold=self.gripper_threshold,
            open_lead_steps=gripper_open_lead_steps,
        )
        self.reset()

    @property
    def action_horizon(self) -> int:
        return int(self.policy.cfg.action_horizon)

    @property
    def temporal_ensemble(self) -> bool:
        return self.ensembler.enabled

    def reset(self) -> None:
        """Reset all rollout-scoped smoothing, ensemble, and gripper state."""
        self.ensembler.reset()
        self.ema_xyz: Optional[np.ndarray] = None
        self.last_command_xyz: Optional[np.ndarray] = None
        self.inference_index = 0
        self.latest_subgoal_overlay: Optional[np.ndarray] = None

    def predict(
        self,
        *,
        table_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        actual_pose: Sequence[float],
        command_pose: Sequence[float],
    ) -> BasePolicyPrediction:
        actual = np.asarray(actual_pose, dtype=np.float64).reshape(6)
        command = np.asarray(command_pose, dtype=np.float64).reshape(6)
        update_subgoal = self.show_subgoal and self.inference_index % self.subgoal_update_steps == 0
        self.inference_index += 1
        policy_output = self.policy.act(
            table_rgb,
            wrist_rgb if self.use_wrist else None,
            state_xyz=actual[:3],
            return_subgoal_change=update_subgoal,
        )
        if update_subgoal:
            chunk, change_grid = policy_output
            self.latest_subgoal_overlay = make_subgoal_overlay(
                self.policy.model_input_u8(table_rgb),
                change_grid,
                self.subgoal_alpha,
            )
        else:
            chunk = policy_output
        raw_chunk = np.asarray(chunk, dtype=np.float32)
        deployment_chunk = raw_chunk.copy()
        if self.target_mode == "joystick":
            deployment_chunk[:, :-1] *= self.action_scale
        elif self.target_mode == "delta" and self.delta_scale != 1.0:
            deployment_chunk[:, :3] = (
                actual[:3]
                + self.delta_scale * (deployment_chunk[:, :3] - actual[:3])
            )
        selected_row = self.ensembler.select(deployment_chunk)
        requested: PolicyTarget = policy_row_to_target(
            chunk=selected_row[None, :],
            target_mode=self.target_mode,
            include_rz=self.include_rz,
            enable_rz=self.enable_rz,
            action_scale=1.0,
            delta_scale=1.0,
            actual_pose=actual,
            command_pose=command,
            gripper_threshold=self.gripper_threshold,
        )
        requested.target_pose[:3] = (
            requested.target_pose[:3]
            if self.ema_xyz is None
            else self.target_ema * requested.target_pose[:3] + (1.0 - self.target_ema) * self.ema_xyz
        )
        self.ema_xyz = requested.target_pose[:3].copy()
        if (
            self.target_deadband > 0.0
            and self.last_command_xyz is not None
            and np.linalg.norm(requested.target_pose[:3] - self.last_command_xyz) < self.target_deadband
        ):
            requested.target_pose[:3] = self.last_command_xyz
        target = clamp_target_pose(
            requested.target_pose,
            actual,
            self.workspace_min,
            self.workspace_max,
            self.max_target_lead,
        )
        if self.target_mode == "joystick":
            # The command-relative rollout carries the safety-clamped target
            # into its smoothing state for the next prediction.
            self.ema_xyz = target[:3].copy()
        self.last_command_xyz = target[:3].copy()
        action = action_from_target(command, target, requested.gripper_state)
        return BasePolicyPrediction(
            action=action,
            target_pose=target,
            gripper_state=requested.gripper_state,
            raw_chunk=raw_chunk.copy(),
            subgoal_overlay=self.latest_subgoal_overlay,
        )

    def commit_manual_target(self, target_pose: Sequence[float]) -> None:
        """Re-anchor smoothing to an executed VR target during full takeover."""
        target = np.asarray(target_pose, dtype=np.float64).reshape(6)
        self.ema_xyz = target[:3].copy()
        self.last_command_xyz = target[:3].copy()
