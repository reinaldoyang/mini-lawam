"""Inference wrapper for the trained Stage 1 correction and Stage 2 gate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .stage1_dataset import make_low_dim
from .stage1_model import Stage1CorrectionPolicy, Stage1ModelConfig, denormalize_arm_residual
from .stage2_model import Stage2GateConfig, Stage2GatedCorrectionPolicy


@dataclass(frozen=True)
class GatedCorrection:
    active: bool
    gate_probability: float
    arm_residual: np.ndarray
    gripper_label: int
    gripper_state: float


class GateLatch:
    """Probability threshold with hysteresis to avoid frame-wise gate chatter."""

    def __init__(self, on_threshold: float, hysteresis: float) -> None:
        self.on_threshold = float(on_threshold)
        self.off_threshold = self.on_threshold - float(hysteresis)
        if not 0.0 <= self.on_threshold <= 1.0:
            raise ValueError("gate threshold must be in [0,1]")
        if not 0.0 <= float(hysteresis) <= self.on_threshold:
            raise ValueError("gate hysteresis must be in [0, gate threshold]")
        self.active = False

    def reset(self) -> None:
        self.active = False

    def update(self, probability: float) -> bool:
        value = float(probability)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"gate probability must be finite and in [0,1], got {value}")
        if self.active:
            self.active = value >= self.off_threshold
        else:
            self.active = value >= self.on_threshold
        return self.active


def compose_gated_action(
    base_action: Sequence[float],
    correction: GatedCorrection,
    *,
    enable_rz: bool,
) -> np.ndarray:
    """Apply the gated XYZ/RZ residual and absolute gripper prediction."""
    base = np.asarray(base_action, dtype=np.float32)
    if base.shape != (7,) or not np.all(np.isfinite(base)):
        raise ValueError(f"base action must be a finite (7,) array, got {base!r}")
    if not correction.active:
        return base.copy()
    residual = np.asarray(correction.arm_residual, dtype=np.float32)
    if residual.shape != (4,) or not np.all(np.isfinite(residual)):
        raise ValueError(f"arm residual must be a finite (4,) array, got {residual!r}")
    result = base.copy()
    result[:3] += residual[:3]
    if enable_rz:
        result[5] += residual[3]
    result[6] = float(correction.gripper_state)
    return result


def _load_checkpoint(path: Path, device: torch.device) -> tuple[Stage2GatedCorrectionPolicy, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("stage") != "residual_gate_stage2":
        raise RuntimeError(f"{path} is not a Stage 2 gated-correction checkpoint")
    config = Stage1ModelConfig.from_dict(checkpoint.get("model_config", {}))
    if config.arm_action_dim != 4 or config.gripper_classes != 2:
        raise RuntimeError(
            "gated rollout requires a 4D XYZ/RZ arm head and binary gripper head; "
            f"checkpoint has arm_action_dim={config.arm_action_dim}, "
            f"gripper_classes={config.gripper_classes}"
        )
    correction_policy = Stage1CorrectionPolicy(config, initialize_pretrained=False)
    gate_config = Stage2GateConfig.from_dict(checkpoint.get("gate_config", {}))
    model = Stage2GatedCorrectionPolicy(correction_policy, gate_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] not in (3, 4):
        raise ValueError(f"correction image must be HWC RGB, got {value.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(value[..., :3])).permute(2, 0, 1).float()
    return torch.clamp(tensor / 255.0, 0.0, 1.0)


class GatedResidualPolicy:
    """Stateful one-step Stage 1/2 inference with training-time preprocessing."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device,
        gate_threshold: float | None = None,
        gate_hysteresis: float = 0.05,
    ) -> None:
        self.path = Path(checkpoint_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Stage 2 checkpoint does not exist: {self.path}")
        requested = torch.device(device)
        self.device = torch.device("cpu") if requested.type == "cuda" and not torch.cuda.is_available() else requested
        self.model, self.checkpoint = _load_checkpoint(self.path, self.device)
        self.config = self.model.correction_policy.config

        self.low_dim_mean = np.asarray(self.checkpoint.get("low_dim_mean"), dtype=np.float32)
        self.low_dim_std = np.asarray(self.checkpoint.get("low_dim_std"), dtype=np.float32)
        if self.low_dim_mean.shape != (self.config.low_dim_dim,) or self.low_dim_std.shape != (
            self.config.low_dim_dim,
        ):
            raise RuntimeError(
                "checkpoint low-dimensional statistics do not match model config: "
                f"mean={self.low_dim_mean.shape}, std={self.low_dim_std.shape}, "
                f"expected={(self.config.low_dim_dim,)}"
            )
        if not np.all(np.isfinite(self.low_dim_mean)) or not np.all(np.isfinite(self.low_dim_std)):
            raise RuntimeError("checkpoint low-dimensional statistics contain non-finite values")
        self.low_dim_std = np.maximum(self.low_dim_std, 1e-6)

        self.action_clip = torch.as_tensor(
            np.asarray(self.checkpoint.get("action_clip"), dtype=np.float32),
            device=self.device,
        )
        if self.action_clip.shape != (4,) or not bool(torch.isfinite(self.action_clip).all()):
            raise RuntimeError(f"checkpoint action_clip must be four finite values, got {self.action_clip}")
        if not bool((self.action_clip > 0).all()):
            raise RuntimeError("checkpoint action_clip values must be positive")

        semantics = self.checkpoint.get("gate_training_semantics", {})
        trained_threshold = float(semantics.get("gate_eval_threshold", 0.5))
        threshold = trained_threshold if gate_threshold is None else float(gate_threshold)
        self.gate = GateLatch(threshold, gate_hysteresis)
        self.table_history: deque[torch.Tensor] = deque(maxlen=self.config.temporal_context)
        self.wrist_history: deque[torch.Tensor] = deque(maxlen=self.config.temporal_context)
        self.low_dim_history: deque[torch.Tensor] = deque(maxlen=self.config.temporal_context)

    @property
    def gate_threshold(self) -> float:
        return self.gate.on_threshold

    @property
    def gate_off_threshold(self) -> float:
        return self.gate.off_threshold

    def reset(self) -> None:
        self.gate.reset()
        self.table_history.clear()
        self.wrist_history.clear()
        self.low_dim_history.clear()

    def assert_base_policy_compatible(
        self,
        base_checkpoint: str | Path,
        *,
        target_mode: str,
    ) -> None:
        summary = self.checkpoint.get("data_summary", {})
        schema = str(summary.get("schema", ""))
        if "forward_command_delta" not in schema:
            raise RuntimeError(f"unsupported correction action schema {schema!r}")
        expected_mode = str(summary.get("base_policy_target_mode", ""))
        if expected_mode and expected_mode != str(target_mode):
            raise RuntimeError(
                f"correction checkpoint expects base target_mode={expected_mode!r}, got {target_mode!r}"
            )
        expected_checkpoint = str(summary.get("base_policy_checkpoint", ""))
        if expected_checkpoint and Path(expected_checkpoint).name != Path(base_checkpoint).name:
            raise RuntimeError(
                "correction checkpoint was collected with a different base policy: "
                f"expected {Path(expected_checkpoint).name!r}, got {Path(base_checkpoint).name!r}"
            )

    def _temporal_batch(self, table: torch.Tensor, wrist: torch.Tensor, low_dim: torch.Tensor) -> dict:
        self.table_history.append(table)
        self.wrist_history.append(wrist)
        self.low_dim_history.append(low_dim)

        def padded(values: deque[torch.Tensor]) -> torch.Tensor:
            items = list(values)
            items = [items[0]] * (self.config.temporal_context - len(items)) + items
            stacked = torch.stack(items, dim=0)
            if self.config.temporal_context == 1:
                stacked = stacked[0]
            return stacked.unsqueeze(0).to(self.device, non_blocking=True)

        return {
            "table_cam": padded(self.table_history),
            "wrist_cam": padded(self.wrist_history),
            "low_dim": padded(self.low_dim_history),
        }

    @torch.no_grad()
    def predict(
        self,
        *,
        table_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        robot_observation: Mapping[str, np.ndarray],
        gripper_state: float,
        base_action: Sequence[float],
    ) -> GatedCorrection:
        base = np.asarray(base_action, dtype=np.float32)
        low_dim = make_low_dim(
            robot_observation["eef_pos_base"],
            robot_observation["eef_quat_base"],
            robot_observation["joint_pos"],
            gripper_state,
            base,
            mode=self.config.low_dim_mode,
        )
        if low_dim.shape != (self.config.low_dim_dim,):
            raise RuntimeError(
                f"runtime low-dimensional input has shape {low_dim.shape}, "
                f"expected {(self.config.low_dim_dim,)}"
            )
        normalized = (low_dim.astype(np.float32) - self.low_dim_mean) / self.low_dim_std
        batch = self._temporal_batch(
            _image_tensor(table_rgb),
            _image_tensor(wrist_rgb),
            torch.from_numpy(normalized.astype(np.float32)),
        )
        output = self.model(batch)
        probability = float(F.softmax(output["gate_logits"], dim=-1)[0, 1].item())
        active = self.gate.update(probability)
        residual = denormalize_arm_residual(output["arm_residual_norm"], self.action_clip)[0]
        arm = residual.detach().cpu().numpy().astype(np.float32)
        gripper_label = int(output["gripper_logits"].argmax(dim=-1)[0].item())
        gripper_state_value = 1.0 if gripper_label == 1 else -1.0
        return GatedCorrection(
            active=active,
            gate_probability=probability,
            arm_residual=arm,
            gripper_label=gripper_label,
            gripper_state=gripper_state_value,
        )
