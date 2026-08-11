"""Stage 2 intervention gate on top of a frozen Stage 1 correction policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_model import Stage1CorrectionPolicy, build_mlp


@dataclass
class Stage2GateConfig:
    hidden_dim: int = 128
    hidden_depth: int = 3
    classes: int = 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "Stage2GateConfig":
        known = {key: value for key, value in values.items() if key in cls.__dataclass_fields__}
        return cls(**known)


class Stage2GatedCorrectionPolicy(nn.Module):
    """Frozen Stage 1 representation plus a trainable binary intervention head."""

    def __init__(
        self,
        correction_policy: Stage1CorrectionPolicy,
        gate_config: Optional[Stage2GateConfig] = None,
    ) -> None:
        super().__init__()
        self.correction_policy = correction_policy
        self.gate_config = gate_config or Stage2GateConfig()
        if int(self.gate_config.classes) != 2:
            raise ValueError("Stage 2 gate must have exactly two classes")
        self.gate_head = build_mlp(
            correction_policy.config.fusion_output_dim,
            self.gate_config.classes,
            self.gate_config.hidden_dim,
            self.gate_config.hidden_depth,
        )
        freeze_except_gate(self)

    def train(self, mode: bool = True):
        super().train(mode)
        # Stage 2 must see exactly the fixed representation learned in Stage 1.
        self.correction_policy.eval()
        self.gate_head.train(mode)
        return self

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.correction_policy.encode(batch)

    def gate_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.gate_head(self.encode(batch))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature = self.encode(batch)
        return {
            "feature": feature,
            "arm_residual_norm": self.correction_policy.arm_head.mode(feature),
            "gripper_logits": self.correction_policy.gripper_head(feature),
            "gate_logits": self.gate_head(feature),
        }

    def gate_loss(
        self,
        batch: dict[str, torch.Tensor],
        gate_target: torch.Tensor,
        *,
        downsample_negatives: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.gate_logits(batch)
        target = gate_target.long()
        valid = torch.ones_like(target, dtype=torch.bool)

        if downsample_negatives:
            positives = target.bool()
            positive_rate = positives.float().mean().detach().clamp(min=1e-3)
            keep_negative = torch.rand_like(target.float()) <= positive_rate
            valid = positives | keep_negative
            if not bool(valid.any()):
                valid = torch.ones_like(target, dtype=torch.bool)

        loss = F.cross_entropy(logits[valid], target[valid])
        prediction = logits.argmax(dim=-1)
        positives = target.bool()
        true_positive = prediction.bool() & positives
        recall = true_positive.float().sum() / positives.float().sum().clamp(min=1.0)
        return loss, {
            "gate_ce": loss.detach(),
            "gate_accuracy": (prediction == target).float().mean().detach(),
            "gate_recall": recall.detach(),
            "retained_fraction": valid.float().mean().detach(),
        }


def freeze_except_gate(model: Stage2GatedCorrectionPolicy) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.gate_head.parameters():
        parameter.requires_grad = True

