"""Stage 1 arm-residual and gripper-correction network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import ARM_CORRECTION_INDICES, ARM_CORRECTION_MEANING


@dataclass
class Stage1ModelConfig:
    low_dim_dim: int = 5
    low_dim_mode: str = "image_bc_xyz_rz_grip"
    arm_action_dim: int = 4
    image_encoder: str = "small_cnn"
    image_pretrained: bool = False
    freeze_image_backbone: bool = False
    image_feature_dim: int = 128
    spatial_keypoints: int = 32
    temporal_context: int = 1
    fusion_hidden_dim: int = 512
    fusion_hidden_depth: int = 1
    fusion_output_dim: int = 512
    action_head_hidden_dim: int = 128
    action_head_hidden_depth: int = 3
    action_head_type: str = "deterministic"
    num_gmm_modes: int = 5
    gripper_classes: int = 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "Stage1ModelConfig":
        known = {key: value for key, value in values.items() if key in cls.__dataclass_fields__}
        return cls(**known)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    *,
    output_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for _ in range(int(hidden_depth)):
        layers.extend((nn.Linear(last_dim, int(hidden_dim)), nn.ReLU()))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class SmallImageEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(128, int(output_dim)), nn.ReLU())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.conv(image))


class SpatialSoftmax(nn.Module):
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        attention = F.softmax(feature.reshape(batch, channels, height * width), dim=-1)
        pos_x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        pos_y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        expected_x = torch.sum(attention * grid_x.reshape(1, 1, -1), dim=-1)
        expected_y = torch.sum(attention * grid_y.reshape(1, 1, -1), dim=-1)
        return torch.cat((expected_x, expected_y), dim=-1)


class ResNet18SpatialEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        pretrained: bool,
        freeze_backbone: bool,
        spatial_keypoints: int,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        self.keypoint_conv = nn.Conv2d(512, int(spatial_keypoints), kernel_size=1)
        self.spatial = SpatialSoftmax()
        self.projection = nn.Sequential(
            nn.Linear(int(spatial_keypoints) * 2, int(output_dim)),
            nn.ReLU(),
        )
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = (image - self.image_mean.to(dtype=image.dtype)) / self.image_std.to(dtype=image.dtype)
        keypoints = self.spatial(self.keypoint_conv(self.backbone(normalized)))
        return self.projection(keypoints)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


def make_image_encoder(config: Stage1ModelConfig, *, initialize_pretrained: bool) -> nn.Module:
    if config.image_encoder == "small_cnn":
        return SmallImageEncoder(config.image_feature_dim)
    if config.image_encoder == "resnet18_spatial":
        return ResNet18SpatialEncoder(
            config.image_feature_dim,
            pretrained=bool(config.image_pretrained and initialize_pretrained),
            freeze_backbone=config.freeze_image_backbone,
            spatial_keypoints=config.spatial_keypoints,
        )
    raise ValueError(f"unknown image_encoder={config.image_encoder!r}")


class DeterministicActionHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, hidden_depth: int) -> None:
        super().__init__()
        self.network = build_mlp(
            input_dim,
            action_dim,
            hidden_dim,
            hidden_depth,
            output_activation=nn.Tanh(),
        )

    def mode(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)

    def loss(
        self,
        feature: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        per_sample = F.smooth_l1_loss(
            self.network(feature),
            target,
            reduction="none",
            beta=0.1,
        ).mean(dim=-1)
        return weighted_mean(per_sample, sample_weight)


class GMMActionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_modes: int,
        hidden_dim: int,
        hidden_depth: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_modes = int(num_modes)
        output_dim = self.num_modes * (1 + 2 * self.action_dim)
        self.network = build_mlp(input_dim, output_dim, hidden_dim, hidden_depth)

    def _distribution(self, feature: torch.Tensor):
        raw = self.network(feature)
        logits = raw[..., : self.num_modes]
        parameters = raw[..., self.num_modes :].reshape(
            *raw.shape[:-1],
            self.num_modes,
            2 * self.action_dim,
        )
        means, raw_scales = torch.split(parameters, self.action_dim, dim=-1)
        means = torch.tanh(means)
        scales = F.softplus(raw_scales) + 1e-4
        components = torch.distributions.Independent(
            torch.distributions.Normal(means, scales),
            1,
        )
        return logits, means, torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits),
            components,
        )

    def mode(self, feature: torch.Tensor) -> torch.Tensor:
        logits, means, _ = self._distribution(feature)
        best = logits.argmax(dim=-1)
        gather = best[..., None, None].expand(*best.shape, 1, self.action_dim)
        return means.gather(-2, gather).squeeze(-2)

    def loss(
        self,
        feature: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, _, distribution = self._distribution(feature)
        return weighted_mean(-distribution.log_prob(target), sample_weight)


def weighted_mean(value: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    if sample_weight is None:
        return value.mean()
    weight = sample_weight.to(device=value.device, dtype=value.dtype).reshape(-1)
    return (value.reshape(-1) * weight).sum() / weight.sum().clamp(min=1e-6)


class Stage1CorrectionPolicy(nn.Module):
    """Two-camera correction policy with arm and gripper heads."""

    def __init__(self, config: Stage1ModelConfig, *, initialize_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config
        self.table_encoder = make_image_encoder(config, initialize_pretrained=initialize_pretrained)
        self.wrist_encoder = make_image_encoder(config, initialize_pretrained=initialize_pretrained)
        fusion_input = config.image_feature_dim * 2 + config.low_dim_dim
        self.fusion = build_mlp(
            fusion_input,
            config.fusion_output_dim,
            config.fusion_hidden_dim,
            config.fusion_hidden_depth,
        )
        self.temporal: Optional[nn.GRU] = None
        if config.temporal_context > 1:
            self.temporal = nn.GRU(
                input_size=config.fusion_output_dim,
                hidden_size=config.fusion_output_dim,
                batch_first=True,
            )
        head_type = str(config.action_head_type)
        if head_type == "deterministic":
            self.arm_head: nn.Module = DeterministicActionHead(
                config.fusion_output_dim,
                config.arm_action_dim,
                config.action_head_hidden_dim,
                config.action_head_hidden_depth,
            )
        elif head_type == "gmm":
            self.arm_head = GMMActionHead(
                config.fusion_output_dim,
                config.arm_action_dim,
                config.num_gmm_modes,
                config.action_head_hidden_dim,
                config.action_head_hidden_depth,
            )
        else:
            raise ValueError(f"unknown action_head_type={head_type!r}")
        self.gripper_head = build_mlp(
            config.fusion_output_dim,
            config.gripper_classes,
            config.action_head_hidden_dim,
            config.action_head_hidden_depth,
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        table = batch["table_cam"]
        wrist = batch["wrist_cam"]
        low_dim = batch["low_dim"]
        if table.ndim == 5:
            batch_size, time = table.shape[:2]
            table = table.reshape(batch_size * time, *table.shape[2:])
            wrist = wrist.reshape(batch_size * time, *wrist.shape[2:])
            low_dim = low_dim.reshape(batch_size * time, low_dim.shape[-1])
            feature = self.fusion(
                torch.cat((self.table_encoder(table), self.wrist_encoder(wrist), low_dim), dim=-1)
            ).reshape(batch_size, time, -1)
            if self.temporal is None:
                return feature[:, -1]
            sequence, _ = self.temporal(feature)
            return sequence[:, -1]
        feature = self.fusion(
            torch.cat((self.table_encoder(table), self.wrist_encoder(wrist), low_dim), dim=-1)
        )
        if self.temporal is not None:
            sequence, _ = self.temporal(feature[:, None])
            return sequence[:, -1]
        return feature

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature = self.encode(batch)
        return {
            "feature": feature,
            "arm_residual_norm": self.arm_head.mode(feature),
            "gripper_logits": self.gripper_head(feature),
        }

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        arm_target_norm: torch.Tensor,
        gripper_target: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor],
        gripper_loss_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature = self.encode(batch)
        arm_loss = self.arm_head.loss(feature, arm_target_norm, sample_weight)
        gripper_ce = F.cross_entropy(
            self.gripper_head(feature),
            gripper_target.long(),
            reduction="none",
        )
        gripper_loss = weighted_mean(gripper_ce, sample_weight)
        total = arm_loss + float(gripper_loss_weight) * gripper_loss
        return total, {
            "arm_loss": arm_loss.detach(),
            "gripper_ce": gripper_loss.detach(),
        }


def select_arm_correction(canonical_residual: torch.Tensor) -> torch.Tensor:
    """Project canonical 7D residuals to the controllable XYZ/RZ arm space."""
    if canonical_residual.shape[-1] != 7:
        raise ValueError(f"canonical residual must end in dimension 7, got {canonical_residual.shape}")
    return canonical_residual[..., list(ARM_CORRECTION_INDICES)]


def make_action_clip(max_xyz: float, max_rotation: float, *, device=None) -> torch.Tensor:
    return torch.tensor(
        [max_xyz, max_xyz, max_xyz, max_rotation],
        dtype=torch.float32,
        device=device,
    )


def normalize_arm_residual(target: torch.Tensor, action_clip: torch.Tensor) -> torch.Tensor:
    clip = action_clip.to(device=target.device, dtype=target.dtype).clamp(min=1e-8)
    return torch.clamp(target / clip, -1.0, 1.0)


def denormalize_arm_residual(value: torch.Tensor, action_clip: torch.Tensor) -> torch.Tensor:
    clip = action_clip.to(device=value.device, dtype=value.dtype)
    return torch.clamp(value, -1.0, 1.0) * clip


def load_stage1_checkpoint(path: str | Path, *, device: str | torch.device = "cpu"):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    config = Stage1ModelConfig.from_dict(checkpoint["model_config"])
    model = Stage1CorrectionPolicy(config, initialize_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint
