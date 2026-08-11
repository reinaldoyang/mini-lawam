from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hil.stage1_model import Stage1CorrectionPolicy, Stage1ModelConfig
from hil.stage2_model import Stage2GateConfig, Stage2GatedCorrectionPolicy


def test_stage2_only_trains_binary_gate_head() -> None:
    correction = Stage1CorrectionPolicy(
        Stage1ModelConfig(
            low_dim_dim=5,
            image_feature_dim=8,
            fusion_hidden_dim=16,
            fusion_output_dim=16,
            action_head_hidden_dim=8,
            action_head_hidden_depth=1,
        )
    )
    model = Stage2GatedCorrectionPolicy(
        correction,
        Stage2GateConfig(hidden_dim=8, hidden_depth=1),
    )
    model.train()
    batch = {
        "table_cam": torch.zeros(2, 3, 32, 32),
        "wrist_cam": torch.zeros(2, 3, 32, 32),
        "low_dim": torch.zeros(2, 5),
    }
    target = torch.tensor([0, 1])

    output = model(batch)
    loss, _ = model.gate_loss(batch, target, downsample_negatives=False)
    loss.backward()

    assert output["gate_logits"].shape == (2, 2)
    assert all(not parameter.requires_grad for parameter in model.correction_policy.parameters())
    assert all(parameter.requires_grad for parameter in model.gate_head.parameters())
    assert all(parameter.grad is None for parameter in model.correction_policy.parameters())
    assert any(parameter.grad is not None for parameter in model.gate_head.parameters())
    assert not model.correction_policy.training

