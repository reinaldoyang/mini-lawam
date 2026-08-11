from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from hil.stage1_dataset import Stage1CorrectionDataset, make_low_dim
from hil.stage1_model import (
    Stage1CorrectionPolicy,
    Stage1ModelConfig,
    make_action_clip,
    normalize_arm_residual,
    select_arm_correction,
)


def test_stage1_projects_canonical_residual_to_xyz_rz() -> None:
    canonical = torch.tensor([[1.0, 2.0, 3.0, 40.0, 50.0, 6.0, 1.0]])

    projected = select_arm_correction(canonical)

    torch.testing.assert_close(projected, torch.tensor([[1.0, 2.0, 3.0, 6.0]]))


def test_default_low_dim_is_base_xyz_rz_plus_current_gripper() -> None:
    base = np.asarray([0.1, 0.2, 0.3, 4.0, 5.0, 0.6, 1.0], dtype=np.float32)

    low_dim = make_low_dim(
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.zeros(6),
        -1.0,
        base,
        mode="image_bc_xyz_rz_grip",
    )

    np.testing.assert_allclose(low_dim, [0.1, 0.2, 0.3, 0.6, -1.0])


def test_stage1_model_emits_four_arm_values_and_two_gripper_logits() -> None:
    config = Stage1ModelConfig(
        low_dim_dim=5,
        image_feature_dim=16,
        fusion_hidden_dim=32,
        fusion_output_dim=32,
        action_head_hidden_dim=16,
        action_head_hidden_depth=1,
        action_head_type="deterministic",
    )
    model = Stage1CorrectionPolicy(config)
    batch = {
        "table_cam": torch.zeros(2, 3, 32, 32),
        "wrist_cam": torch.zeros(2, 3, 32, 32),
        "low_dim": torch.zeros(2, 5),
    }

    output = model(batch)

    assert output["arm_residual_norm"].shape == (2, 4)
    assert output["gripper_logits"].shape == (2, 2)
    clip = make_action_clip(0.05, 0.1)
    target = normalize_arm_residual(torch.tensor([[0.1, 0.0, 0.0, 0.2]]), clip)
    torch.testing.assert_close(target, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))


def test_stage1_derives_binary_gripper_target_from_executed_action(tmp_path) -> None:
    path = tmp_path / "old_corrections.hdf5"
    with h5py.File(path, "w") as file:
        group = file.create_group("data/demo_0")
        obs = group.create_group("obs")
        obs.create_dataset("table_cam", data=np.zeros((1, 2, 3, 3), dtype=np.uint8))
        obs.create_dataset("wrist_cam", data=np.zeros((1, 2, 3, 3), dtype=np.uint8))
        obs.create_dataset("eef_pos_base", data=np.zeros((1, 3), dtype=np.float32))
        obs.create_dataset(
            "eef_quat_base",
            data=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        obs.create_dataset("joint_pos", data=np.zeros((1, 6), dtype=np.float32))
        obs.create_dataset("gripper_state", data=-np.ones((1, 1), dtype=np.float32))
        group.create_dataset("bc_actions", data=np.zeros((1, 7), dtype=np.float32))
        executed = np.zeros((1, 7), dtype=np.float32)
        executed[0, 6] = 1.0
        group.create_dataset("executed_actions", data=executed)
        group.create_dataset("residual_targets", data=np.zeros((1, 7), dtype=np.float32))
        group.create_dataset("intervene_mask", data=np.ones(1, dtype=np.bool_))
        # Simulate an older file whose stored correction label is not binary.
        group.create_dataset("gripper_labels", data=np.zeros(1, dtype=np.int64))

    sample = Stage1CorrectionDataset(path)[0]

    assert int(sample["gripper_label"]) == 1
