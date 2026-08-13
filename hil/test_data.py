from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from hil.data import CorrectionEpisode, CorrectionHDF5Writer


def test_hdf5_writer_emits_gated_residual_schema(tmp_path) -> None:
    episode = CorrectionEpisode()
    base = np.asarray([0.01, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    human = np.asarray([0, 0.02, 0, 0, 0, 0, 1], dtype=np.float32)
    episode.append(
        obs={
            "table_cam": np.zeros((2, 3, 3), dtype=np.uint8),
            "wrist_cam": np.ones((2, 3, 3), dtype=np.uint8),
            "eef_pos_base": np.zeros(3, dtype=np.float32),
            "eef_quat_base": np.asarray([1, 0, 0, 0], dtype=np.float32),
            "joint_pos": np.zeros(6, dtype=np.float32),
            "gripper_state": np.asarray([-1], dtype=np.float32),
            "base_policy_action": base,
            "quest_controller": np.zeros(9, dtype=np.float32),
        },
        base_action=base,
        human_action=human,
        executed_action=human,
        residual_target=human - base,
        manual_control=True,
        intervention=True,
        gripper_label=1,
        timestamp=1.0,
    )
    output = tmp_path / "corrections.hdf5"
    args = SimpleNamespace(
        hdf5_compression="lzf",
        hdf5_write_batch_size=8,
        image_height=2,
        image_width=3,
        ckpt="model.pt",
        table_cam_serial="table",
        wrist_cam_serial="wrist",
        intervention_translation_deadband=0.0005,
        intervention_rz_deadband=0.002,
    )
    policy = SimpleNamespace(target_mode="joystick", include_rz=True)

    writer = CorrectionHDF5Writer(output, args, policy)
    try:
        name, count, interventions = writer.write_episode(episode)
    finally:
        writer.close()

    assert (name, count, interventions) == ("demo_0", 1, 1)
    with h5py.File(output, "r") as dataset:
        demo = dataset["data/demo_0"]
        assert demo["obs/table_cam"].shape == (1, 2, 3, 3)
        assert demo["base_policy_actions"].shape == (1, 7)
        assert demo["bc_actions"].id == demo["base_policy_actions"].id
        assert demo["actions"].id == demo["executed_actions"].id
        assert demo["base_actions"].id == demo["base_policy_actions"].id
        assert demo["obs/base_policy_action"].id == demo["obs/bc_action"].id
        assert bool(demo["intervene_mask"][0])
        assert int(demo["gripper_labels"][0]) == 1
        assert dataset["meta"].attrs["gripper_label_schema"] == "binary_executed_state_v1"
        assert dataset["meta"].attrs["gripper_labels"] == "0=open, 1=close"
        assert dataset["meta"].attrs["intervention_label_schema"] == "vr_active_motion_or_gripper_edge_v2"
        assert dataset["meta"].attrs["bc_actions_meaning"] == "compatibility alias of base_policy_actions"
        assert int(dataset["data"].attrs["total_interventions"]) == 1
