import tempfile
from pathlib import Path

import h5py
import numpy as np

from mini_lawam.data import (
    MiniLaWAMDataset,
    _read_gripper_target,
    _read_target_delta,
    _read_target_joystick,
)
from mini_lawam.rollout_ur7e import (
    compose_target_xyz,
    scale_delta_chunk,
    scale_joystick_chunk,
    select_gripper_with_open_lookahead,
)


def test_joystick_target_uses_same_index_xyz_and_gripper_only():
    actions = np.asarray(
        [
            [0.00, 0.00, 0.00, 9.0, 8.0, 7.0, -1.0],
            [0.05, 0.00, -0.05, 9.0, 8.0, 7.0, 1.0],
            [0.00, 0.05, 0.00, 9.0, 8.0, 7.0, -1.0],
        ],
        dtype=np.float32,
    )

    target = _read_target_joystick({"actions": actions}, t=1, n=2, grip_col=6)

    np.testing.assert_array_equal(
        target,
        np.asarray(
            [
                [0.05, 0.00, -0.05, 1.0],
                [0.00, 0.05, 0.00, -1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_joystick_scale_changes_xyz_but_not_gripper():
    chunk = np.asarray(
        [
            [0.05, -0.05, 0.00, -1.0],
            [0.00, 0.05, 0.05, 1.0],
        ],
        dtype=np.float32,
    )

    scaled = scale_joystick_chunk(chunk, action_scale=0.3)

    np.testing.assert_allclose(scaled[:, :3], chunk[:, :3] * 0.3)
    np.testing.assert_array_equal(scaled[:, 3], chunk[:, 3])
    np.testing.assert_array_equal(
        chunk[:, 3], np.asarray([-1.0, 1.0], dtype=np.float32)
    )
    current = np.asarray([0.4, -0.1, 0.2])
    np.testing.assert_allclose(
        compose_target_xyz(scaled[0, :3], current, target_mode="joystick"),
        current + chunk[0, :3] * 0.3,
    )


def test_joystick_xyz_can_keep_same_index_with_next_row_gripper():
    actions = np.asarray(
        [
            [0.05, 0.00, 0.00, 0.0, 0.0, 0.0, 1.0],
            [0.00, 0.05, 0.00, 0.0, 0.0, 0.0, -1.0],
            [0.00, 0.00, 0.05, 0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    group = {"actions": actions}

    target = _read_target_joystick(group, t=0, n=2, grip_col=6)
    target[:, 3:4] = _read_gripper_target(
        group, t=0, n=2, grip_col=6, offset=1
    )

    np.testing.assert_array_equal(target[:, :3], actions[:2, :3])
    np.testing.assert_array_equal(target[:, 3], actions[1:3, 6])


def test_existing_delta_target_and_scale_semantics_are_preserved():
    group = {
        "obs": {
            "eef": np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [1.1, 2.2, 3.3],
                    [1.2, 2.4, 3.6],
                ],
                dtype=np.float32,
            )
        },
        "actions": np.asarray(
            [
                [0, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, -1],
            ],
            dtype=np.float32,
        ),
    }
    delta = _read_target_delta(group, t=0, n=2, pos_key="eef", grip_col=6)
    absolute_chunk = delta.copy()
    absolute_chunk[:, :3] += group["obs"]["eef"][0]

    scaled = scale_delta_chunk(
        absolute_chunk, anchor_xyz=group["obs"]["eef"][0], delta_scale=0.3
    )

    np.testing.assert_allclose(
        scaled[:, :3],
        group["obs"]["eef"][0] + 0.3 * delta[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(scaled[:, 3], delta[:, 3])
    np.testing.assert_allclose(
        compose_target_xyz(scaled[0, :3], None, target_mode="delta"),
        scaled[0, :3],
    )


def test_tail_actions_include_pre_release_anchor_with_masked_padding():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tail.hdf5"
        with h5py.File(path, "w") as f:
            demo = f.create_group("data/demo_0")
            obs = demo.create_group("obs")
            obs.create_dataset("table_cam", data=np.zeros((10, 2, 2, 3), dtype=np.uint8))
            obs.create_dataset("eef_pos_base", data=np.zeros((10, 3), dtype=np.float32))
            actions = np.zeros((10, 7), dtype=np.float32)
            actions[:, 6] = 1.0
            actions[9, 6] = -1.0
            demo.create_dataset("actions", data=actions)

        legacy = MiniLaWAMDataset(
            str(path), gap=4, horizon=4, target_mode="joystick",
            gripper_target_offset=1, include_tail_actions=False,
        )
        tail = MiniLaWAMDataset(
            str(path), gap=4, horizon=4, target_mode="joystick",
            gripper_target_offset=1, include_tail_actions=True,
        )
        assert ("demo_0", 8) not in legacy.index
        assert ("demo_0", 8) in tail.index

        item = tail[tail.index.index(("demo_0", 8))]
        assert int(item["actions_mask"][:, 0].sum()) == 1
        assert item["gripper_targets"][0, 0].item() == 0.0


def test_open_lookahead_preserves_grasp_then_latches_release():
    chunk = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    # While currently open, closing remains an immediate chunk[0] decision.
    grip, latched, source = select_gripper_with_open_lookahead(
        chunk, step_index=0, last_cmd="open", release_latched=False,
        open_lead_steps=1,
    )
    assert grip == 1.0
    assert not latched
    assert source == 0

    # Once closed, the next-step open forecast triggers release now.
    grip, latched, source = select_gripper_with_open_lookahead(
        chunk, step_index=0, last_cmd="close", release_latched=False,
        open_lead_steps=1,
    )
    assert grip == -1.0
    assert latched
    assert source == 1

    # The latch prevents an unchanged image from commanding close again.
    all_close = np.ones((3, 4), dtype=np.float32)
    grip, latched, source = select_gripper_with_open_lookahead(
        all_close, step_index=0, last_cmd="open", release_latched=True,
        open_lead_steps=1,
    )
    assert grip == -1.0
    assert latched
    assert source == -1


if __name__ == "__main__":
    test_joystick_target_uses_same_index_xyz_and_gripper_only()
    test_joystick_scale_changes_xyz_but_not_gripper()
    test_joystick_xyz_can_keep_same_index_with_next_row_gripper()
    test_existing_delta_target_and_scale_semantics_are_preserved()
    test_tail_actions_include_pre_release_anchor_with_masked_padding()
    test_open_lookahead_preserves_grasp_then_latches_release()
    print("6 focused target-mode tests passed")
