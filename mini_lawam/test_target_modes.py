import numpy as np

from mini_lawam.data import _read_target_delta, _read_target_joystick
from mini_lawam.rollout_ur7e import (
    compose_target_xyz,
    scale_delta_chunk,
    scale_joystick_chunk,
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


if __name__ == "__main__":
    test_joystick_target_uses_same_index_xyz_and_gripper_only()
    test_joystick_scale_changes_xyz_but_not_gripper()
    test_existing_delta_target_and_scale_semantics_are_preserved()
    print("3 focused target-mode tests passed")
