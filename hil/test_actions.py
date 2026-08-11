from __future__ import annotations

import numpy as np

from hil.actions import action_from_target, clamp_target_pose, policy_row_to_target


def test_joystick_row_advances_command_target_and_preserves_gripper() -> None:
    command = np.asarray([0.3, 0.1, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    chunk = np.asarray([[0.01, -0.02, 0.03, 0.2, 1.0]], dtype=np.float32)

    result = policy_row_to_target(
        chunk=chunk,
        target_mode="joystick",
        include_rz=True,
        enable_rz=False,
        action_scale=0.5,
        delta_scale=1.0,
        actual_pose=command,
        command_pose=command,
        gripper_threshold=0.0,
    )

    np.testing.assert_allclose(result.target_pose[:3], [0.305, 0.09, 0.215], atol=1e-7)
    np.testing.assert_allclose(result.target_pose[3:6], command[3:6], atol=1e-7)
    assert result.gripper_state == 1.0


def test_action_rotation_is_composed_instead_of_subtracting_rotvecs() -> None:
    start = np.asarray([0.0, 0.0, 0.0, 0.2, -0.1, 0.3])
    end = np.asarray([0.01, 0.02, 0.03, -0.2, 0.4, 0.1])

    action = action_from_target(start, end, -1.0)

    np.testing.assert_allclose(action[:3], [0.01, 0.02, 0.03], atol=1e-7)
    assert not np.allclose(action[3:6], end[3:6] - start[3:6])
    assert action[6] == -1.0


def test_target_clamp_limits_workspace_and_measured_tcp_lead() -> None:
    actual = np.asarray([0.4, 0.0, 0.2, 0.0, 0.0, 0.0])
    requested = np.asarray([1.0, 0.0, 0.2, 0.0, 0.0, 0.0])

    target = clamp_target_pose(requested, actual, [0.0, -1.0, 0.0], [0.5, 1.0, 1.0], 0.02)

    np.testing.assert_allclose(target[:3], [0.42, 0.0, 0.2], atol=1e-7)
