from __future__ import annotations

import numpy as np

from hil.vr import QuestPose, VRClutch


def pose(position=(0.0, 0.0, 0.0), *, grip=False, trigger=False) -> QuestPose:
    return QuestPose(
        position=np.asarray(position, dtype=np.float64),
        quaternion=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        trigger=trigger,
        grip=grip,
    )


def test_quest_csv_accepts_button_protocol() -> None:
    parsed = QuestPose.from_csv("0,0,0,0,0,0,1,0,1,1,0,1,0")
    assert parsed.grip
    assert parsed.button_a
    assert parsed.button_x
    assert not parsed.trigger


def test_side_grip_takes_over_and_release_returns_control() -> None:
    clutch = VRClutch(
        mapping_matrix=np.eye(3),
        position_scale=1.0,
        max_linear_speed=10.0,
        max_angular_speed=10.0,
        locked_rotvec=[0.0, 0.0, 0.0],
    )
    actual = np.asarray([0.3, 0.1, 0.2, 0.0, 0.0, 0.0])

    clutch.update(pose(), 0.0, actual_pose=actual, current_gripper_state=-1.0)
    started = clutch.update(pose(grip=True), 0.1, actual_pose=actual, current_gripper_state=-1.0)
    moved = clutch.update(pose((0.02, -0.01, 0.03), grip=True), 0.2, actual_pose=actual, current_gripper_state=-1.0)
    released = clutch.update(pose(), 0.3, actual_pose=actual, current_gripper_state=-1.0)

    assert started.active and started.started
    np.testing.assert_allclose(started.target_pose, actual, atol=1e-7)
    np.testing.assert_allclose(moved.target_pose[:3], actual[:3] + [0.02, -0.01, 0.03], atol=1e-7)
    assert not released.active and released.released
    assert released.target_pose is None


def test_front_trigger_toggles_gripper_only_during_takeover() -> None:
    clutch = VRClutch(locked_rotvec=[0.0, 0.0, 0.0])
    actual = np.zeros(6)

    inactive = clutch.update(pose(trigger=True), 0.0, actual_pose=actual, current_gripper_state=-1.0)
    clutch.update(pose(), 0.1, actual_pose=actual, current_gripper_state=-1.0)
    clutch.update(pose(grip=True), 0.2, actual_pose=actual, current_gripper_state=-1.0)
    toggled = clutch.update(pose(grip=True, trigger=True), 0.3, actual_pose=actual, current_gripper_state=-1.0)

    assert not inactive.gripper_toggled
    assert toggled.gripper_toggled
    assert toggled.gripper_state == 1.0


def test_reset_can_require_physical_release_before_rearming() -> None:
    clutch = VRClutch(locked_rotvec=[0.0, 0.0, 0.0])
    actual = np.zeros(6)
    clutch.reset(require_release=True)

    still_held = clutch.update(pose(grip=True), 0.0, actual_pose=actual, current_gripper_state=-1.0)
    clutch.update(pose(), 0.1, actual_pose=actual, current_gripper_state=-1.0)
    fresh_press = clutch.update(pose(grip=True), 0.2, actual_pose=actual, current_gripper_state=-1.0)

    assert not still_held.active
    assert fresh_press.active and fresh_press.started
