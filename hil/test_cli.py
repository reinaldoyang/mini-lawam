from __future__ import annotations

import numpy as np

from hil.collect_corrections import build_parser, correction_activity


def _camera_args() -> list[str]:
    return ["--table-cam-serial", "table", "--wrist-cam-serial", "wrist"]


def test_vr_defaults_match_demonstration_recorder() -> None:
    args = build_parser().parse_args(_camera_args())

    assert args.vr_position_scale == 1.2
    assert args.vr_max_linear_speed == 0.2
    assert args.vr_rz_scale == -1.0
    assert args.vr_max_angular_speed == 0.5
    assert args.intervention_translation_deadband == 0.0005
    assert args.intervention_rz_deadband == 0.002


def test_demonstration_recorder_vr_flag_aliases() -> None:
    args = build_parser().parse_args(
        _camera_args()
        + [
            "--position-scale",
            "0.8",
            "--max-linear-speed",
            "0.15",
            "--rz-scale",
            "-0.75",
            "--max-angular-speed",
            "0.4",
        ]
    )

    assert args.vr_position_scale == 0.8
    assert args.vr_max_linear_speed == 0.15
    assert args.vr_rz_scale == -0.75
    assert args.vr_max_angular_speed == 0.4


def test_correction_activity_separates_ownership_from_active_input() -> None:
    still = np.asarray([0.0001, 0, 0, 0, 0, 0.001, -1], dtype=np.float32)
    moving = np.asarray([0.0006, 0, 0, 0, 0, 0.001, -1], dtype=np.float32)

    assert correction_activity(
        still,
        manual_control=True,
        gripper_toggled=False,
        translation_deadband=0.0005,
        rz_deadband=0.002,
    ) == (False, False)
    assert correction_activity(
        moving,
        manual_control=True,
        gripper_toggled=False,
        translation_deadband=0.0005,
        rz_deadband=0.002,
    ) == (True, False)
    assert correction_activity(
        still,
        manual_control=True,
        gripper_toggled=True,
        translation_deadband=0.0005,
        rz_deadband=0.002,
    ) == (False, True)
    assert correction_activity(
        moving,
        manual_control=False,
        gripper_toggled=True,
        translation_deadband=0.0005,
        rz_deadband=0.002,
    ) == (False, False)
