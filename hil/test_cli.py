from __future__ import annotations

from hil.collect_corrections import build_parser


def _camera_args() -> list[str]:
    return ["--table-cam-serial", "table", "--wrist-cam-serial", "wrist"]


def test_vr_defaults_match_demonstration_recorder() -> None:
    args = build_parser().parse_args(_camera_args())

    assert args.vr_position_scale == 1.2
    assert args.vr_max_linear_speed == 0.2
    assert args.vr_rz_scale == -1.0
    assert args.vr_max_angular_speed == 0.5


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

