from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")

from hil.gated_policy import GateLatch, GatedCorrection, compose_gated_action
from hil.rollout_gated import build_parser


def _correction(*, active: bool) -> GatedCorrection:
    return GatedCorrection(
        active=active,
        gate_probability=0.8,
        arm_residual=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        gripper_label=1,
        gripper_state=1.0,
    )


def test_gate_hysteresis_prevents_threshold_chatter() -> None:
    gate = GateLatch(on_threshold=0.3, hysteresis=0.05)

    assert not gate.update(0.29)
    assert gate.update(0.31)
    assert gate.update(0.26)
    assert not gate.update(0.24)


def test_gated_action_changes_only_xyz_rz_and_gripper() -> None:
    base = np.asarray([1, 2, 3, 4, 5, 6, -1], dtype=np.float32)

    composed = compose_gated_action(base, _correction(active=True), enable_rz=True)

    np.testing.assert_allclose(composed, [1.1, 2.2, 3.3, 4, 5, 6.4, 1])


def test_disabled_gate_or_rz_preserves_base_channels() -> None:
    base = np.asarray([1, 2, 3, 4, 5, 6, -1], dtype=np.float32)

    np.testing.assert_array_equal(compose_gated_action(base, _correction(active=False), enable_rz=True), base)
    composed = compose_gated_action(base, _correction(active=True), enable_rz=False)
    assert composed[5] == base[5]


def test_rollout_defaults_match_current_checkpoints() -> None:
    args = build_parser().parse_args(
        ["--table-cam-serial", "table", "--wrist-cam-serial", "wrist"]
    )

    assert args.correction_frame_hw == [256, 256]
    assert args.correction_source_frame_hw == [168, 224]
    assert args.gate_threshold is None
    assert args.gate_hysteresis == 0.05
    assert args.control_hz == 20.0
    assert args.trace_dir is None
    assert args.save_frames == 0


def test_trace_capture_arguments_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--table-cam-serial",
            "table",
            "--wrist-cam-serial",
            "wrist",
            "--trace-dir",
            "results/hil/traces",
            "--save-frames",
            "8",
        ]
    )

    assert args.trace_dir == "results/hil/traces"
    assert args.save_frames == 8
