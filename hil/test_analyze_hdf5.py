from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from hil.analyze_hdf5 import analyze_file, true_runs


def _make_dataset(path) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("base_policy_actions", data=np.zeros((8, 7), dtype=np.float32))
        demo.create_dataset(
            "intervene_mask",
            data=np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.bool_),
        )
        demo.create_dataset(
            "manual_control_mask",
            data=np.asarray([0, 1, 1, 1, 1, 0, 1, 1], dtype=np.bool_),
        )
        demo.create_dataset("timestamps", data=np.arange(8, dtype=np.float64) * 0.05)
        demo.attrs["outcome"] = "saved"


def test_true_runs_counts_contiguous_regions() -> None:
    mask = np.asarray([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.bool_)

    assert true_runs(mask) == [(1, 3), (4, 5), (6, 8)]


def test_analyzer_distinguishes_correction_bursts_and_takeovers(tmp_path) -> None:
    path = tmp_path / "corrections.hdf5"
    _make_dataset(path)

    report = analyze_file(path)
    episode = report["episodes"][0]
    total = report["total"]

    assert episode["frames"] == 8
    assert episode["correction_frames"] == 4
    assert episode["correction_events"] == 3
    assert episode["takeover_frames"] == 6
    assert episode["takeover_events"] == 2
    assert episode["longest_correction_frames"] == 2
    assert total["correction_fraction"] == 0.5
    assert total["episodes_with_corrections"] == 1
