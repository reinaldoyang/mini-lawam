from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from hil.delete_episodes import copy_without_episodes, parse_episode_selectors, write_pruned_copy_atomic


def _make_dataset(path) -> None:
    with h5py.File(path, "w") as file:
        file.attrs["root_attr"] = "preserved"
        file.create_group("meta").attrs["schema"] = "test"
        data = file.create_group("data")
        data.attrs["total"] = 999
        for index, length in enumerate((2, 3, 4)):
            demo = data.create_group(f"demo_{index}")
            actions = demo.create_dataset("bc_actions", data=np.zeros((length, 7), dtype=np.float32))
            demo["base_actions"] = actions
            executed = demo.create_dataset("executed_actions", data=np.ones((length, 7), dtype=np.float32))
            demo["actions"] = executed
            demo.create_dataset("intervene_mask", data=np.arange(length) % 2 == 0)
            demo.create_dataset("manual_control_mask", data=np.arange(length) % 2 == 0)


def test_parse_episode_selectors_supports_names_lists_and_ranges() -> None:
    available = [f"demo_{index}" for index in range(8)]
    assert parse_episode_selectors(["1,demo_3", "5-7"], available) == {
        "demo_1",
        "demo_3",
        "demo_5",
        "demo_6",
        "demo_7",
    }


def test_copy_removes_episode_updates_totals_and_preserves_hard_links(tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    _make_dataset(source)

    mapping = copy_without_episodes(source, output, {"demo_1"})

    assert mapping == {"demo_0": "demo_0", "demo_2": "demo_2"}
    with h5py.File(output, "r") as file:
        assert list(file["data"]) == ["demo_0", "demo_2"]
        assert file.attrs["root_attr"] == "preserved"
        assert file["meta"].attrs["schema"] == "test"
        assert int(file["data"].attrs["total"]) == 6
        assert int(file["data"].attrs["total_interventions"]) == 3
        assert file["data/demo_0/bc_actions"].id == file["data/demo_0/base_actions"].id
        assert file["data/demo_2/executed_actions"].id == file["data/demo_2/actions"].id


def test_copy_can_renumber_retained_episodes(tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "output.hdf5"
    _make_dataset(source)

    mapping = copy_without_episodes(source, output, {"demo_1"}, renumber=True)

    assert mapping == {"demo_0": "demo_0", "demo_2": "demo_1"}
    with h5py.File(output, "r") as file:
        assert list(file["data"]) == ["demo_0", "demo_1"]
        assert file["data/demo_1/bc_actions"].shape == (4, 7)


def test_atomic_copy_can_update_a_pruned_working_file(tmp_path) -> None:
    source = tmp_path / "source.hdf5"
    working = tmp_path / "source_pruned.hdf5"
    _make_dataset(source)

    write_pruned_copy_atomic(source, working, {"demo_0"})
    write_pruned_copy_atomic(working, working, {"demo_2"}, overwrite=True)

    with h5py.File(source, "r") as file:
        assert list(file["data"]) == ["demo_0", "demo_1", "demo_2"]
    with h5py.File(working, "r") as file:
        assert list(file["data"]) == ["demo_1"]
