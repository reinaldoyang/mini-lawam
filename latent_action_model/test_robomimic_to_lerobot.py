from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch

from latent_action_model.convert_robomimic_to_lerobot import convert_dataset
from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES


def _write_synthetic_hdf5(path: Path) -> None:
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        for episode_index in range(3):
            length = 6 + episode_index
            demo = data.create_group(f"demo_{episode_index}")
            obs = demo.create_group("obs")
            frames = np.zeros((length, 32, 48, 3), dtype=np.uint8)
            frames[..., 0] = np.arange(length, dtype=np.uint8)[:, None, None]
            frames[..., 1] = 30 + episode_index
            obs.create_dataset("table_cam", data=frames)
            positions = np.zeros((length, 3), dtype=np.float32)
            positions[:, 0] = np.linspace(0.0, 0.1, length)
            positions[:, 1] = episode_index
            obs.create_dataset("eef_pos_base", data=positions)
            quaternions = np.zeros((length, 4), dtype=np.float32)
            quaternions[:, 3] = 1.0
            obs.create_dataset("eef_quat_base", data=quaternions)
            actions = np.zeros((length, 7), dtype=np.float32)
            actions[:, 0] = 0.01
            actions[length // 2 :, 6] = 1.0
            actions[: length // 2, 6] = -1.0
            demo.create_dataset("actions", data=actions)


def test_conversion_is_consumable_by_stage1_loader() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "source.hdf5"
        destination = root / "converted"
        _write_synthetic_hdf5(source)

        summary = convert_dataset(source, destination, fps=10.0, task="synthetic")
        assert summary["episodes"] == 3
        assert summary["frames"] == 21
        assert (destination / "meta/info.json").is_file()
        assert (destination / "meta/modality.json").is_file()
        assert (destination / "data/chunk-000/file-000.parquet").is_file()

        mixture_name = "_synthetic_ur_lam_test"
        DATASET_NAMED_MIXTURES[mixture_name] = [
            (destination.name, 1.0, "robomind_ur_1rgb")
        ]
        try:
            dataset = LeRobotLAMDataset(
                data_root_dir=root,
                data_mix=mixture_name,
                num_frames=2,
                mode="all",
                val_tail_ratio=0.0,
                video_backend="pyav",
                image_hw=(32, 32),
                frame_dt_sec=0.1,
            )
            sample = dataset[0]
        finally:
            DATASET_NAMED_MIXTURES.pop(mixture_name, None)

        assert sample["frames"].shape == (2, 3, 32, 32)
        assert sample["frames"].dtype == torch.uint8
        assert sample["proprio"].shape == (2, 7)
        assert sample["embodiment_id"] == 3


def test_conversion_refuses_to_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "source.hdf5"
        destination = root / "converted"
        _write_synthetic_hdf5(source)
        destination.mkdir()
        try:
            convert_dataset(source, destination)
        except FileExistsError as exc:
            assert "Output path already exists" in str(exc)
        else:
            raise AssertionError("converter overwrote an existing directory")


if __name__ == "__main__":
    test_conversion_is_consumable_by_stage1_loader()
    test_conversion_refuses_to_overwrite()
    print("2 focused robomimic-to-LeRobot conversion tests passed")
