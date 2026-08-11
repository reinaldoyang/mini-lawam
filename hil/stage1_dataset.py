"""Dataset utilities for Stage 1 arm-residual and gripper-correction training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


LOW_DIM_MODES = (
    "full",
    "image_only",
    "image_bc_action",
    "image_bc_xyz_grip",
    "image_bc_xyz_rz_grip",
    "minimal_pose",
)


def validate_low_dim_mode(mode: str) -> str:
    selected = str(mode)
    if selected not in LOW_DIM_MODES:
        raise ValueError(f"unknown low_dim_mode={selected!r}; expected one of {LOW_DIM_MODES}")
    return selected


@dataclass(frozen=True)
class Stage1SampleRef:
    path: str
    demo: str
    step: int


def find_hdf5_files(data_path: str | Path) -> list[str]:
    root = Path(data_path).expanduser()
    if root.is_file():
        return [str(root.resolve())]
    if not root.exists():
        return []
    files = list(root.rglob("*.hdf5")) + list(root.rglob("*.h5"))
    return sorted({str(path.resolve()) for path in files})


def _demo_names(file: h5py.File) -> Iterable[str]:
    if "data" not in file:
        return ()
    return sorted(name for name in file["data"] if name.startswith("demo_"))


def _base_policy_actions(group: h5py.Group) -> h5py.Dataset:
    """Return the canonical field, with support for older HIL files."""
    if "base_policy_actions" in group:
        return group["base_policy_actions"]
    if "bc_actions" in group:
        return group["bc_actions"]
    raise RuntimeError(f"{group.name} has no base_policy_actions (or legacy bc_actions) dataset")


def build_sample_index(
    data_path: str | Path,
    *,
    intervention_only: bool = False,
) -> list[Stage1SampleRef]:
    refs: list[Stage1SampleRef] = []
    for path in find_hdf5_files(data_path):
        with h5py.File(path, "r") as file:
            for demo in _demo_names(file):
                group = file["data"][demo]
                required = ("executed_actions", "residual_targets", "intervene_mask", "obs")
                missing = [key for key in required if key not in group]
                if missing:
                    raise RuntimeError(f"{path}:{demo} is missing Stage 1 keys: {missing}")
                count = int(_base_policy_actions(group).shape[0])
                mask = np.asarray(group["intervene_mask"], dtype=bool)
                if mask.shape != (count,):
                    raise RuntimeError(f"{path}:{demo} intervene_mask has shape {mask.shape}, expected {(count,)}")
                for step in range(count):
                    if intervention_only and not bool(mask[step]):
                        continue
                    refs.append(Stage1SampleRef(path=path, demo=demo, step=step))
    return refs


def split_refs(
    refs: Sequence[Stage1SampleRef],
    *,
    val_fraction: float,
    seed: int,
    split_unit: str,
) -> tuple[list[Stage1SampleRef], list[Stage1SampleRef]]:
    items = list(refs)
    rng = np.random.RandomState(int(seed))
    if split_unit == "demo":
        demos = sorted({(ref.path, ref.demo) for ref in items})
        indices = np.arange(len(demos))
        rng.shuffle(indices)
        val_count = int(round(len(demos) * float(val_fraction)))
        val_count = min(max(val_count, 1 if len(demos) > 1 else 0), len(demos))
        val_demos = {demos[index] for index in indices[:val_count].tolist()}
        train = [ref for ref in items if (ref.path, ref.demo) not in val_demos]
        val = [ref for ref in items if (ref.path, ref.demo) in val_demos]
        return train, val
    if split_unit != "frame":
        raise ValueError("split_unit must be 'demo' or 'frame'")
    indices = np.arange(len(items))
    rng.shuffle(indices)
    val_count = int(round(len(items) * float(val_fraction)))
    val_count = min(max(val_count, 1 if len(items) > 1 else 0), len(items))
    val_indices = set(indices[:val_count].tolist())
    return (
        [ref for index, ref in enumerate(items) if index not in val_indices],
        [ref for index, ref in enumerate(items) if index in val_indices],
    )


def make_low_dim(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    joint_pos: np.ndarray,
    gripper_state: np.ndarray | float,
    base_policy_action: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    selected = validate_low_dim_mode(mode)
    bc = np.asarray(base_policy_action, dtype=np.float32).reshape(-1)
    grip = np.asarray(gripper_state, dtype=np.float32).reshape(-1)
    grip = grip[:1] if grip.size else np.zeros(1, dtype=np.float32)
    if selected == "image_only":
        return np.zeros(0, dtype=np.float32)
    if selected == "image_bc_action":
        return bc
    if selected == "image_bc_xyz_grip":
        return np.concatenate((bc[:3], grip)).astype(np.float32)
    if selected == "image_bc_xyz_rz_grip":
        if bc.shape != (7,):
            raise RuntimeError(f"canonical base action must have shape (7,), got {bc.shape}")
        return np.concatenate((bc[[0, 1, 2, 5]], grip)).astype(np.float32)
    if selected == "minimal_pose":
        return np.concatenate(
            (
                np.asarray(eef_pos, dtype=np.float32).reshape(-1),
                np.asarray(eef_quat, dtype=np.float32).reshape(-1),
                grip,
            )
        ).astype(np.float32)
    return np.concatenate(
        (
            np.asarray(eef_pos, dtype=np.float32).reshape(-1),
            np.asarray(eef_quat, dtype=np.float32).reshape(-1),
            np.asarray(joint_pos, dtype=np.float32).reshape(-1),
            grip,
            bc,
        )
    ).astype(np.float32)


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    value = np.asarray(image)
    if value.ndim == 4:
        return torch.stack([_image_tensor(frame) for frame in value], dim=0)
    if value.ndim != 3 or value.shape[-1] not in (3, 4):
        raise RuntimeError(f"expected HWC RGB image, got {value.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(value[..., :3])).permute(2, 0, 1).float()
    return torch.clamp(tensor / 255.0, 0.0, 1.0)


class Stage1CorrectionDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        *,
        refs: Optional[Sequence[Stage1SampleRef]] = None,
        intervention_only: bool = False,
        low_dim_mode: str = "image_bc_xyz_rz_grip",
        low_dim_mean: Optional[np.ndarray] = None,
        low_dim_std: Optional[np.ndarray] = None,
        temporal_context: int = 1,
    ) -> None:
        super().__init__()
        self.data_path = str(data_path)
        self.refs = list(refs) if refs is not None else build_sample_index(
            data_path,
            intervention_only=intervention_only,
        )
        self.low_dim_mode = validate_low_dim_mode(low_dim_mode)
        self.low_dim_mean = None if low_dim_mean is None else np.asarray(low_dim_mean, dtype=np.float32)
        self.low_dim_std = None if low_dim_std is None else np.asarray(low_dim_std, dtype=np.float32)
        self.temporal_context = max(int(temporal_context), 1)

    def __len__(self) -> int:
        return len(self.refs)

    def _arrays(self, ref: Stage1SampleRef) -> dict[str, np.ndarray]:
        with h5py.File(ref.path, "r") as file:
            group = file["data"][ref.demo]
            obs = group["obs"]
            first = max(ref.step - self.temporal_context + 1, 0)
            steps = list(range(first, ref.step + 1))
            steps = [steps[0]] * (self.temporal_context - len(steps)) + steps
            low_dim = []
            base_policy_actions = _base_policy_actions(group)
            for step in steps:
                base_policy_action = np.asarray(base_policy_actions[step], dtype=np.float32)
                low_dim.append(
                    make_low_dim(
                        obs["eef_pos_base"][step],
                        obs["eef_quat_base"][step],
                        obs["joint_pos"][step],
                        obs["gripper_state"][step],
                        base_policy_action,
                        mode=self.low_dim_mode,
                    )
                )
            arrays = {
                "table_cam": np.stack([np.asarray(obs["table_cam"][step]) for step in steps]),
                "wrist_cam": np.stack([np.asarray(obs["wrist_cam"][step]) for step in steps]),
                "low_dim": np.stack(low_dim).astype(np.float32),
                "base_policy_action": np.asarray(base_policy_actions[ref.step], dtype=np.float32),
                "residual_target": np.asarray(group["residual_targets"][ref.step], dtype=np.float32),
                "intervene_mask": np.asarray(group["intervene_mask"][ref.step], dtype=np.int64),
                "gripper_label": np.asarray(
                    int(float(group["executed_actions"][ref.step, 6]) > 0.0),
                    dtype=np.int64,
                ),
            }
        if self.temporal_context == 1:
            arrays["table_cam"] = arrays["table_cam"][0]
            arrays["wrist_cam"] = arrays["wrist_cam"][0]
            arrays["low_dim"] = arrays["low_dim"][0]
        if arrays["residual_target"].shape != (7,):
            raise RuntimeError(f"{ref.path}:{ref.demo}:{ref.step} residual target must have shape (7,)")
        label = int(arrays["gripper_label"])
        if label not in (0, 1):
            raise RuntimeError(f"{ref.path}:{ref.demo}:{ref.step} invalid gripper label {label}")
        return arrays

    def load_low_dim(self, index: int) -> np.ndarray:
        ref = self.refs[index]
        with h5py.File(ref.path, "r") as file:
            group = file["data"][ref.demo]
            obs = group["obs"]
            base_policy_actions = _base_policy_actions(group)
            return make_low_dim(
                obs["eef_pos_base"][ref.step],
                obs["eef_quat_base"][ref.step],
                obs["joint_pos"][ref.step],
                obs["gripper_state"][ref.step],
                base_policy_actions[ref.step],
                mode=self.low_dim_mode,
            )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        arrays = self._arrays(self.refs[index])
        low_dim = arrays["low_dim"]
        if self.low_dim_mean is not None and self.low_dim_std is not None:
            low_dim = (low_dim - self.low_dim_mean) / np.maximum(self.low_dim_std, 1e-6)
        return {
            "table_cam": _image_tensor(arrays["table_cam"]),
            "wrist_cam": _image_tensor(arrays["wrist_cam"]),
            "low_dim": torch.from_numpy(low_dim.astype(np.float32)),
            "base_policy_action": torch.from_numpy(arrays["base_policy_action"].astype(np.float32)),
            "residual_target": torch.from_numpy(arrays["residual_target"].astype(np.float32)),
            "intervene_mask": torch.tensor(int(arrays["intervene_mask"]), dtype=torch.long),
            "gripper_label": torch.tensor(int(arrays["gripper_label"]), dtype=torch.long),
        }


def compute_low_dim_stats(
    data_path: str | Path,
    refs: Sequence[Stage1SampleRef],
    *,
    low_dim_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    del data_path
    grouped: dict[tuple[str, str], list[int]] = {}
    for ref in refs:
        grouped.setdefault((ref.path, ref.demo), []).append(ref.step)
    values: list[np.ndarray] = []
    for (path, demo), steps in grouped.items():
        with h5py.File(path, "r") as file:
            group = file["data"][demo]
            obs = group["obs"]
            base_policy_actions = _base_policy_actions(group)
            for step in steps:
                values.append(
                    make_low_dim(
                        obs["eef_pos_base"][step],
                        obs["eef_quat_base"][step],
                        obs["joint_pos"][step],
                        obs["gripper_state"][step],
                        base_policy_actions[step],
                        mode=low_dim_mode,
                    )
                )
    if not values:
        raise RuntimeError("cannot compute low-dimensional statistics from an empty split")
    if values[0].size == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    stacked = np.stack(values).astype(np.float32)
    return stacked.mean(axis=0), stacked.std(axis=0) + 1e-6


def collate_batch(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]}
