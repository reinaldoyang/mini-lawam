"""In-memory episode buffer and append-only HDF5 writer for HIL corrections."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .constants import ACTION_DIM, ACTION_MEANING, ACTION_SCHEMA, GRIPPER_LABEL_MEANING


OBS_KEYS = (
    "table_cam",
    "wrist_cam",
    "eef_pos_base",
    "eef_quat_base",
    "joint_pos",
    "gripper_state",
    "base_policy_action",
    "quest_controller",
)


class CorrectionEpisode:
    """One episode buffered in memory until the operator saves it."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.obs: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
        self.base_policy_actions: list[np.ndarray] = []
        self.human_delta_actions: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.residual_targets: list[np.ndarray] = []
        self.manual_control_mask: list[bool] = []
        self.intervene_mask: list[bool] = []
        self.gripper_labels: list[int] = []
        self.timestamps: list[float] = []

    def __len__(self) -> int:
        return len(self.base_policy_actions)

    def append(
        self,
        *,
        obs: Mapping[str, np.ndarray],
        base_action: Sequence[float],
        human_action: Sequence[float],
        executed_action: Sequence[float],
        residual_target: Sequence[float],
        manual_control: bool,
        intervention: bool,
        gripper_label: int,
        timestamp: float,
    ) -> None:
        missing = sorted(set(OBS_KEYS) - set(obs))
        if missing:
            raise ValueError(f"observation is missing keys: {missing}")
        actions = {
            "base_action": np.asarray(base_action, dtype=np.float32),
            "human_action": np.asarray(human_action, dtype=np.float32),
            "executed_action": np.asarray(executed_action, dtype=np.float32),
            "residual_target": np.asarray(residual_target, dtype=np.float32),
        }
        for name, value in actions.items():
            if value.shape != (ACTION_DIM,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite ({ACTION_DIM},) array, got {value!r}")
        for key in OBS_KEYS:
            value = np.asarray(obs[key])
            if not np.all(np.isfinite(value)):
                raise ValueError(f"observation {key} contains non-finite values")
            self.obs[key].append(value.copy())
        self.base_policy_actions.append(actions["base_action"])
        self.human_delta_actions.append(actions["human_action"])
        self.executed_actions.append(actions["executed_action"])
        self.residual_targets.append(actions["residual_target"])
        self.manual_control_mask.append(bool(manual_control))
        self.intervene_mask.append(bool(intervention))
        label = int(gripper_label)
        if label not in (0, 1):
            raise ValueError(f"gripper_label must be binary OPEN=0 or CLOSE=1, got {label}")
        self.gripper_labels.append(label)
        self.timestamps.append(float(timestamp))

    def arrays(self) -> dict[str, object]:
        if not len(self):
            raise RuntimeError("episode is empty")
        return {
            "obs": {key: np.stack(values, axis=0) for key, values in self.obs.items()},
            "base_policy_actions": np.stack(self.base_policy_actions, axis=0),
            "human_delta_actions": np.stack(self.human_delta_actions, axis=0),
            "executed_actions": np.stack(self.executed_actions, axis=0),
            "residual_targets": np.stack(self.residual_targets, axis=0),
            "manual_control_mask": np.asarray(self.manual_control_mask, dtype=np.bool_),
            "intervene_mask": np.asarray(self.intervene_mask, dtype=np.bool_),
            "gripper_labels": np.asarray(self.gripper_labels, dtype=np.int64),
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
        }


class CorrectionHDF5Writer:
    """Append complete correction episodes using the legacy gated-residual keys."""

    def __init__(self, path: str | Path, args, policy) -> None:
        import h5py

        self.h5py = h5py
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        requested = str(args.hdf5_compression).lower()
        self.compression = None if requested == "none" else requested
        self.write_batch_size = int(args.hdf5_write_batch_size)
        self.image_shape = (int(args.image_height), int(args.image_width), 3)
        self.file = h5py.File(self.path, "a")
        self.data = self.file.require_group("data")
        meta = self.file.require_group("meta")

        existing_gripper_labels = meta.attrs.get("gripper_labels")
        if len(self.data) and existing_gripper_labels != GRIPPER_LABEL_MEANING:
            self.file.close()
            raise ValueError(
                "cannot append binary gripper labels to an existing dataset with "
                f"gripper_labels={existing_gripper_labels!r}; write a new output file"
            )

        identity = {
            "dataset_type": "hil_gated_residual",
            "schema": ACTION_SCHEMA,
            "base_policy": "mini_lawam",
            "base_policy_checkpoint": str(Path(args.ckpt).resolve()),
            "base_policy_target_mode": policy.target_mode,
            "base_policy_include_rz": bool(policy.include_rz),
            "table_camera_serial": str(args.table_cam_serial),
            "wrist_camera_serial": str(args.wrist_cam_serial),
            "image_height": self.image_shape[0],
            "image_width": self.image_shape[1],
        }
        for key, expected in identity.items():
            if key in meta.attrs and meta.attrs[key] != expected:
                self.file.close()
                raise ValueError(
                    f"existing dataset metadata {key}={meta.attrs[key]!r} does not match {expected!r}"
                )
            meta.attrs[key] = expected
        if "created_unix_time" not in meta.attrs:
            meta.attrs["created_unix_time"] = time.time()
        self.args_json = json.dumps(vars(args), sort_keys=True, default=str)
        meta.attrs["args_json"] = self.args_json
        meta.attrs["action_meaning"] = ACTION_MEANING
        meta.attrs["action_alignment"] = "obs[k] -> forward command action[k]"
        meta.attrs["base_policy_actions_meaning"] = (
            "safe, post-processed Mini-LaWAM action at the same observation"
        )
        meta.attrs["bc_actions_meaning"] = "compatibility alias of base_policy_actions"
        meta.attrs["executed_actions_meaning"] = (
            "VR action while side grip is held, otherwise base_policy_actions"
        )
        meta.attrs["intervene_mask_meaning"] = "Quest side-grip held; explicit full-control takeover"
        meta.attrs["manual_control_mask_meaning"] = "same as intervene_mask for momentary VR takeover"
        meta.attrs["gripper_label_schema"] = "binary_executed_state_v1"
        meta.attrs["gripper_labels"] = GRIPPER_LABEL_MEANING
        meta.attrs["gripper_encoding"] = "open=-1, close=+1"
        self._update_totals()
        self.file.flush()

    def _options(self, *, chunks=True) -> dict[str, object]:
        options: dict[str, object] = {"chunks": chunks}
        if self.compression is not None:
            options["compression"] = self.compression
        return options

    def _next_demo_name(self) -> str:
        index = 0
        while f"demo_{index}" in self.data:
            index += 1
        return f"demo_{index}"

    def _write_images(self, group, name: str, values: np.ndarray) -> None:
        if values.ndim != 4 or tuple(values.shape[1:]) != self.image_shape:
            raise ValueError(f"{name} must have shape (N,{self.image_shape}), got {values.shape}")
        dataset = group.create_dataset(
            name,
            shape=values.shape,
            dtype=np.uint8,
            **self._options(chunks=(1, *self.image_shape)),
        )
        for start in range(0, values.shape[0], self.write_batch_size):
            end = min(values.shape[0], start + self.write_batch_size)
            dataset[start:end] = values[start:end]

    def write_episode(self, episode: CorrectionEpisode, *, outcome: str = "saved") -> tuple[str, int, int]:
        arrays = episode.arrays()
        sample_count = int(arrays["base_policy_actions"].shape[0])
        name = self._next_demo_name()
        previous_total = int(self.data.attrs.get("total", 0))
        group = self.data.create_group(name)
        try:
            obs_group = group.create_group("obs")
            obs = arrays["obs"]
            self._write_images(obs_group, "table_cam", obs["table_cam"])
            self._write_images(obs_group, "wrist_cam", obs["wrist_cam"])
            for key in OBS_KEYS:
                if key in ("table_cam", "wrist_cam"):
                    continue
                obs_group.create_dataset(key, data=obs[key], **self._options())
            for key in (
                "base_policy_actions",
                "human_delta_actions",
                "executed_actions",
                "residual_targets",
                "manual_control_mask",
                "intervene_mask",
                "gripper_labels",
                "timestamps",
            ):
                group.create_dataset(key, data=arrays[key], **self._options())
            # Compatibility aliases without duplicating data on disk.
            group["bc_actions"] = group["base_policy_actions"]
            group["base_actions"] = group["base_policy_actions"]
            group["actions"] = group["executed_actions"]
            obs_group["bc_action"] = obs_group["base_policy_action"]
            interventions = int(np.asarray(arrays["intervene_mask"]).sum())
            group.attrs["num_samples"] = sample_count
            group.attrs["num_interventions"] = interventions
            group.attrs["num_manual_control"] = int(np.asarray(arrays["manual_control_mask"]).sum())
            group.attrs["outcome"] = str(outcome)
            group.attrs["created_unix_time"] = time.time()
            group.attrs["args_json"] = self.args_json
            self._update_totals()
            self.file.flush()
        except BaseException:
            if name in self.data:
                del self.data[name]
            self.data.attrs["total"] = previous_total
            self.file.flush()
            raise
        return name, sample_count, interventions

    def _update_totals(self) -> None:
        total = manual = interventions = 0
        for name in self.data:
            group = self.data.get(name)
            if not isinstance(group, self.h5py.Group):
                continue
            action_key = "base_policy_actions" if "base_policy_actions" in group else "bc_actions"
            if action_key not in group:
                continue
            total += int(group[action_key].shape[0])
            if "manual_control_mask" in group:
                manual += int(np.asarray(group["manual_control_mask"]).astype(bool).sum())
            if "intervene_mask" in group:
                interventions += int(np.asarray(group["intervene_mask"]).astype(bool).sum())
        self.data.attrs["total"] = total
        self.data.attrs["total_manual_control"] = manual
        self.data.attrs["total_interventions"] = interventions

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass
