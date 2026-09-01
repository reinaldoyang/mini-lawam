"""Targeted video access for local LeRobot v3 datasets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LeRobotEpisodeSource:
    """Resolve and decode selected frames from one LeRobot v3 episode."""

    root: Path
    info: dict[str, Any]
    episode_index: int
    length: int
    metadata: dict[str, Any]
    frame_timestamps: np.ndarray

    @classmethod
    def open_all(
        cls,
        dataset_dir: str | os.PathLike[str],
        camera_keys: Iterable[str],
        episode_indices: Sequence[int] | None = None,
    ) -> dict[int, "LeRobotEpisodeSource"]:
        root = Path(dataset_dir).expanduser().resolve()
        info_path = root / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        if not isinstance(info, dict):
            raise TypeError(f"Expected a JSON object in {info_path}")

        fps = float(info.get("fps", 0.0))
        if fps <= 0:
            raise ValueError(f"LeRobot dataset FPS must be positive, got {fps}")

        requested_cameras = list(dict.fromkeys(camera_keys))
        if not requested_cameras:
            raise ValueError("camera_keys must contain at least one video feature")
        features = info.get("features", {})
        if not isinstance(features, dict):
            raise TypeError(f"Expected 'features' to be an object in {info_path}")
        available_cameras = sorted(
            key
            for key, feature in features.items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        )
        unknown = sorted(set(requested_cameras) - set(available_cameras))
        if unknown:
            raise KeyError(
                f"Unknown LeRobot camera key(s) {unknown}; "
                f"available video keys: {available_cameras}"
            )

        episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
        if not episode_files:
            raise FileNotFoundError(
                f"No LeRobot episode metadata found under {root / 'meta/episodes'}"
            )

        import pandas as pd

        columns = ["episode_index", "length"]
        for camera_key in requested_cameras:
            prefix = f"videos/{camera_key}"
            columns.extend(
                [
                    f"{prefix}/chunk_index",
                    f"{prefix}/file_index",
                    f"{prefix}/from_timestamp",
                ]
            )
        metadata = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in episode_files],
            ignore_index=True,
        )
        if metadata.empty:
            raise ValueError(f"LeRobot dataset contains no episodes: {root}")
        if metadata["episode_index"].duplicated().any():
            duplicates = sorted(
                int(value)
                for value in metadata.loc[
                    metadata["episode_index"].duplicated(keep=False), "episode_index"
                ].unique()
            )
            raise ValueError(f"Duplicate episode metadata rows: {duplicates}")

        requested_episodes = None
        if episode_indices is not None:
            requested_episodes = {int(value) for value in episode_indices}
            metadata = metadata[metadata["episode_index"].isin(requested_episodes)]
            found = {int(value) for value in metadata["episode_index"].tolist()}
            missing = sorted(requested_episodes - found)
            if missing:
                raise KeyError(f"Episode index(es) not found in {root}: {missing}")

        data_files = sorted((root / "data").rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No LeRobot data Parquet files found under {root / 'data'}")
        frame_table = pd.concat(
            [
                pd.read_parquet(
                    path,
                    columns=["episode_index", "frame_index", "timestamp"],
                )
                for path in data_files
            ],
            ignore_index=True,
        )
        selected_ids = {int(value) for value in metadata["episode_index"].tolist()}
        frame_table = frame_table[frame_table["episode_index"].isin(selected_ids)]
        frame_table = frame_table.sort_values(["episode_index", "frame_index"])
        timestamps_by_episode = {
            int(episode_index): group["timestamp"].to_numpy(dtype=np.float64)
            for episode_index, group in frame_table.groupby("episode_index", sort=False)
        }

        sources: dict[int, LeRobotEpisodeSource] = {}
        for row in metadata.sort_values("episode_index").to_dict(orient="records"):
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            if length <= 0:
                raise ValueError(
                    f"LeRobot episode {episode_index} has invalid length {length}"
                )
            frame_timestamps = timestamps_by_episode.get(episode_index)
            if frame_timestamps is None or len(frame_timestamps) != length:
                actual = 0 if frame_timestamps is None else len(frame_timestamps)
                raise ValueError(
                    f"Episode {episode_index} metadata length {length} does not match "
                    f"its {actual} timestamp rows"
                )
            sources[episode_index] = cls(
                root=root,
                info=info,
                episode_index=episode_index,
                length=length,
                metadata=row,
                frame_timestamps=frame_timestamps,
            )
        if not sources:
            scope = sorted(requested_episodes) if requested_episodes is not None else "all"
            raise ValueError(f"No LeRobot episodes selected from {root}; scope={scope}")
        return sources

    @classmethod
    def open(
        cls,
        dataset_dir: str | os.PathLike[str],
        episode_index: int | None,
        camera_keys: Iterable[str],
    ) -> "LeRobotEpisodeSource":
        episode_indices = None if episode_index is None else [episode_index]
        sources = cls.open_all(
            dataset_dir,
            camera_keys=camera_keys,
            episode_indices=episode_indices,
        )
        selected_index = min(sources) if episode_index is None else int(episode_index)
        return sources[selected_index]

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    def video_path(self, camera_key: str) -> Path:
        prefix = f"videos/{camera_key}"
        chunk_index = int(self.metadata[f"{prefix}/chunk_index"])
        file_index = int(self.metadata[f"{prefix}/file_index"])
        template = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        try:
            relative_path = str(template).format(
                video_key=camera_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid video_path template in {self.root / 'meta/info.json'}: {template}"
            ) from exc
        path = self.root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Video for episode {self.episode_index}, camera {camera_key!r} "
                f"not found: {path}"
            )
        return path

    def frames(self, camera_key: str, frame_indices: Iterable[int]) -> np.ndarray:
        indices = np.asarray(list(frame_indices), dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("frame_indices must contain at least one frame")
        invalid = indices[(indices < 0) | (indices >= self.length)]
        if invalid.size:
            raise IndexError(
                f"Episode {self.episode_index} frame index {int(invalid[0])} is outside "
                f"the valid range 0..{self.length - 1}"
            )

        prefix = f"videos/{camera_key}"
        start_timestamp = float(self.metadata[f"{prefix}/from_timestamp"])
        timestamps = start_timestamp + self.frame_timestamps[indices]

        # The upstream video helper seeks only the requested timestamps in the
        # shared MP4 shard; it does not materialize the complete episode.
        from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps

        return get_frames_by_timestamps(
            self.video_path(camera_key).as_posix(),
            timestamps,
            video_backend="pyav",
        )
