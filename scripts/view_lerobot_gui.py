#!/usr/bin/env python3
"""Interactive viewer for a local LeRobot v3 dataset.

The viewer reads raw Parquet state/action rows and episode MP4 files directly;
it does not apply training normalization, cropping, or augmentation.

Examples:
    python scripts/view_lerobot_gui.py --dataset dataset/ur_lam_finetune
    python scripts/view_lerobot_gui.py --dataset dataset/ur_lam_finetune --episode 10
    python scripts/view_lerobot_gui.py --dataset dataset/ur_lam_finetune --check

Keyboard controls:
    left/right     previous/next frame
    up/down        previous/next episode
    pageup/pagedown previous/next episode
    home/end       first/last frame
    space          play/pause
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import av
import numpy as np
import pandas as pd


STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required LeRobot metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _read_parquet_tree(path: Path, description: str) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No {description} Parquet files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _to_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _jsonable_vector(value: Any) -> list[float]:
    return [float(item) for item in _to_vector(value)]


@dataclass
class EpisodeData:
    episode_id: int
    rows: pd.DataFrame
    videos: dict[str, list[np.ndarray]]
    video_paths: dict[str, Path]
    task: str

    @property
    def length(self) -> int:
        return len(self.rows)


class LeRobotDatasetReader:
    """Read the metadata, tabular rows, and episode videos of LeRobot v3 data."""

    def __init__(self, dataset_dir: str | Path, camera_keys: Iterable[str] | None = None):
        self.root = Path(dataset_dir).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset directory not found: {self.root}")

        self.info = _read_json(self.root / "meta/info.json")
        self.features = self.info.get("features", {})
        if not isinstance(self.features, dict):
            raise TypeError("meta/info.json field 'features' must be an object")

        self.fps = float(self.info.get("fps", 0.0))
        if self.fps <= 0:
            raise ValueError(f"Dataset FPS must be positive, got {self.fps}")

        self.episodes = _read_parquet_tree(self.root / "meta/episodes", "episode")
        self.data = _read_parquet_tree(self.root / "data", "data")
        for table_name, table in (("episodes", self.episodes), ("data", self.data)):
            if "episode_index" not in table.columns:
                raise KeyError(f"{table_name} table has no 'episode_index' column")

        self.episodes = self.episodes.sort_values("episode_index").reset_index(drop=True)
        self.data = self.data.sort_values(
            [column for column in ("episode_index", "frame_index", "index") if column in self.data]
        ).reset_index(drop=True)

        episode_ids = [int(value) for value in self.episodes["episode_index"].tolist()]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Episode metadata contains duplicate episode_index values")
        if not episode_ids:
            raise ValueError("Dataset contains no episodes")
        self.episode_ids = episode_ids
        self._episode_row_by_id = {
            int(row["episode_index"]): row for _, row in self.episodes.iterrows()
        }

        detected_video_keys = [
            key
            for key, feature in self.features.items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        ]
        if not detected_video_keys:
            for column in self.episodes.columns:
                if column.startswith("videos/") and column.endswith("/from_timestamp"):
                    detected_video_keys.append(
                        column[len("videos/") : -len("/from_timestamp")]
                    )
        detected_video_keys = sorted(set(detected_video_keys))
        if not detected_video_keys:
            raise ValueError("Dataset metadata contains no video features")

        requested = list(camera_keys or [])
        unknown = sorted(set(requested) - set(detected_video_keys))
        if unknown:
            raise KeyError(
                f"Unknown camera key(s) {unknown}; available keys: {detected_video_keys}"
            )
        self.video_keys = requested or detected_video_keys

        self.state_key = STATE_KEY if STATE_KEY in self.data.columns else self._find_vector_key("state")
        self.action_key = ACTION_KEY if ACTION_KEY in self.data.columns else self._find_vector_key("action")
        self.state_names = self._feature_names(self.state_key)
        self.action_names = self._feature_names(self.action_key)
        self.tasks = self._load_tasks()

    def _find_vector_key(self, kind: str) -> str | None:
        candidates = [key for key in self.features if kind in key.lower() and key in self.data.columns]
        return candidates[0] if candidates else None

    def _feature_names(self, feature_key: str | None) -> list[str]:
        if feature_key is None:
            return []
        feature = self.features.get(feature_key, {})
        names = feature.get("names", []) if isinstance(feature, dict) else []
        if not isinstance(names, list):
            return []
        return [str(name) for name in names]

    def _load_tasks(self) -> dict[int, str]:
        path = self.root / "meta/tasks.parquet"
        if not path.is_file():
            return {}
        table = pd.read_parquet(path)
        if not {"task_index", "task"}.issubset(table.columns):
            return {}
        return {
            int(row["task_index"]): str(row["task"])
            for _, row in table.iterrows()
        }

    def _episode_task(self, rows: pd.DataFrame, episode_row: pd.Series) -> str:
        if "task_index" in rows.columns and len(rows):
            task_index = int(rows.iloc[0]["task_index"])
            if task_index in self.tasks:
                return self.tasks[task_index]
        tasks = episode_row.get("tasks")
        if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks):
            return str(tasks[0])
        return "unknown"

    def _video_path(self, episode_row: pd.Series, video_key: str) -> Path:
        prefix = f"videos/{video_key}"
        chunk_column = f"{prefix}/chunk_index"
        file_column = f"{prefix}/file_index"
        chunk_index = int(episode_row.get(chunk_column, 0))
        file_index = int(episode_row.get(file_column, episode_row["episode_index"]))
        template = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        try:
            relative = str(template).format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid video_path template in meta/info.json: {template}") from exc
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Video for episode {int(episode_row['episode_index'])}, "
                f"camera {video_key} not found: {path}"
            )
        return path

    def _decode_episode_video(
        self,
        path: Path,
        *,
        start_timestamp: float,
        end_timestamp: float,
        expected_frames: int,
    ) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        tolerance = 0.51 / self.fps
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in {path}")
            stream = container.streams.video[0]
            for decoded_index, frame in enumerate(container.decode(stream)):
                if frame.pts is not None and frame.time_base is not None:
                    timestamp = float(frame.pts * frame.time_base)
                else:
                    timestamp = decoded_index / self.fps
                if timestamp < start_timestamp - tolerance:
                    continue
                if timestamp > end_timestamp + tolerance and len(frames) >= expected_frames:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == expected_frames:
                    break

        if len(frames) != expected_frames:
            raise ValueError(
                f"Decoded {len(frames)} frames from {path}, expected {expected_frames} "
                f"between {start_timestamp:.6f}s and {end_timestamp:.6f}s"
            )
        return frames

    def load_episode(self, episode_id: int, *, decode_videos: bool = True) -> EpisodeData:
        if episode_id not in self._episode_row_by_id:
            raise KeyError(
                f"Episode {episode_id} not found; range is "
                f"{self.episode_ids[0]}..{self.episode_ids[-1]}"
            )
        episode_row = self._episode_row_by_id[episode_id]
        rows = self.data[self.data["episode_index"] == episode_id].copy()
        if "frame_index" in rows.columns:
            rows = rows.sort_values("frame_index")
        rows = rows.reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"Episode {episode_id} has metadata but no data rows")

        metadata_length = int(episode_row.get("length", len(rows)))
        if metadata_length != len(rows):
            raise ValueError(
                f"Episode {episode_id} metadata length {metadata_length} "
                f"does not match {len(rows)} data rows"
            )

        videos: dict[str, list[np.ndarray]] = {}
        video_paths: dict[str, Path] = {}
        if decode_videos:
            for video_key in self.video_keys:
                path = self._video_path(episode_row, video_key)
                prefix = f"videos/{video_key}"
                start = float(episode_row.get(f"{prefix}/from_timestamp", 0.0))
                default_end = max(0.0, (len(rows) - 1) / self.fps)
                end = float(episode_row.get(f"{prefix}/to_timestamp", default_end))
                video_paths[video_key] = path
                videos[video_key] = self._decode_episode_video(
                    path,
                    start_timestamp=start,
                    end_timestamp=end,
                    expected_frames=len(rows),
                )

        return EpisodeData(
            episode_id=episode_id,
            rows=rows,
            videos=videos,
            video_paths=video_paths,
            task=self._episode_task(rows, episode_row),
        )

    def summary(self, episode_id: int, *, decode_videos: bool = True) -> dict[str, Any]:
        episode = self.load_episode(episode_id, decode_videos=decode_videos)
        first_row = episode.rows.iloc[0]
        return {
            "dataset": str(self.root),
            "codebase_version": self.info.get("codebase_version"),
            "fps": self.fps,
            "total_episodes": len(self.episode_ids),
            "total_frames": int(self.info.get("total_frames", len(self.data))),
            "video_keys": self.video_keys,
            "episode": episode_id,
            "episode_frames": episode.length,
            "task": episode.task,
            "decoded_video_shapes": {
                key: list(frames[0].shape) if frames else None
                for key, frames in episode.videos.items()
            },
            "state_key": self.state_key,
            "state_names": self.state_names,
            "first_state": (
                _jsonable_vector(first_row[self.state_key]) if self.state_key else []
            ),
            "action_key": self.action_key,
            "action_names": self.action_names,
            "first_action": (
                _jsonable_vector(first_row[self.action_key]) if self.action_key else []
            ),
        }


class LeRobotViewer:
    def __init__(
        self,
        reader: LeRobotDatasetReader,
        *,
        initial_episode: int,
        playback_fps: float | None = None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, TextBox

        self.plt = plt
        self.Button = Button
        self.Slider = Slider
        self.TextBox = TextBox
        self.reader = reader
        self.episode_position = self.reader.episode_ids.index(initial_episode)
        self.episode = self.reader.load_episode(initial_episode)
        self.frame_index = 0
        self.playing = False
        self.playback_fps = float(playback_fps or self.reader.fps)
        if self.playback_fps <= 0:
            raise ValueError(f"playback_fps must be positive, got {self.playback_fps}")

        self._build_figure()
        self.timer = self.fig.canvas.new_timer(
            interval=max(1, int(round(1000.0 / self.playback_fps)))
        )
        self.timer.add_callback(self._timer_tick)
        self._draw()

    def _build_figure(self) -> None:
        camera_count = max(1, len(self.reader.video_keys))
        self.fig = self.plt.figure(figsize=(5.2 * camera_count + 5.2, 7.2))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(f"LeRobot Viewer - {self.reader.root.name}")
        grid = self.fig.add_gridspec(
            1,
            camera_count + 1,
            width_ratios=[5] * camera_count + [4.5],
        )
        self.image_axes = []
        self.image_artists = []
        for index in range(camera_count):
            axis = self.fig.add_subplot(grid[0, index])
            axis.axis("off")
            self.image_axes.append(axis)
            self.image_artists.append(None)

        self.text_axis = self.fig.add_subplot(grid[0, camera_count])
        self.text_axis.axis("off")
        self.text_artist = self.text_axis.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            transform=self.text_axis.transAxes,
        )
        self.fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.22, wspace=0.12)

        slider_axis = self.fig.add_axes([0.08, 0.135, 0.62, 0.035])
        self.slider = self.Slider(
            slider_axis,
            "frame",
            0,
            max(1, self.episode.length - 1),
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider)

        episode_axis = self.fig.add_axes([0.79, 0.13, 0.11, 0.05])
        self.episode_box = self.TextBox(
            episode_axis,
            "episode ",
            initial=str(self.episode.episode_id),
        )
        self.episode_box.on_submit(self._on_episode_submit)

        button_specs = [
            ("<< episode", self._previous_episode),
            ("< frame", self._previous_frame),
            ("Play", self._toggle_play),
            ("frame >", self._next_frame),
            ("episode >>", self._next_episode),
        ]
        self.buttons = []
        button_width = 0.13
        button_gap = 0.015
        start_x = 0.08
        for index, (label, callback) in enumerate(button_specs):
            axis = self.fig.add_axes(
                [start_x + index * (button_width + button_gap), 0.045, button_width, 0.055]
            )
            button = self.Button(axis, label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.play_button = self.buttons[2]

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _row(self) -> pd.Series:
        return self.episode.rows.iloc[self.frame_index]

    def _format_vector(self, title: str, value: Any, names: list[str]) -> list[str]:
        vector = _to_vector(value)
        lines = [f"{title}:"]
        for index, item in enumerate(vector):
            name = names[index] if index < len(names) else str(index)
            lines.append(f"  {name:<25} {float(item):+10.5f}")
        if vector.size == 0:
            lines.append("  <not available>")
        return lines

    def _format_text(self) -> str:
        row = self._row()
        timestamp = float(row.get("timestamp", self.frame_index / self.reader.fps))
        dataset_index = row.get("index", "n/a")
        lines = [
            f"dataset : {self.reader.root.name}",
            (
                f"episode: {self.episode.episode_id} "
                f"({self.episode_position + 1}/{len(self.reader.episode_ids)})"
            ),
            f"frame  : {self.frame_index}/{self.episode.length - 1}",
            f"time   : {timestamp:.3f} s",
            f"index  : {dataset_index}",
            f"task   : {self.episode.task}",
            "",
        ]
        if self.reader.state_key:
            lines.extend(
                self._format_vector(
                    "state",
                    row[self.reader.state_key],
                    self.reader.state_names,
                )
            )
        lines.append("")
        if self.reader.action_key:
            lines.extend(
                self._format_vector(
                    "action",
                    row[self.reader.action_key],
                    self.reader.action_names,
                )
            )
        return "\n".join(lines)

    def _draw(self, *, sync_slider: bool = True) -> None:
        for index, video_key in enumerate(self.reader.video_keys):
            frame = self.episode.videos[video_key][self.frame_index]
            axis = self.image_axes[index]
            artist = self.image_artists[index]
            if artist is None:
                self.image_artists[index] = axis.imshow(frame)
            else:
                artist.set_data(frame)
            axis.set_title(video_key, fontsize=10)
        self.text_artist.set_text(self._format_text())
        self.fig.suptitle(
            f"{self.reader.root.name} — episode {self.episode.episode_id}, "
            f"frame {self.frame_index}",
            fontsize=12,
        )
        if sync_slider:
            self.slider.eventson = False
            self.slider.set_val(self.frame_index)
            self.slider.eventson = True
        self.fig.canvas.draw_idle()

    def _on_slider(self, value: float) -> None:
        self.frame_index = min(int(value), self.episode.length - 1)
        self._draw(sync_slider=False)

    def _set_frame(self, frame_index: int) -> None:
        self.frame_index = max(0, min(int(frame_index), self.episode.length - 1))
        self._draw()

    def _next_frame(self, _event: Any = None) -> None:
        self._set_frame(self.frame_index + 1)

    def _previous_frame(self, _event: Any = None) -> None:
        self._set_frame(self.frame_index - 1)

    def _load_episode_position(self, position: int) -> None:
        self._stop_playback()
        position %= len(self.reader.episode_ids)
        episode_id = self.reader.episode_ids[position]
        try:
            episode = self.reader.load_episode(episode_id)
        except Exception as exc:
            print(f"Could not load episode {episode_id}: {exc}", file=sys.stderr)
            return
        self.episode_position = position
        self.episode = episode
        self.frame_index = 0
        self.slider.valmax = max(1, episode.length - 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.episode_box.eventson = False
        self.episode_box.set_val(str(episode_id))
        self.episode_box.eventson = True
        self._draw()

    def _next_episode(self, _event: Any = None) -> None:
        self._load_episode_position(self.episode_position + 1)

    def _previous_episode(self, _event: Any = None) -> None:
        self._load_episode_position(self.episode_position - 1)

    def _on_episode_submit(self, value: str) -> None:
        try:
            episode_id = int(value.strip())
            position = self.reader.episode_ids.index(episode_id)
        except (ValueError, TypeError):
            print(
                f"Invalid episode {value!r}; available range is "
                f"{self.reader.episode_ids[0]}..{self.reader.episode_ids[-1]}",
                file=sys.stderr,
            )
            self.episode_box.eventson = False
            self.episode_box.set_val(str(self.episode.episode_id))
            self.episode_box.eventson = True
            return
        self._load_episode_position(position)

    def _toggle_play(self, _event: Any = None) -> None:
        if self.playing:
            self._stop_playback()
        else:
            self.playing = True
            self.play_button.label.set_text("Pause")
            self.timer.start()
            self.fig.canvas.draw_idle()

    def _stop_playback(self) -> None:
        if hasattr(self, "timer"):
            self.timer.stop()
        self.playing = False
        if hasattr(self, "play_button"):
            self.play_button.label.set_text("Play")

    def _timer_tick(self) -> None:
        if not self.playing:
            return
        if self.frame_index >= self.episode.length - 1:
            self._stop_playback()
            self.fig.canvas.draw_idle()
            return
        self._set_frame(self.frame_index + 1)

    def _on_key(self, event: Any) -> None:
        if event.key == "right":
            self._next_frame()
        elif event.key == "left":
            self._previous_frame()
        elif event.key in {"up", "pageup"}:
            self._previous_episode()
        elif event.key in {"down", "pagedown"}:
            self._next_episode()
        elif event.key == "home":
            self._set_frame(0)
        elif event.key == "end":
            self._set_frame(self.episode.length - 1)
        elif event.key in {" ", "space"}:
            self._toggle_play()

    def _on_close(self, _event: Any) -> None:
        self._stop_playback()

    def show(self) -> None:
        self.plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="dataset/ur_lam_finetune",
        help="Path to a local LeRobot v3 dataset directory.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Initial episode_index. Defaults to the first episode.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Video feature key to show. Repeat for multiple cameras; defaults to all.",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=None,
        help="GUI playback speed. Defaults to the dataset FPS.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Decode the selected episode, print a JSON summary, and exit without a GUI.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reader = LeRobotDatasetReader(args.dataset, camera_keys=args.camera)
    episode_id = args.episode if args.episode is not None else reader.episode_ids[0]
    if episode_id not in reader.episode_ids:
        raise ValueError(
            f"Episode {episode_id} not found; available range is "
            f"{reader.episode_ids[0]}..{reader.episode_ids[-1]}"
        )

    if args.check:
        print(json.dumps(reader.summary(episode_id, decode_videos=True), indent=2))
        return

    viewer = LeRobotViewer(
        reader,
        initial_episode=episode_id,
        playback_fps=args.playback_fps,
    )
    viewer.show()


if __name__ == "__main__":
    main()
