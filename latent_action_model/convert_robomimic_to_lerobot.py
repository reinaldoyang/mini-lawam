"""Convert this project's robomimic HDF5 recordings to LeRobot v3 for Stage 1.

The converter preserves the table-camera RGB stream, converts the recorded EEF
quaternion to an axis-angle rotation vector, and emits the seven-dimensional UR
state expected by ``robomind_ur_1rgb``:

    [eef_position_xyz, eef_orientation_rotvec, gripper]

The source recordings do not contain measured gripper position. The latched
gripper command from ``actions[:, 6]`` is therefore used as a proxy state and is
called out in the generated dataset metadata.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Iterable

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


STATE_DIM = 7
ACTION_DIM = 7
VIDEO_FEATURE_KEY = "observation.images.table_cam"
STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"


def _natural_demo_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _validate_rgb_frames(frames: np.ndarray, *, demo_name: str) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"{demo_name}: expected RGB frames [T,H,W,3], got {frames.shape}"
        )
    if frames.dtype != np.uint8:
        raise TypeError(f"{demo_name}: expected uint8 RGB frames, got {frames.dtype}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    if height % 2 or width % 2:
        raise ValueError(
            f"{demo_name}: H.264 yuv420p requires even image dimensions, got {height}x{width}"
        )
    return np.ascontiguousarray(frames)


def _quaternion_to_rotvec(quaternions: np.ndarray, order: str) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"expected quaternions [T,4], got {quaternions.shape}")
    if not np.isfinite(quaternions).all():
        raise ValueError("quaternion array contains NaN or Inf")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("quaternion array contains a near-zero quaternion")
    normalized = quaternions / norms
    if order == "wxyz":
        normalized = normalized[:, [1, 2, 3, 0]]
    elif order != "xyzw":
        raise ValueError(f"unsupported quaternion order: {order}")
    return Rotation.from_quat(normalized).as_rotvec().astype(np.float32)


def _encode_rgb_video(
    frames: np.ndarray,
    output_path: Path,
    *,
    fps: float,
    codec: str,
    crf: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    rate = Fraction(str(fps)).limit_denominator(100_000)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.gop_size = max(1, int(round(fps)))
        stream.options = {"crf": str(crf), "preset": "medium"}
        for frame_index, frame_rgb in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            video_frame.pts = frame_index
            video_frame.time_base = Fraction(1, rate)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _write_data_parquet(path: Path, rows: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "index": pa.array(rows["index"], type=pa.int64()),
            "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
            "frame_index": pa.array(rows["frame_index"], type=pa.int64()),
            "timestamp": pa.array(rows["timestamp"], type=pa.float64()),
            "task_index": pa.array(rows["task_index"], type=pa.int64()),
            STATE_FEATURE_KEY: pa.array(
                rows[STATE_FEATURE_KEY], type=pa.list_(pa.float32(), STATE_DIM)
            ),
            ACTION_FEATURE_KEY: pa.array(
                rows[ACTION_FEATURE_KEY], type=pa.list_(pa.float32(), ACTION_DIM)
            ),
        }
    )
    pq.write_table(table, path, compression="zstd")


def _feature_metadata(height: int, width: int, fps: float) -> dict:
    video_info = {
        "video.fps": fps,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    return {
        STATE_FEATURE_KEY: {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": [
                "eef_x",
                "eef_y",
                "eef_z",
                "rotvec_x",
                "rotvec_y",
                "rotvec_z",
                "gripper_command_proxy",
            ],
        },
        ACTION_FEATURE_KEY: {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": [
                "command_x",
                "command_y",
                "command_z",
                "command_rx",
                "command_ry",
                "command_rz",
                "gripper_command",
            ],
            "fps": fps,
        },
        VIDEO_FEATURE_KEY: {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "video_info": video_info,
            "info": video_info,
        },
        "timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _modality_metadata() -> dict:
    return {
        "state": {
            "end_effector_position": {
                "start": 0,
                "end": 3,
                "absolute": True,
                "dtype": "float32",
                "original_key": STATE_FEATURE_KEY,
            },
            "eef_orientation_rotvec": {
                "start": 3,
                "end": 6,
                "rotation_type": "axis_angle",
                "absolute": True,
                "dtype": "float32",
                "original_key": STATE_FEATURE_KEY,
            },
            "gripper": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": STATE_FEATURE_KEY,
            },
        },
        "action": {
            "command_translation": {
                "start": 0,
                "end": 3,
                "absolute": False,
                "dtype": "float32",
                "original_key": ACTION_FEATURE_KEY,
            },
            "command_rotation": {
                "start": 3,
                "end": 6,
                "rotation_type": "axis_angle",
                "absolute": False,
                "dtype": "float32",
                "original_key": ACTION_FEATURE_KEY,
            },
            "gripper_command": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": ACTION_FEATURE_KEY,
            },
        },
        "video": {
            "camera_top": {"original_key": VIDEO_FEATURE_KEY},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def convert_dataset(
    source_hdf5: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 20.0,
    task: str = "robot manipulation",
    camera_key: str = "table_cam",
    position_key: str = "eef_pos_base",
    quaternion_key: str = "eef_quat_base",
    quaternion_order: str = "xyzw",
    gripper_action_col: int = 6,
    codec: str = "libx264",
    crf: int = 18,
    max_episodes: int | None = None,
) -> dict[str, int | float | str]:
    """Convert one robomimic HDF5 file into a new LeRobot v3 directory."""
    source_path = Path(source_hdf5).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source HDF5 not found: {source_path}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}")
    if destination.exists():
        raise FileExistsError(
            f"Output path already exists: {destination}. Choose a new path so source "
            "and previous conversions cannot be overwritten."
        )
    destination.mkdir(parents=True)

    rows: dict[str, list] = {
        "index": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "task_index": [],
        STATE_FEATURE_KEY: [],
        ACTION_FEATURE_KEY: [],
    }
    episode_rows: list[dict] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    expected_hw: tuple[int, int] | None = None
    global_index = 0

    with h5py.File(source_path, "r") as source:
        if "data" not in source:
            raise KeyError(f"{source_path} does not contain a 'data' group")
        demo_names = sorted(source["data"].keys(), key=_natural_demo_key)
        if max_episodes is not None:
            demo_names = demo_names[:max_episodes]
        if not demo_names:
            raise ValueError(f"{source_path} contains no demos")

        for episode_index, demo_name in enumerate(demo_names):
            demo = source["data"][demo_name]
            required_paths = (
                f"obs/{camera_key}",
                f"obs/{position_key}",
                f"obs/{quaternion_key}",
                "actions",
            )
            missing = [key for key in required_paths if key not in demo]
            if missing:
                raise KeyError(f"{demo_name}: missing required datasets {missing}")

            frames = _validate_rgb_frames(demo[f"obs/{camera_key}"][...], demo_name=demo_name)
            positions = np.asarray(demo[f"obs/{position_key}"][...], dtype=np.float32)
            rotvec = _quaternion_to_rotvec(
                demo[f"obs/{quaternion_key}"][...], quaternion_order
            )
            actions = np.asarray(demo["actions"][...], dtype=np.float32)
            length = int(frames.shape[0])
            if positions.shape != (length, 3):
                raise ValueError(f"{demo_name}: position shape {positions.shape} != ({length}, 3)")
            if rotvec.shape != (length, 3):
                raise ValueError(f"{demo_name}: rotvec shape {rotvec.shape} != ({length}, 3)")
            if actions.ndim != 2 or actions.shape[0] != length or actions.shape[1] < ACTION_DIM:
                raise ValueError(
                    f"{demo_name}: actions must be [T,>=7] with T={length}, got {actions.shape}"
                )
            if not 0 <= gripper_action_col < actions.shape[1]:
                raise ValueError(
                    f"{demo_name}: gripper action column {gripper_action_col} outside "
                    f"action width {actions.shape[1]}"
                )

            hw = (int(frames.shape[1]), int(frames.shape[2]))
            if expected_hw is None:
                expected_hw = hw
            elif hw != expected_hw:
                raise ValueError(
                    f"all episodes must share one resolution; {demo_name} has {hw}, "
                    f"expected {expected_hw}"
                )

            state = np.concatenate(
                [positions, rotvec, actions[:, gripper_action_col : gripper_action_col + 1]],
                axis=1,
            ).astype(np.float32)
            action = actions[:, :ACTION_DIM].astype(np.float32)
            all_states.append(state)
            all_actions.append(action)

            video_relative = Path(
                f"videos/{VIDEO_FEATURE_KEY}/chunk-000/file-{episode_index:03d}.mp4"
            )
            _encode_rgb_video(
                frames,
                destination / video_relative,
                fps=fps,
                codec=codec,
                crf=crf,
            )

            episode_from_index = global_index
            for frame_index in range(length):
                rows["index"].append(global_index)
                rows["episode_index"].append(episode_index)
                rows["frame_index"].append(frame_index)
                rows["timestamp"].append(frame_index / fps)
                rows["task_index"].append(0)
                rows[STATE_FEATURE_KEY].append(state[frame_index].tolist())
                rows[ACTION_FEATURE_KEY].append(action[frame_index].tolist())
                global_index += 1

            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "length": length,
                    "tasks": [task],
                    "dataset_from_index": episode_from_index,
                    "dataset_to_index": global_index,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "data/from_index": episode_from_index,
                    "data/to_index": global_index,
                    f"videos/{VIDEO_FEATURE_KEY}/chunk_index": 0,
                    f"videos/{VIDEO_FEATURE_KEY}/file_index": episode_index,
                    f"videos/{VIDEO_FEATURE_KEY}/from_timestamp": 0.0,
                    f"videos/{VIDEO_FEATURE_KEY}/to_timestamp": max(0.0, (length - 1) / fps),
                }
            )
            print(
                f"[{episode_index + 1}/{len(demo_names)}] {demo_name}: "
                f"{length} frames -> {video_relative}"
            )

    assert expected_hw is not None
    height, width = expected_hw
    _write_data_parquet(destination / "data/chunk-000/file-000.parquet", rows)
    episodes_path = destination / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)
    tasks_path = destination / "meta/tasks.parquet"
    pd.DataFrame([{"task_index": 0, "task": task}]).to_parquet(tasks_path, index=False)

    state_values = np.concatenate(all_states, axis=0)
    action_values = np.concatenate(all_actions, axis=0)
    _write_json(
        destination / "meta/stats_gr00t.json",
        {
            STATE_FEATURE_KEY: _stats(state_values),
            ACTION_FEATURE_KEY: _stats(action_values),
        },
    )
    _write_json(destination / "meta/modality.json", _modality_metadata())
    _write_json(
        destination / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": "ur",
            "fps": fps,
            "total_episodes": len(episode_rows),
            "total_frames": global_index,
            "total_tasks": 1,
            "total_videos": len(episode_rows),
            "total_chunks": 1,
            "chunks_size": 1000,
            "splits": {"train": f"0:{len(episode_rows)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": (
                "videos/{video_key}/chunk-{chunk_index:03d}/"
                "file-{file_index:03d}.mp4"
            ),
            "features": _feature_metadata(height, width, fps),
            "source": {
                "format": "robomimic_hdf5",
                "path": str(source_path),
                "camera_key": camera_key,
                "position_key": position_key,
                "quaternion_key": quaternion_key,
                "quaternion_order": quaternion_order,
                "gripper_state": (
                    f"proxy from actions[:, {gripper_action_col}]; measured gripper state "
                    "was not present in the source"
                ),
            },
        },
    )

    summary: dict[str, int | float | str] = {
        "output_dir": str(destination),
        "episodes": len(episode_rows),
        "frames": global_index,
        "fps": fps,
        "height": height,
        "width": width,
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source robomimic HDF5 file.")
    parser.add_argument("--output", required=True, help="New LeRobot v3 directory.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--task", default="robot manipulation")
    parser.add_argument("--camera-key", default="table_cam")
    parser.add_argument("--position-key", default="eef_pos_base")
    parser.add_argument("--quaternion-key", default="eef_quat_base")
    parser.add_argument("--quaternion-order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--gripper-action-col", type=int, default=6)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional subset for conversion smoke tests.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    convert_dataset(
        args.input,
        args.output,
        fps=args.fps,
        task=args.task,
        camera_key=args.camera_key,
        position_key=args.position_key,
        quaternion_key=args.quaternion_key,
        quaternion_order=args.quaternion_order,
        gripper_action_col=args.gripper_action_col,
        codec=args.codec,
        crf=args.crf,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
