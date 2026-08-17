#!/usr/bin/env python3
"""Summarize HIL correction and VR-takeover activity in one HDF5 file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .delete_episodes import demo_names


@dataclass(frozen=True)
class EpisodeStats:
    demo: str
    frames: int
    duration_sec: float
    correction_frames: int
    correction_fraction: float
    correction_duration_sec: float
    correction_events: int
    longest_correction_frames: int
    takeover_frames: int
    takeover_duration_sec: float
    takeover_events: int
    outcome: str


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open ranges for contiguous true regions in a 1D mask."""
    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {values.shape}")
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _frame_count(group: h5py.Group) -> int:
    for key in ("base_policy_actions", "bc_actions", "executed_actions", "actions"):
        dataset = group.get(key)
        if isinstance(dataset, h5py.Dataset) and dataset.ndim:
            return int(dataset.shape[0])
    dataset = group.get("intervene_mask")
    if isinstance(dataset, h5py.Dataset) and dataset.ndim:
        return int(dataset.shape[0])
    raise ValueError(f"{group.name} has no action array or intervene_mask")


def _mask(group: h5py.Group, key: str, frames: int) -> np.ndarray:
    dataset = group.get(key)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{group.name} is missing required dataset {key!r}")
    values = np.asarray(dataset, dtype=np.bool_)
    if values.shape != (frames,):
        raise ValueError(f"{group.name}/{key} has shape {values.shape}, expected {(frames,)}")
    return values


def _frame_durations(group: h5py.Group, frames: int, fallback_hz: float) -> np.ndarray:
    fallback_dt = 1.0 / float(fallback_hz)
    durations = np.full(frames, fallback_dt, dtype=np.float64)
    dataset = group.get("timestamps")
    if not isinstance(dataset, h5py.Dataset):
        return durations
    timestamps = np.asarray(dataset, dtype=np.float64)
    if timestamps.shape != (frames,) or frames < 2 or not np.all(np.isfinite(timestamps)):
        return durations
    differences = np.diff(timestamps)
    if np.any(differences <= 0.0):
        return durations
    durations[:-1] = differences
    durations[-1] = float(np.median(differences))
    return durations


def _outcome(group: h5py.Group) -> str:
    value = group.attrs.get("outcome", "unknown")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)


def analyze_episode(name: str, group: h5py.Group, *, fallback_hz: float) -> EpisodeStats:
    frames = _frame_count(group)
    correction_mask = _mask(group, "intervene_mask", frames)
    takeover_mask = _mask(group, "manual_control_mask", frames)
    durations = _frame_durations(group, frames, fallback_hz)
    correction_runs = true_runs(correction_mask)
    takeover_runs = true_runs(takeover_mask)
    correction_frames = int(correction_mask.sum())
    takeover_frames = int(takeover_mask.sum())
    return EpisodeStats(
        demo=name,
        frames=frames,
        duration_sec=float(durations.sum()),
        correction_frames=correction_frames,
        correction_fraction=correction_frames / frames if frames else 0.0,
        correction_duration_sec=float(durations[correction_mask].sum()),
        correction_events=len(correction_runs),
        longest_correction_frames=max((end - start for start, end in correction_runs), default=0),
        takeover_frames=takeover_frames,
        takeover_duration_sec=float(durations[takeover_mask].sum()),
        takeover_events=len(takeover_runs),
        outcome=_outcome(group),
    )


def aggregate(episodes: Sequence[EpisodeStats]) -> dict[str, int | float]:
    frames = sum(item.frames for item in episodes)
    correction_frames = sum(item.correction_frames for item in episodes)
    takeover_frames = sum(item.takeover_frames for item in episodes)
    correction_events = sum(item.correction_events for item in episodes)
    takeover_events = sum(item.takeover_events for item in episodes)
    return {
        "episodes": len(episodes),
        "episodes_with_corrections": sum(item.correction_frames > 0 for item in episodes),
        "frames": frames,
        "duration_sec": sum(item.duration_sec for item in episodes),
        "correction_frames": correction_frames,
        "correction_fraction": correction_frames / frames if frames else 0.0,
        "correction_duration_sec": sum(item.correction_duration_sec for item in episodes),
        "correction_events": correction_events,
        "mean_frames_per_correction_event": (
            correction_frames / correction_events if correction_events else 0.0
        ),
        "longest_correction_frames": max(
            (item.longest_correction_frames for item in episodes),
            default=0,
        ),
        "takeover_frames": takeover_frames,
        "takeover_duration_sec": sum(item.takeover_duration_sec for item in episodes),
        "takeover_events": takeover_events,
        "mean_frames_per_takeover": takeover_frames / takeover_events if takeover_events else 0.0,
    }


def analyze_file(path: str | Path, *, fallback_hz: float = 20.0) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    with h5py.File(source, "r") as file:
        names = demo_names(file)
        if not names:
            raise ValueError("HDF5 file contains no data/demo_N episodes")
        episodes = [
            analyze_episode(name, file["data"][name], fallback_hz=fallback_hz)
            for name in names
        ]
    return {
        "file": str(source),
        "definitions": {
            "correction_event": "one contiguous intervene_mask=True region",
            "takeover_event": "one contiguous manual_control_mask=True region",
            "fallback_hz": float(fallback_hz),
        },
        "episodes": [asdict(item) for item in episodes],
        "total": aggregate(episodes),
    }


def _print_report(report: dict[str, object], *, summary_only: bool) -> None:
    if not summary_only:
        for item in report["episodes"]:
            print(
                f"{item['demo']:<12} frames={item['frames']:<6} "
                f"correction_frames={item['correction_frames']:<5} "
                f"({item['correction_fraction']:.1%}) "
                f"correction_events={item['correction_events']:<4} "
                f"takeovers={item['takeover_events']:<4} "
                f"manual_frames={item['takeover_frames']:<5} "
                f"outcome={item['outcome']}"
            )
    total = report["total"]
    print("\nTOTAL")
    print(f"  episodes:                    {total['episodes']}")
    print(f"  episodes with corrections:   {total['episodes_with_corrections']}")
    print(f"  frames:                      {total['frames']}")
    print(f"  recorded duration:           {total['duration_sec']:.2f} s")
    print(
        f"  correction frames:           {total['correction_frames']} "
        f"({total['correction_fraction']:.1%})"
    )
    print(f"  correction duration:         {total['correction_duration_sec']:.2f} s")
    print(f"  correction events/bursts:    {total['correction_events']}")
    print(f"  mean frames/correction:      {total['mean_frames_per_correction_event']:.2f}")
    print(f"  longest correction:          {total['longest_correction_frames']} frames")
    print(f"  VR takeover events:          {total['takeover_events']}")
    print(f"  VR takeover frames:          {total['takeover_frames']}")
    print(f"  VR takeover duration:        {total['takeover_duration_sec']:.2f} s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path, help="HIL correction .hdf5/.h5 file")
    parser.add_argument(
        "--fallback-hz",
        type=float,
        default=20.0,
        help="Rate used for duration estimates when timestamps are absent or invalid.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Hide per-episode rows.")
    parser.add_argument("--json-output", type=Path, help="Also save the complete report as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.hdf5.expanduser().is_file():
        parser.error(f"input file does not exist: {args.hdf5.expanduser()}")
    if not np.isfinite(args.fallback_hz) or args.fallback_hz <= 0.0:
        parser.error("--fallback-hz must be positive and finite")
    try:
        report = analyze_file(args.hdf5, fallback_hz=args.fallback_hz)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    _print_report(report, summary_only=args.summary_only)
    if args.json_output is not None:
        output = args.json_output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON: {output}")


if __name__ == "__main__":
    main()
