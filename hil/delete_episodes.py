"""List or safely remove complete episodes from an HDF5 dataset."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


DEMO_RE = re.compile(r"demo_(\d+)$")
SELECTOR_RE = re.compile(r"(?:demo_)?(\d+)$")
RANGE_RE = re.compile(r"(?:demo_)?(\d+)-(?:demo_)?(\d+)$")


def _demo_sort_key(name: str) -> tuple[int, int | str]:
    match = DEMO_RE.fullmatch(name)
    return (0, int(match.group(1))) if match else (1, name)


def demo_names(file: h5py.File) -> list[str]:
    if "data" not in file or not isinstance(file["data"], h5py.Group):
        raise ValueError("HDF5 file does not contain a 'data' group")
    return sorted((name for name in file["data"] if DEMO_RE.fullmatch(name)), key=_demo_sort_key)


def parse_episode_selectors(selectors: Sequence[str], available: Iterable[str]) -> set[str]:
    """Convert values such as ``2``, ``demo_4``, and ``7-10`` into demo names."""
    selected: set[str] = set()
    for argument in selectors:
        for token in argument.split(","):
            token = token.strip()
            if not token:
                continue
            match = SELECTOR_RE.fullmatch(token)
            if match:
                selected.add(f"demo_{int(match.group(1))}")
                continue
            match = RANGE_RE.fullmatch(token)
            if match:
                start, end = (int(match.group(1)), int(match.group(2)))
                if end < start:
                    raise ValueError(f"episode range must be ascending: {token!r}")
                selected.update(f"demo_{index}" for index in range(start, end + 1))
                continue
            raise ValueError(f"invalid episode selector {token!r}; use values such as 2, demo_4, or 7-10")

    available_set = set(available)
    missing = sorted(selected - available_set, key=_demo_sort_key)
    if missing:
        raise ValueError(f"episodes not found in the file: {', '.join(missing)}")
    if not selected:
        raise ValueError("no episodes were selected")
    return selected


def _sample_count(group: h5py.Group) -> int:
    for key in ("base_policy_actions", "bc_actions", "executed_actions", "actions"):
        if key in group and isinstance(group[key], h5py.Dataset) and group[key].ndim:
            return int(group[key].shape[0])
    return int(group.attrs.get("num_samples", 0))


def _mask_count(group: h5py.Group, key: str, attr: str) -> int:
    if key in group and isinstance(group[key], h5py.Dataset):
        return int(np.count_nonzero(group[key][...]))
    return int(group.attrs.get(attr, 0))


def episode_summary(group: h5py.Group) -> tuple[int, int, int, str]:
    samples = _sample_count(group)
    interventions = _mask_count(group, "intervene_mask", "num_interventions")
    manual = _mask_count(group, "manual_control_mask", "num_manual_control")
    outcome = group.attrs.get("outcome", "unknown")
    if isinstance(outcome, bytes):
        outcome = outcome.decode("utf-8", errors="replace")
    return samples, interventions, manual, str(outcome)


def print_episode_list(path: Path) -> None:
    total_samples = total_interventions = 0
    with h5py.File(path, "r") as file:
        names = demo_names(file)
        for name in names:
            samples, interventions, manual, outcome = episode_summary(file["data"][name])
            total_samples += samples
            total_interventions += interventions
            print(
                f"{name:<12} samples={samples:<6} interventions={interventions:<6} "
                f"manual={manual:<6} outcome={outcome}"
            )
    print(f"TOTAL        episodes={len(names)} samples={total_samples} interventions={total_interventions}")


def _copy_attrs(source, destination) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def copy_without_episodes(
    source_path: str | Path,
    destination_path: str | Path,
    removed: set[str],
    *,
    renumber: bool = False,
) -> dict[str, str]:
    """Copy a dataset while omitting selected demos. Returns old-to-new demo names."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    mapping: dict[str, str] = {}

    with h5py.File(source_path, "r") as source, h5py.File(destination_path, "w") as destination:
        names = demo_names(source)
        missing = removed - set(names)
        if missing:
            missing_text = ", ".join(sorted(missing, key=_demo_sort_key))
            raise ValueError(f"episodes not found in the file: {missing_text}")

        _copy_attrs(source, destination)
        for name in source:
            if name != "data":
                source.copy(name, destination, name=name)

        source_data = source["data"]
        destination_data = destination.create_group("data")
        _copy_attrs(source_data, destination_data)

        kept_index = 0
        for name in sorted(source_data, key=_demo_sort_key):
            if DEMO_RE.fullmatch(name):
                if name in removed:
                    continue
                target_name = f"demo_{kept_index}" if renumber else name
                kept_index += 1
                mapping[name] = target_name
            else:
                target_name = name
            source.copy(source_data[name], destination_data, name=target_name)

        total = total_manual = total_interventions = 0
        for name in mapping.values():
            group = destination_data[name]
            total += _sample_count(group)
            total_manual += _mask_count(group, "manual_control_mask", "num_manual_control")
            total_interventions += _mask_count(group, "intervene_mask", "num_interventions")
        destination_data.attrs["total"] = total
        destination_data.attrs["total_manual_control"] = total_manual
        destination_data.attrs["total_interventions"] = total_interventions
        destination.flush()
    return mapping


def _staged_copy(
    source: Path,
    destination: Path,
    removed: set[str],
    *,
    renumber: bool,
) -> tuple[dict[str, str], Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        mapping = copy_without_episodes(source, temporary, removed, renumber=renumber)
        return mapping, temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_pruned_copy_atomic(
    source_path: str | Path,
    destination_path: str | Path,
    removed: set[str],
    *,
    renumber: bool = False,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write a pruned copy and expose it only after the copy succeeds.

    ``source_path`` and ``destination_path`` may be the same when ``overwrite``
    is true. This is useful for updating a disposable working copy while an
    original dataset is retained elsewhere.
    """
    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input file does not exist: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")

    mapping, staged = _staged_copy(source, destination, removed, renumber=renumber)
    try:
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List or remove data/demo_N episode groups from an HDF5 dataset."
    )
    parser.add_argument("hdf5", type=Path, help="Input .hdf5 or .h5 file")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--list", action="store_true", help="List episodes without changing any file")
    operation.add_argument(
        "--episodes",
        nargs="+",
        metavar="EPISODE",
        help="Episodes to remove; accepts 2, demo_4, comma lists, and ranges such as 7-10",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="Output path (default: INPUT_pruned.hdf5)")
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input only after a successful copy; preserves INPUT.bak",
    )
    parser.add_argument("--renumber", action="store_true", help="Rename retained demos consecutively from demo_0")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    source = args.hdf5.expanduser().resolve()
    if not source.is_file():
        parser.error(f"input file does not exist: {source}")

    if args.list:
        print_episode_list(source)
        return

    try:
        with h5py.File(source, "r") as file:
            available = demo_names(file)
        removed = parse_episode_selectors(args.episodes, available)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.in_place:
        backup = source.with_name(f"{source.name}.bak")
        if backup.exists():
            parser.error(f"backup already exists: {backup}; move or remove it before using --in-place")
        mapping, staged = _staged_copy(source, source, removed, renumber=args.renumber)
        try:
            os.replace(source, backup)
            try:
                os.replace(staged, source)
            except BaseException:
                os.replace(backup, source)
                raise
        finally:
            staged.unlink(missing_ok=True)
        destination = source
        print(f"Saved original file as: {backup}")
    else:
        destination = (args.output or source.with_name(f"{source.stem}_pruned{source.suffix}")).expanduser().resolve()
        if destination == source:
            parser.error("--output must differ from the input; use --in-place for a recoverable replacement")
        if destination.exists() and not args.overwrite:
            parser.error(f"output already exists: {destination}; choose another path or pass --overwrite")
        mapping, staged = _staged_copy(source, destination, removed, renumber=args.renumber)
        try:
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)

    removed_text = ", ".join(sorted(removed, key=_demo_sort_key))
    print(f"Removed {len(removed)} episode(s): {removed_text}")
    print(f"Kept {len(mapping)} episode(s) in: {destination}")
    if args.renumber:
        changed = [f"{old}->{new}" for old, new in mapping.items() if old != new]
        if changed:
            print(f"Renumbered: {', '.join(changed)}")


if __name__ == "__main__":
    main()
