#!/usr/bin/env python3
"""Merge multiple recorded HDF5 datasets into one.

Each input file is expected to follow the record_real.py layout:
    /data/demo_0, /data/demo_1, ...   (each an episode group)
    /data.attrs["env_args"], /data.attrs["total"]
    /meta                             (dataset-level metadata, written once)

Demos are copied over and renumbered sequentially (demo_0, demo_1, ...) so
nothing collides. `total` is recomputed and `meta`/`env_args` are taken from
the first input.

Example:
    python ./scripts/merge_hdf5.py \
        --inputs ./dataset/hil_mini_lawam_vr/hil_corrections_11ep.hdf5 \
                 ./dataset/hil_mini_lawam_vr/hil_corrections_13ep.hdf5 \
        --output ./dataset/hil_mini_lawam_vr/hil_corrections_24ep.hdf5
"""

import argparse
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser(description="Merge recorded HDF5 datasets.")
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="Input .hdf5 files (2 or more)."
    )
    parser.add_argument("--output", required=True, help="Output .hdf5 path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        parser.error("Provide at least two --inputs to merge.")

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        parser.error(f"{out_path} already exists. Pass --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in args.inputs:
        if not Path(p).exists():
            parser.error(f"Input not found: {p}")

    with h5py.File(out_path, "w") as fout:
        data_out = fout.create_group("data")
        next_idx = 0
        total = 0
        meta_copied = False
        env_args = None

        for in_path in args.inputs:
            with h5py.File(in_path, "r") as fin:
                if "data" not in fin:
                    print(f"[WARN] {in_path}: no /data group, skipping.")
                    continue

                # Carry meta + env_args from the first file that has them.
                if not meta_copied and "meta" in fin:
                    fin.copy("meta", fout, name="meta")
                    meta_copied = True
                if env_args is None and "env_args" in fin["data"].attrs:
                    env_args = fin["data"].attrs["env_args"]

                # Copy demos in numeric order, renumbered sequentially.
                demo_names = sorted(
                    fin["data"].keys(),
                    key=lambda n: int(n.split("_")[1]) if n.startswith("demo_") else 1 << 30,
                )
                n_copied = 0
                for name in demo_names:
                    src = fin["data"][name]
                    dst_name = f"demo_{next_idx}"
                    fin["data"].copy(src, data_out, name=dst_name)
                    if "actions" in data_out[dst_name]:
                        total += int(data_out[dst_name]["actions"].shape[0])
                    next_idx += 1
                    n_copied += 1

                print(f"[INFO] {in_path}: copied {n_copied} demos.")

        if env_args is not None:
            data_out.attrs["env_args"] = env_args
        data_out.attrs["total"] = int(total)

        print(
            f"[DONE] Wrote {next_idx} demos ({total} steps) -> {out_path}"
        )


if __name__ == "__main__":
    main()
