"""Click-to-pick the anchor (arm) patch on a frame, store its (row, col).

Shows a frame from the dataset with the 16x16 patch grid overlaid. Click the
robot-arm patch; the grid cell is saved to a JSON file that the eval script can
read via --anchor-file. Also always writes a labeled grid PNG so you can read
the coordinates by eye on a headless machine.

Usage:
    python -m scripts.pick_anchor --hdf5 dataset/multi_egg.hdf5 --demo 0 --frame 0
    # click the arm, close the window, then:
    python scripts/eval_lam_on_dataset.py --hdf5 dataset/multi_egg.hdf5 \
        --sequence 0 --anchor-file results/lam_check/anchor.json
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
from torchvision.transforms import v2

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_frame(hdf5, demo_idx, frame, size):
    with h5py.File(hdf5, "r") as f:
        key = list(f["data"].keys())[demo_idx]
        img = np.ascontiguousarray(f["data"][key]["obs"]["table_cam"][frame])
    x = torch.from_numpy(img).permute(2, 0, 1)
    x = v2.Resize((size, size), antialias=True)(x).to(torch.uint8)
    return x.permute(1, 2, 0).numpy(), key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="dataset/multi_egg.hdf5")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--out", default="results/lam_check/anchor.json")
    args = ap.parse_args()

    img, key = load_frame(args.hdf5, args.demo, args.frame, args.size)
    patch = args.size // args.grid
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    for i in range(args.grid + 1):
        ax.axhline(i * patch, color="w", lw=0.5, alpha=0.4)
        ax.axvline(i * patch, color="w", lw=0.5, alpha=0.4)
    ax.set_xticks([(c + 0.5) * patch for c in range(args.grid)])
    ax.set_xticklabels(range(args.grid), fontsize=6)
    ax.set_yticks([(r + 0.5) * patch for r in range(args.grid)])
    ax.set_yticklabels(range(args.grid), fontsize=6)
    ax.set_xlabel("col"); ax.set_ylabel("row")
    ax.set_title(f"{key} frame {args.frame} — click the ARM patch, then close window")

    state, box = {}, [None]

    def onclick(ev):
        if ev.xdata is None or ev.ydata is None:
            return
        col = int(min(args.grid - 1, max(0, ev.xdata // patch)))
        row = int(min(args.grid - 1, max(0, ev.ydata // patch)))
        state.update(row=row, col=col, demo=args.demo, frame=args.frame, hdf5=args.hdf5)
        if box[0] is not None:
            box[0].remove()
        box[0] = Rectangle((col * patch, row * patch), patch, patch,
                           fill=False, edgecolor="lime", lw=2.5)
        ax.add_patch(box[0])
        ax.set_title(f"row={row} col={col}  (click again to change; close to save)")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)

    # Always save a labeled grid preview (works even with a non-interactive backend).
    prev = os.path.splitext(args.out)[0] + "_grid.png"
    fig.savefig(prev, dpi=120, bbox_inches="tight")
    backend = matplotlib.get_backend()
    print(f"[grid preview] {prev}  (rows/cols labeled 0-{args.grid - 1})")
    print(f"[backend] {backend}")

    if backend.lower() != "agg":
        try:
            plt.show()
        except Exception as e:  # noqa: BLE001
            print(f"[no interactive display] {e}")

    if state:
        with open(args.out, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[saved] {state} -> {args.out}")
        print("Now run:\n  CUDA_VISIBLE_DEVICES=0 python scripts/eval_lam_on_dataset.py "
              f"--hdf5 {args.hdf5} --sequence {args.demo} --anchor-file {args.out}")
    else:
        print("No click captured (headless backend). Open the grid preview PNG, read the "
              "(row, col) of the arm cell, and pass it directly, e.g. --anchor 8 9")


if __name__ == "__main__":
    main()
