#!/usr/bin/env python3
"""Interactive GUI viewer for robomimic-style HDF5 robot datasets.

Shows each timestep's camera images alongside the end-effector pose, joint
positions and action vector. Switch frames with the Next/Prev buttons or the
slider, and switch demos with the demo buttons. Arrow keys also work:
    left/right  -> previous/next frame
    up/down     -> next/previous demo

Example:
    python data/view_hdf5_gui.py --input datasets/no_rotation_100.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("TkAgg")  # interactive backend; falls back below if unavailable
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


def natural_demo_key(name: str):
    prefix, _, suffix = name.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return name, name


def is_image(dataset: h5py.Dataset) -> bool:
    return dataset.ndim == 4 and dataset.shape[-1] in (1, 3, 4)


class HDF5Viewer:
    def __init__(self, path: str):
        self.file = h5py.File(path, "r")
        if "data" not in self.file:
            raise KeyError("expected top-level group 'data'")
        self.path = path
        self.demos = sorted(self.file["data"].keys(), key=natural_demo_key)
        if not self.demos:
            raise ValueError("no demos found under 'data'")

        self.demo_index = 0
        self.frame_index = 0

        self._load_demo(self.demo_index)
        self._build_figure()
        self._draw()

    # ---- data access -------------------------------------------------------
    def _load_demo(self, demo_index: int):
        self.demo_index = demo_index % len(self.demos)
        demo_name = self.demos[self.demo_index]
        demo = self.file["data"][demo_name]
        obs = demo["obs"]

        self.demo_name = demo_name
        self.image_keys = [k for k in obs.keys() if is_image(obs[k])]
        self.vector_keys = [k for k in obs.keys() if not is_image(obs[k])]
        self.obs = obs
        self.actions = demo["actions"] if "actions" in demo else None

        lengths = [obs[k].shape[0] for k in obs.keys()]
        if self.actions is not None:
            lengths.append(self.actions.shape[0])
        self.num_frames = min(lengths)
        self.frame_index = min(self.frame_index, self.num_frames - 1)

    # ---- figure ------------------------------------------------------------
    def _build_figure(self):
        n_img = max(1, len(self.image_keys))
        self.fig = plt.figure(figsize=(5 * n_img + 4, 6))
        self.fig.canvas.manager.set_window_title(f"HDF5 Viewer - {Path(self.path).name}")

        gs = self.fig.add_gridspec(
            2, n_img + 1,
            height_ratios=[10, 1],
            width_ratios=[4] * n_img + [5],
        )

        self.img_axes = []
        self.img_artists = []
        for i in range(n_img):
            ax = self.fig.add_subplot(gs[0, i])
            ax.axis("off")
            self.img_axes.append(ax)
            self.img_artists.append(None)

        self.text_ax = self.fig.add_subplot(gs[0, n_img])
        self.text_ax.axis("off")
        self.text_handle = self.text_ax.text(
            0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9,
            transform=self.text_ax.transAxes,
        )

        # slider spanning the image columns
        slider_ax = self.fig.add_subplot(gs[1, :n_img])
        self.slider = Slider(
            slider_ax, "frame", 0, max(1, self.num_frames - 1),
            valinit=self.frame_index, valstep=1,
        )
        self.slider.on_changed(self._on_slider)

        # buttons in the bottom-right cell area
        self._add_buttons()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.16, wspace=0.15)

    def _add_buttons(self):
        # axes are placed in figure coordinates along the bottom
        specs = [
            ("<< demo", self._prev_demo),
            ("< frame", self._prev_frame),
            ("frame >", self._next_frame),
            ("demo >>", self._next_demo),
        ]
        self.buttons = []
        width = 0.09
        gap = 0.01
        start = 0.60
        for i, (label, cb) in enumerate(specs):
            ax = self.fig.add_axes([start + i * (width + gap), 0.03, width, 0.06])
            btn = Button(ax, label)
            btn.on_clicked(cb)
            self.buttons.append(btn)

    # ---- callbacks ---------------------------------------------------------
    def _on_slider(self, value):
        self.frame_index = int(value)
        self._draw(update_slider=False)

    def _next_frame(self, _event=None):
        self.frame_index = (self.frame_index + 1) % self.num_frames
        self._sync_slider()
        self._draw(update_slider=False)

    def _prev_frame(self, _event=None):
        self.frame_index = (self.frame_index - 1) % self.num_frames
        self._sync_slider()
        self._draw(update_slider=False)

    def _next_demo(self, _event=None):
        self._change_demo(self.demo_index + 1)

    def _prev_demo(self, _event=None):
        self._change_demo(self.demo_index - 1)

    def _change_demo(self, new_index):
        self.frame_index = 0
        self._load_demo(new_index)
        self.slider.valmax = max(1, self.num_frames - 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self._sync_slider()
        self._draw(update_slider=False)

    def _on_key(self, event):
        if event.key == "right":
            self._next_frame()
        elif event.key == "left":
            self._prev_frame()
        elif event.key == "up":
            self._next_demo()
        elif event.key == "down":
            self._prev_demo()

    def _sync_slider(self):
        self.slider.eventson = False
        self.slider.set_val(self.frame_index)
        self.slider.eventson = True

    # ---- rendering ---------------------------------------------------------
    def _format_text(self) -> str:
        t = self.frame_index
        lines = [
            f"file : {Path(self.path).name}",
            f"demo : {self.demo_name}  ({self.demo_index + 1}/{len(self.demos)})",
            f"frame: {t}/{self.num_frames - 1}",
            "",
        ]

        def fmt(arr):
            return "[" + ", ".join(f"{v:+.4f}" for v in np.asarray(arr).ravel()) + "]"

        for key in self.vector_keys:
            lines.append(f"{key}:")
            lines.append(f"  {fmt(self.obs[key][t])}")
        if self.actions is not None:
            lines.append("action:")
            lines.append(f"  {fmt(self.actions[t])}")
        return "\n".join(lines)

    def _draw(self, update_slider: bool = True):
        t = self.frame_index
        for i, key in enumerate(self.image_keys):
            frame = self.obs[key][t]
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            ax = self.img_axes[i]
            if self.img_artists[i] is None:
                self.img_artists[i] = ax.imshow(frame)
            else:
                self.img_artists[i].set_data(frame)
                self.img_artists[i].set_extent((0, frame.shape[1], frame.shape[0], 0))
            ax.set_title(key, fontsize=10)

        self.text_handle.set_text(self._format_text())
        if update_slider:
            self._sync_slider()
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

    def close(self):
        self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="datasets/no_rotation_100.hdf5", help="Path to HDF5 dataset.")
    args = parser.parse_args()

    viewer = HDF5Viewer(args.input)
    try:
        viewer.show()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
