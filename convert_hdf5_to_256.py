"""Resize the camera images in a robomimic/IsaacLab HDF5 to 256x256.

Copies an HDF5 file verbatim EXCEPT the image datasets (`obs/table_cam`,
`obs/wrist_cam`), which are resized from their native HxW to 256x256 and written
back as uint8 HWC. Uses the SAME resize as training
(`torchvision.transforms.v2.Resize((256,256), antialias=True)` on a CHW uint8
tensor, see mini_lawam/data.py::_frame) so pre-converted frames match exactly
what the on-the-fly loader would produce -- no train/deploy skew.

Everything else -- group hierarchy, all attrs (root, `data`, `meta`, per-demo),
and every non-image dataset (`actions`, `eef_pos_base`, `joint_pos`, ...) -- is
copied unchanged, preserving gzip compression.

Usage:
    python convert_hdf5_to_256.py \
        --in  dataset/new_100ep_multi_egg_exp_plate.hdf5 \
        --out dataset/new_100ep_multi_egg_exp_plate_256.hdf5

    # override which datasets are treated as images, or the target size:
    python convert_hdf5_to_256.py --in a.hdf5 --out b.hdf5 \
        --image-keys table_cam wrist_cam --size 256
"""

import argparse
import os

import h5py
import numpy as np
import torch
from torchvision.transforms import v2


def resize_frames(frames_thwc: np.ndarray, resize, device, batch: int) -> np.ndarray:
    """(T,H,W,3) uint8 -> (T,size,size,3) uint8, matching data.py::_frame."""
    out = np.empty((frames_thwc.shape[0],) + tuple(resize.size) + (3,), dtype=np.uint8)
    for s in range(0, frames_thwc.shape[0], batch):
        chunk = frames_thwc[s:s + batch]                                  # [b,H,W,3] u8
        x = torch.from_numpy(np.ascontiguousarray(chunk)).permute(0, 3, 1, 2)  # [b,3,H,W]
        x = x.to(device)
        x = resize(x).to(torch.uint8)                                    # [b,3,size,size]
        out[s:s + batch] = x.permute(0, 2, 3, 1).cpu().numpy()           # [b,size,size,3]
    return out


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="input HDF5")
    ap.add_argument("--out", dest="dst", required=True, help="output HDF5")
    ap.add_argument("--size", type=int, default=256, help="square target size")
    ap.add_argument("--image-keys", nargs="+", default=["table_cam", "wrist_cam"],
                    help="obs/* dataset names to resize (missing ones skipped)")
    ap.add_argument("--batch", type=int, default=256, help="frames resized per step")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting an existing --out")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("--in and --out must differ (this does not edit in place).")
    if os.path.exists(args.dst) and not args.overwrite:
        raise SystemExit(f"{args.dst} exists; pass --overwrite to replace it.")

    resize = v2.Resize((args.size, args.size), antialias=True)
    img_keys = set(args.image_keys)
    device = args.device
    print(f"resize -> {args.size}x{args.size} | device={device} | image keys={sorted(img_keys)}")

    with h5py.File(args.src, "r") as fin, h5py.File(args.dst, "w") as fout:
        copy_attrs(fin, fout)                       # root attrs
        gin, gout = fin["data"], fout.create_group("data")
        copy_attrs(gin, gout)                        # data-group attrs (env_args, total, ...)

        # copy `meta` and any other top-level groups/datasets verbatim
        for k in fin.keys():
            if k != "data":
                fin.copy(k, fout)

        demos = list(gin.keys())
        for di, demo in enumerate(demos):
            din = gin[demo]
            dout = gout.create_group(demo)
            copy_attrs(din, dout)

            n_img = 0
            for sub in din.keys():
                if sub == "obs":
                    obs_in = din["obs"]
                    obs_out = dout.create_group("obs")
                    copy_attrs(obs_in, obs_out)
                    for name, ds in obs_in.items():
                        if name in img_keys and ds.ndim == 4:
                            frames = resize_frames(ds[...], resize, device, args.batch)
                            chunks = (min(32, frames.shape[0]),) + frames.shape[1:3] + (1,)
                            d = obs_out.create_dataset(
                                name, data=frames, dtype="uint8",
                                compression="gzip", chunks=chunks)
                            copy_attrs(ds, d)
                            n_img += 1
                        else:
                            obs_in.copy(name, obs_out)   # non-image obs verbatim
                else:
                    din.copy(sub, dout)                  # actions, etc. verbatim

            print(f"[{di + 1}/{len(demos)}] {demo}: resized {n_img} image dataset(s), "
                  f"T={din.attrs.get('num_samples', '?')}")

    print(f"done -> {args.dst}")


if __name__ == "__main__":
    main()
