"""Visualize the LaWM-predicted subgoal u_hat_T = LaWM(u_t, ConvPrior(u_t)).

The subgoal lives in DINO feature space (256 tokens x 768), so it cannot be
rendered as pixels directly. A table-only checkpoint gets the original 2x4
panel. For a checkpoint trained with ``use_wrist=True``, the report expands to
2x5 and adds the current wrist RGB input and its DINO PCA map:

    row 1: table o_t | wrist o_t | table o_T | pred change | true change
    row 2: PCA(table)| PCA(wrist)| PCA(u_hat)| PCA(u_T)    | cos agreement

    pred-change = per-patch ||u_hat_T - u_t|| (where the model THINKS motion happens)
    true-change = per-patch ||u_T - u_t||     (where motion actually happens)
    All PCA maps, including wrist, share one projection.

Good subgoal: PCA(u_hat_T) resembles PCA(u_T) rather than PCA(u_t); pred-change
lights up the same patches as true-change; the cos map is dark (high) except
possibly at motion patches.

Works with a phase-1 prior checkpoint or a full phase-2/joint checkpoint (only
the "prior" weights are used).

Example:
    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.viz_subgoal \
        --ckpt results/mini_lawam/prior_phase1.pt --hdf5 dataset/multi_egg_114ep.hdf5 \
        --demo demo_0 --t 40

LeRobot v3 example (only the requested frames are decoded):
    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.viz_subgoal \
        --ckpt results/mini_lawam/prior_phase1.pt \
        --lerobot dataset/0827_cardboard_box_50 --episode 0 \
        --camera observation.images.cam_high --t 40 80 120
"""

import argparse
import os
from contextlib import ExitStack

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from mini_lawam.lerobot_video import LeRobotEpisodeSource
from mini_lawam.model import MiniLaWAM, MiniLaWAMConfig

GRID = 16  # DINO 256x256 / patch16 -> 16x16 tokens
DEFAULT_HDF5 = "dataset/multi_egg.hdf5"
DEFAULT_LEROBOT_CAMERA = "observation.images.cam_high"
DEFAULT_LEROBOT_WRIST_CAMERA = "observation.images.cam_left_wrist"


def to_grid(tok: torch.Tensor) -> np.ndarray:
    """[K, ...] token tensor -> [16, 16, ...] numpy."""
    return tok.reshape(GRID, GRID, *tok.shape[1:]).cpu().numpy()


def joint_pca_rgb(feats):
    """List of [K, D] token maps -> list of [16,16,3] RGB via ONE shared PCA."""
    x = torch.cat(feats, dim=0)                      # [n*K, D]
    mean = x.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x - mean, q=3)
    projected = [(f - mean) @ v[:, :3] for f in feats]  # each [K, 3]
    all_projected = torch.cat(projected, dim=0)
    lo = all_projected.min(0).values
    hi = all_projected.max(0).values
    return [to_grid((p - lo) / (hi - lo + 1e-8)) for p in projected]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/mini_lawam/prior_phase1.pt")
    source_group = ap.add_mutually_exclusive_group()
    source_group.add_argument(
        "--hdf5",
        default=None,
        help=f"robomimic HDF5 dataset (default when no source is given: {DEFAULT_HDF5})",
    )
    source_group.add_argument(
        "--lerobot",
        default=None,
        help="local LeRobot v3 dataset directory",
    )
    ap.add_argument("--demo", default=None, help="HDF5 demo key; default = first")
    ap.add_argument("--episode", type=int, default=None,
                    help="LeRobot episode_index; default = first episode")
    ap.add_argument("--camera", default=DEFAULT_LEROBOT_CAMERA,
                    help="LeRobot main camera video feature key")
    ap.add_argument("--wrist-camera", default=DEFAULT_LEROBOT_WRIST_CAMERA,
                    help="LeRobot wrist camera key (used when cfg.use_wrist=True)")
    ap.add_argument("--t", type=int, nargs="+", default=[0],
                    help="one or more start frames, e.g. --t 0 40 80")
    ap.add_argument("--wrist-key", default="wrist_cam",
                    help="HDF5 wrist observation key (used when cfg.use_wrist=True).")
    ap.add_argument("--out-dir", default="results/mini_lawam/subgoal_viz")
    ap.add_argument("--dpi", type=int, default=150,
                    help="Saved figure resolution (default: 150 for presentation use).")
    args = ap.parse_args()
    if args.dpi < 1:
        raise ValueError("--dpi must be >= 1")
    if args.hdf5 is None and args.lerobot is None:
        args.hdf5 = DEFAULT_HDF5
    if args.lerobot is not None and args.demo is not None:
        ap.error("--demo applies to --hdf5; use --episode with --lerobot")
    if args.hdf5 is not None and args.episode is not None:
        ap.error("--episode applies to --lerobot; use --demo with --hdf5")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MiniLaWAMConfig(**ck["cfg"])
    model = MiniLaWAM(cfg).to(device)
    model.prior.load_state_dict(ck["prior"])
    model.prior.eval()
    H = cfg.future_horizon
    resize = v2.Resize((256, 256), antialias=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"ckpt: {args.ckpt} (phase={ck.get('phase', '?')}, step={ck.get('step', '?')}) "
          f"| future_horizon={H}")

    with ExitStack() as stack:
        if args.lerobot is not None:
            camera_keys = [args.camera]
            if cfg.use_wrist:
                camera_keys.append(args.wrist_camera)
            source = LeRobotEpisodeSource.open(
                args.lerobot,
                episode_index=args.episode,
                camera_keys=camera_keys,
            )
            sample_name = f"episode_{source.episode_index:06d}"
            T = source.length

            def load_main_frames(indices):
                return source.frames(args.camera, indices)

            def load_wrist_frame(index):
                return source.frames(args.wrist_camera, [index])[0]

            print(
                f"dataset: {source.root} (LeRobot {source.info.get('codebase_version', '?')}) "
                f"| episode={source.episode_index} | frames={T} | camera={args.camera}"
            )
        else:
            hdf5_file = stack.enter_context(h5py.File(args.hdf5, "r"))
            demo = args.demo or list(hdf5_file["data"].keys())[0]
            obs = hdf5_file["data"][demo]["obs"]
            cam = obs["table_cam"]
            wrist_cam = None
            if cfg.use_wrist:
                if args.wrist_key not in obs:
                    raise KeyError(
                        f"checkpoint uses wrist input, but demo {demo!r} has no "
                        f"obs/{args.wrist_key!s}"
                    )
                wrist_cam = obs[args.wrist_key]
            sample_name = demo
            T = cam.shape[0]

            def load_main_frames(indices):
                return np.asarray([cam[index] for index in indices])

            def load_wrist_frame(index):
                if wrist_cam is None:
                    raise RuntimeError("wrist frame requested for a checkpoint without wrist input")
                return wrist_cam[index]

        for t in args.t:
            if t < 0:
                print(f"skip t={t}: frame index must be non-negative")
                continue
            if t + H >= T:
                print(f"skip t={t}: t+{H} beyond episode length {T}")
                continue
            raw_t, raw_T = load_main_frames([t, t + H])          # (H,W,3) u8
            raw_wrist = load_wrist_frame(t) if cfg.use_wrist else None

            def prep(img):
                x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
                return resize(x).to(torch.uint8)

            frames = torch.stack([prep(raw_t), prep(raw_T)])[None].to(device)
            vids, _ = gpu_two_view_video_aug(frames, training=False)  # [1,2,3,256,256]
            o_t, o_T = vids[:, 0:1], vids[:, 1:2]

            u_t = model._feat(o_t)[:, :1]                         # [1,1,K,D]
            u_T = model._feat(o_T)[:, :1]
            z_hat = model.prior(u_t[:, 0])
            u_hat = model.lam.decoder(u_t, z_hat)
            if isinstance(u_hat, tuple):
                u_hat = u_hat[0]

            ut, uh, uT = u_t[0, 0], u_hat[0, 0], u_T[0, 0]        # each [K, D]
            wt = None
            img_wrist = None
            if raw_wrist is not None:
                wrist_u8 = prep(raw_wrist)
                wrist_vid, _ = gpu_two_view_video_aug(
                    wrist_u8[None, None].to(device), training=False
                )
                wt = model._feat(wrist_vid)[0, 0]                  # [K, D]
                img_wrist = np.asarray(wrist_u8.permute(1, 2, 0))

            if wt is None:
                pca_ut, pca_uh, pca_uT = joint_pca_rgb([ut, uh, uT])
                pca_wt = None
            else:
                pca_ut, pca_wt, pca_uh, pca_uT = joint_pca_rgb(
                    [ut, wt, uh, uT]
                )
            chg_pred = to_grid((uh - ut).norm(dim=-1))            # [16,16]
            chg_true = to_grid((uT - ut).norm(dim=-1))
            agree = to_grid(F.cosine_similarity(uh, uT, dim=-1))  # [16,16]
            vmax = max(chg_pred.max(), chg_true.max())
            cos_full = float(F.cosine_similarity(uh.flatten(), uT.flatten(), dim=0))

            img_t = np.asarray(prep(raw_t).permute(1, 2, 0))
            img_T = np.asarray(prep(raw_T).permute(1, 2, 0))
            if wt is None:
                panels = [
                    (img_t, None, f"table o_t ({sample_name} t={t})"),
                    (img_T, None, f"table o_T true (t={t + H})"),
                    (chg_pred, "pred change ||u_hat_T - u_t||",
                     "where model PREDICTS motion"),
                    (chg_true, "true change ||u_T - u_t||",
                     "where motion ACTUALLY is"),
                    (pca_ut, None, "PCA(table u_t)"),
                    (pca_uh, None, "PCA(u_hat_T) — predicted subgoal"),
                    (pca_uT, None, "PCA(table u_T true)"),
                    (agree, "cos(u_hat_T, u_T)",
                     f"agreement (full-frame cos={cos_full:.3f})"),
                ]
                ncols, figsize = 4, (18, 9)
            else:
                panels = [
                    (img_t, None, f"table o_t ({sample_name} t={t})"),
                    (img_wrist, None, "wrist o_t (action-head input)"),
                    (img_T, None, f"table o_T true (t={t + H})"),
                    (chg_pred, "pred change ||u_hat_T - u_t||",
                     "where model PREDICTS motion"),
                    (chg_true, "true change ||u_T - u_t||",
                     "where motion ACTUALLY is"),
                    (pca_ut, None, "PCA(table u_t)"),
                    (pca_wt, None, "PCA(wrist o_t)"),
                    (pca_uh, None, "PCA(u_hat_T) — predicted subgoal"),
                    (pca_uT, None, "PCA(table u_T true)"),
                    (agree, "cos(u_hat_T, u_T)",
                     f"agreement (full-frame cos={cos_full:.3f})"),
                ]
                ncols, figsize = 5, (22, 9)
            fig, axes = plt.subplots(2, ncols, figsize=figsize)
            for ax, (im, kind, title) in zip(axes.flat, panels):
                if kind is None:
                    ax.imshow(im)
                elif kind.startswith("cos"):
                    m = ax.imshow(im, cmap="viridis", vmin=0.0, vmax=1.0)
                    fig.colorbar(m, ax=ax, fraction=0.046)
                else:  # change-magnitude maps share one scale
                    ax.imshow(img_t, extent=(0, GRID, GRID, 0))
                    m = ax.imshow(im, cmap="jet", alpha=0.55, vmin=0.0, vmax=vmax,
                                  extent=(0, GRID, GRID, 0))
                    fig.colorbar(m, ax=ax, fraction=0.046)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            out = os.path.join(args.out_dir, f"{sample_name}_t{t}.png")
            wrist_note = " + wrist" if wt is not None else ""
            fig.suptitle(
                f"Mini-LaWAM DINO feature-space report{wrist_note} — "
                f"{sample_name}, t={t}, horizon={H}",
                fontsize=15,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.4, w_pad=1.2)
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            print(
                f"saved {out}  (wrist={'yes' if wt is not None else 'no'}, "
                f"full-frame cos(u_hat_T, u_T)={cos_full:.3f})"
            )


if __name__ == "__main__":
    main()
