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
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from mini_lawam.model import MiniLaWAM, MiniLaWAMConfig

GRID = 16  # DINO 256x256 / patch16 -> 16x16 tokens


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
    ap.add_argument("--hdf5", default="dataset/multi_egg.hdf5")
    ap.add_argument("--demo", default=None, help="demo key; default = first")
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

    with h5py.File(args.hdf5, "r") as f:
        demo = args.demo or list(f["data"].keys())[0]
        obs = f["data"][demo]["obs"]
        cam = obs["table_cam"]
        wrist_cam = None
        if cfg.use_wrist:
            if args.wrist_key not in obs:
                raise KeyError(
                    f"checkpoint uses wrist input, but demo {demo!r} has no "
                    f"obs/{args.wrist_key!s}"
                )
            wrist_cam = obs[args.wrist_key]
        T = cam.shape[0]
        for t in args.t:
            if t + H >= T:
                print(f"skip t={t}: t+{H} beyond episode length {T}")
                continue
            raw_t, raw_T = cam[t], cam[t + H]                     # (H,W,3) u8
            raw_wrist = wrist_cam[t] if wrist_cam is not None else None

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
                    (img_t, None, f"table o_t ({demo} t={t})"),
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
                    (img_t, None, f"table o_t ({demo} t={t})"),
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
            out = os.path.join(args.out_dir, f"{demo}_t{t}.png")
            wrist_note = " + wrist" if wt is not None else ""
            fig.suptitle(
                f"Mini-LaWAM DINO feature-space report{wrist_note} — "
                f"{demo}, t={t}, horizon={H}",
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
