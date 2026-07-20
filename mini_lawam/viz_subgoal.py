"""Visualize the LaWM-predicted subgoal u_hat_T = LaWM(u_t, ConvPrior(u_t)).

The subgoal lives in DINO feature space (256 tokens x 768), so it cannot be
rendered as pixels directly. Instead each sample gets a 2x4 panel:

    row 1:  o_t          | o_T (true future) | pred-change heatmap | true-change heatmap
    row 2:  PCA(u_t)     | PCA(u_hat_T)      | PCA(u_T true)       | cos(u_hat_T, u_T) map

    pred-change = per-patch ||u_hat_T - u_t||  (where the model THINKS motion happens)
    true-change = per-patch ||u_T     - u_t||  (where motion actually happens)
    PCA maps share one projection, so same color = similar feature.

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
    x = x - x.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=3)
    outs = []
    for f in feats:
        p = (f - f.mean(0, keepdim=True)) @ v[:, :3]  # [K, 3]
        p = (p - p.min(0).values) / (p.max(0).values - p.min(0).values + 1e-8)
        outs.append(to_grid(p))
    return outs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/mini_lawam/prior_phase1.pt")
    ap.add_argument("--hdf5", default="dataset/multi_egg.hdf5")
    ap.add_argument("--demo", default=None, help="demo key; default = first")
    ap.add_argument("--t", type=int, nargs="+", default=[0],
                    help="one or more start frames, e.g. --t 0 40 80")
    ap.add_argument("--out-dir", default="results/mini_lawam/subgoal_viz")
    args = ap.parse_args()

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
        cam = f["data"][demo]["obs"]["table_cam"]
        T = cam.shape[0]
        for t in args.t:
            if t + H >= T:
                print(f"skip t={t}: t+{H} beyond episode length {T}")
                continue
            raw_t, raw_T = cam[t], cam[t + H]                     # (H,W,3) u8

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
            pca_ut, pca_uh, pca_uT = joint_pca_rgb([ut, uh, uT])
            chg_pred = to_grid((uh - ut).norm(dim=-1))            # [16,16]
            chg_true = to_grid((uT - ut).norm(dim=-1))
            agree = to_grid(F.cosine_similarity(uh, uT, dim=-1))  # [16,16]
            vmax = max(chg_pred.max(), chg_true.max())
            cos_full = float(F.cosine_similarity(uh.flatten(), uT.flatten(), dim=0))

            img_t = np.asarray(prep(raw_t).permute(1, 2, 0))
            img_T = np.asarray(prep(raw_T).permute(1, 2, 0))
            panels = [
                (img_t, None, f"o_t  ({demo} t={t})"),
                (img_T, None, f"o_T true (t={t + H})"),
                (chg_pred, "pred change ||u_hat_T - u_t||", "where model PREDICTS motion"),
                (chg_true, "true change ||u_T - u_t||", "where motion ACTUALLY is"),
                (pca_ut, None, "PCA(u_t)"),
                (pca_uh, None, "PCA(u_hat_T)  <- predicted subgoal"),
                (pca_uT, None, "PCA(u_T true)"),
                (agree, "cos(u_hat_T, u_T)", f"agreement (full-frame cos={cos_full:.3f})"),
            ]
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
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
            fig.tight_layout()
            fig.savefig(out, dpi=110)
            plt.close(fig)
            print(f"saved {out}  (full-frame cos(u_hat_T, u_T) = {cos_full:.3f})")


if __name__ == "__main__":
    main()
