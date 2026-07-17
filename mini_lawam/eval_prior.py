"""Check whether the ConvPrior actually learned from the IDM teacher.

Run this on a phase-1 prior checkpoint BEFORE starting phase 2 (also works on
joint/phase-2 checkpoints -- anything with a "prior" key).

Reports, over the val split:
    distill_mse      MSE(z_hat, z_teacher)                    <- raw phase-1 loss
    mean_base_mse    MSE(mean_z, z_teacher)                   <- "predict the average"
    shuffle_mse      MSE(z_hat, z_teacher of another sample)  <- chance pairing
    R^2              1 - distill_mse / mean_base_mse
    cos_z            cosine(z_hat, z_teacher)

    wm_cos_pred      cos(LaWM(u_t, z_hat),     u_T)  <- what phase 2 conditions on
    wm_cos_oracle    cos(LaWM(u_t, z_teacher), u_T)  <- ceiling (LaWM itself)
    wm_cos_copy      cos(u_t,                  u_T)  <- floor  (no-motion baseline)

Read it as:
    R^2 near 0 / distill_mse ~ shuffle_mse  -> prior learned nothing sample-specific.
    wm_cos_pred close to wm_cos_oracle and clearly above wm_cos_copy
                                            -> prior is good enough for phase 2.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.eval_prior \
        --ckpt results/mini_lawam/prior_phase1.pt --hdf5 dataset/multi_egg.hdf5
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from mini_lawam.data import MiniLaWAMDataset, split_o_t_o_T
from mini_lawam.model import MiniLaWAM, MiniLaWAMConfig


def flat_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine over flattened [B, ...] tensors -> [B]."""
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/mini_lawam/prior_phase1.pt")
    ap.add_argument("--hdf5", default="dataset/multi_egg.hdf5")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="Same split convention as train.py (seed 0).")
    ap.add_argument("--split", choices=["val", "all"], default="val")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = no limit.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MiniLaWAMConfig(**ck["cfg"])
    print(f"ckpt: {args.ckpt} (phase={ck.get('phase', '?')}, step={ck.get('step', '?')}) | "
          f"future_horizon={cfg.future_horizon}")

    model = MiniLaWAM(cfg).to(device)
    model.prior.load_state_dict(ck["prior"])
    model.prior.eval()

    ds = MiniLaWAMDataset(args.hdf5, gap=cfg.future_horizon, horizon=cfg.action_horizon,
                          sample_stride=2, use_wrist=False)
    if args.split == "val":
        # Same val indices as train.py: seed-0 permutation, first val_frac.
        perm = np.random.default_rng(0).permutation(len(ds))
        idx = perm[:max(1, int(len(ds) * args.val_frac))].tolist()
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f"eval samples: {len(ds)} ({args.split} split)")

    zs_hat, zs_teacher = [], []
    wm_pred, wm_oracle, wm_copy, n = 0.0, 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if args.max_batches and i >= args.max_batches:
            break
        frames = batch["frames_u8"].to(device, non_blocking=True)
        vids, _ = gpu_two_view_video_aug(frames, training=False)
        o_t, o_T = split_o_t_o_T(vids)

        u_t = model._feat(o_t)[:, :1]
        pair = torch.cat([o_t, o_T], dim=1)
        z_teacher, u_T = model._teacher(pair)
        z_hat = model.prior(u_t[:, 0])

        def decode(z):
            out = model.lam.decoder(u_t, z)
            return out[0] if isinstance(out, tuple) else out

        b = z_hat.shape[0]
        wm_pred += float(flat_cos(decode(z_hat), u_T).sum())
        wm_oracle += float(flat_cos(decode(z_teacher), u_T).sum())
        wm_copy += float(flat_cos(u_t, u_T).sum())
        n += b
        zs_hat.append(z_hat.squeeze(1).cpu())
        zs_teacher.append(z_teacher.squeeze(1).cpu())

    z_hat = torch.cat(zs_hat)          # [N, code_dim]
    z_teacher = torch.cat(zs_teacher)  # [N, code_dim]

    distill_mse = float(F.mse_loss(z_hat, z_teacher))
    mean_base_mse = float(F.mse_loss(z_teacher.mean(0, keepdim=True).expand_as(z_teacher),
                                     z_teacher))
    perm = torch.randperm(z_teacher.shape[0], generator=torch.Generator().manual_seed(0))
    shuffle_mse = float(F.mse_loss(z_hat, z_teacher[perm]))
    r2 = 1.0 - distill_mse / max(mean_base_mse, 1e-12)
    cos_z = float(flat_cos(z_hat, z_teacher).mean())

    print("\n--- latent space (z_hat vs z_teacher) ---")
    print(f"distill_mse   = {distill_mse:.4f}")
    print(f"mean_base_mse = {mean_base_mse:.4f}   (always predict dataset-mean z)")
    print(f"shuffle_mse   = {shuffle_mse:.4f}   (z_teacher of a random other sample)")
    print(f"R^2           = {r2:.3f}   (0 = learned nothing beyond the mean)")
    print(f"cos_z         = {cos_z:.3f}")
    print("\n--- world-model subgoal (cos vs true u_T) ---")
    print(f"wm_cos_pred   = {wm_pred / n:.4f}   (LaWM with YOUR z_hat -> phase-2 input)")
    print(f"wm_cos_oracle = {wm_oracle / n:.4f}   (LaWM with teacher z -> ceiling)")
    print(f"wm_cos_copy   = {wm_copy / n:.4f}   (u_t as prediction    -> floor)")
    gap = (wm_oracle - wm_pred) / max(wm_oracle - wm_copy, 1e-12)
    print(f"\nprior recovers {100 * (1 - gap):.1f}% of the oracle-over-copy margin"
          " (>70% = good to proceed to phase 2)")


if __name__ == "__main__":
    main()
