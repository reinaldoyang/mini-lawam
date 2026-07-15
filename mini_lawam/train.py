"""Minimal single-GPU training loop for the LaWAM-inspired BC policy.

Trains only the ConvPrior + MLP action head; DINO and LaWM stay frozen.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train \
        --hdf5 dataset/demo_dataset_100.hdf5 --steps 20000 --batch 32
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from mini_lawam.data import MiniLaWAMDataset, split_o_t_o_T
from mini_lawam.model import MiniLaWAM, MiniLaWAMConfig


def make_loaders(ds, batch, workers, val_frac, seed=0):
    n = len(ds)
    g = np.random.default_rng(seed)
    perm = g.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    dl = lambda idx, sh: DataLoader(  # noqa: E731
        Subset(ds, idx), batch_size=batch, shuffle=sh, num_workers=workers,
        pin_memory=True, drop_last=sh, persistent_workers=workers > 0,
    )
    return dl(train_idx, True), dl(val_idx, False)


def to_inputs(batch, device):
    frames_u8 = batch["frames_u8"].to(device, non_blocking=True)      # [B,2,3,256,256] u8
    vids, _ = gpu_two_view_video_aug(frames_u8, training=False)       # normalize on GPU
    o_t, o_T = split_o_t_o_T(vids)
    actions = batch["actions"].to(device, non_blocking=True)
    mask = batch["actions_mask"].to(device, non_blocking=True)
    return o_t, o_T, actions, mask


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.prior.eval(); model.action_head.eval()
    tot = {}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        o_t, o_T, actions, mask = to_inputs(batch, device)
        out = model(o_t, o_T, actions, actions_mask=mask)
        for k, v in out.items():
            if k != "pred":
                tot[k] = tot.get(k, 0.0) + float(v)
    model.prior.train(); model.action_head.train()
    n = min(max_batches, len(loader)) or 1
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="dataset/multi_egg.hdf5")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--sample-stride", type=int, default=2,
                    help="Subsample start frames to cut redundancy between neighbors.")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", default="results/mini_lawam/ckpt.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MiniLaWAMConfig()

    ds = MiniLaWAMDataset(
        args.hdf5, gap=cfg.action_horizon, horizon=cfg.action_horizon,
        sample_stride=args.sample_stride,
    )
    print(f"dataset: {len(ds)} pairs | action stats mean={np.round(ds.action_mean,4)} "
          f"std={np.round(ds.action_std,4)}")
    train_loader, val_loader = make_loaders(ds, args.batch, args.workers, args.val_frac)

    model = MiniLaWAM(cfg).to(device)
    model.prior.train(); model.action_head.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    step, best_val = 0, float("inf")
    while step < args.steps:
        for batch in train_loader:
            o_t, o_T, actions, mask = to_inputs(batch, device)
            out = model(o_t, o_T, actions, actions_mask=mask)
            opt.zero_grad(set_to_none=True)
            out["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            step += 1

            if step % args.log_every == 0:
                print(f"step {step:>6} | total {float(out['loss_total']):.4f} "
                      f"act {float(out['loss_act']):.4f} distill {float(out['loss_distill']):.4f} "
                      f"wm {float(out['loss_wm']):.4f} | lr {sched.get_last_lr()[0]:.2e}")
            if step % args.eval_every == 0:
                val = evaluate(model, val_loader, device)
                print(f"  [val] " + " ".join(f"{k}={v:.4f}" for k, v in val.items()))
                if val.get("loss_act", 1e9) < best_val:
                    best_val = val["loss_act"]
                    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                    torch.save({
                        "prior": model.prior.state_dict(),
                        "action_head": model.action_head.state_dict(),
                        "cfg": cfg.__dict__,
                        "action_mean": ds.action_mean,
                        "action_std": ds.action_std,
                        "step": step,
                    }, args.out)
                    print(f"  [ckpt] saved best (val loss_act={best_val:.4f}) -> {args.out}")
            if step >= args.steps:
                break
    print("done.")


if __name__ == "__main__":
    main()
