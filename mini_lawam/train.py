"""Minimal single-GPU training loop for the LaWAM-inspired BC policy.

DINO, the LAM IDM (teacher), and the LaWM decoder are ALWAYS frozen.

Modes (--phase):
    1     : train ConvPrior only.  loss = L_distill = MSE(z_hat, z_teacher).
            Action head untouched. Saves a prior-only checkpoint
            (default results/mini_lawam/prior_phase1.pt), best on val loss_distill.
    2     : load the phase-1 prior (--prior-ckpt), train the action head.
            Prior frozen by default; --finetune-prior keeps it trainable.
            loss = L_act + 0.1*L_distill + 0.1*L_wm (weights overridable).
            Saves the full rollout checkpoint, best on val loss_act.
    joint : original single-phase behavior (prior + head together,
            L_act + 1.0*L_distill + 0.1*L_wm).

Example:
    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train --hdf5 dataset/multi_egg.hdf5 \
        --phase 1 --steps 10000 --batch 32
    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train --hdf5 dataset/multi_egg.hdf5 \
        --phase 2 --prior-ckpt results/mini_lawam/prior_phase1.pt --steps 20000 --batch 32
    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train --hdf5 dataset/multi_egg.hdf5 \
        --phase 2 --head attn --gripper-head binary --target joystick \
        --prior-ckpt results/mini_lawam/prior_phase1.pt --steps 20000 --batch 32
"""

import argparse
import csv
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
    gripper_targets = batch["gripper_targets"].to(device, non_blocking=True)
    wrist = None
    if "wrist_u8" in batch:
        w_u8 = batch["wrist_u8"].to(device, non_blocking=True).unsqueeze(1)  # [B,1,3,256,256]
        wrist, _ = gpu_two_view_video_aug(w_u8, training=False)              # same ImageNet norm
    state = batch["state"].to(device, non_blocking=True) if "state" in batch else None
    return o_t, o_T, actions, mask, gripper_targets, wrist, state


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20, prior_only=False, set_train_mode=None):
    model.prior.eval(); model.action_head.eval()
    tot = {}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        o_t, o_T, actions, mask, gripper_targets, wrist, state = to_inputs(
            batch, device
        )
        out = model(
            o_t, o_T, actions, actions_mask=mask,
            gripper_targets=gripper_targets, wrist=wrist, state=state,
            prior_only=prior_only,
        )
        for k, v in out.items():
            if k != "pred":
                tot[k] = tot.get(k, 0.0) + float(v)
    if set_train_mode is not None:
        set_train_mode(True)  # restore per-phase train/eval flags
    else:
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
    ap.add_argument("--horizon", type=int, default=24,
                    help="Horizon in FRAMES for BOTH the LaWM future pair (o_{t+H} -> "
                         "z_teacher, u_T, loss_wm) and the action chunk. "
                         "1.2s @ 20Hz = 24 (paper §C.5).")
    ap.add_argument("--use-wrist", action="store_true",
                    help="Add wrist_cam as an aux view to the action head (paper §C.2).")
    ap.add_argument("--head", choices=["mlp", "attn"], default="mlp",
                    help="Action head: 'mlp' = pooled-features MLP (v0); 'attn' = "
                         "token-level cross-attention (no mean-pooling, fixes precision).")
    ap.add_argument(
        "--gripper-head", choices=["regression", "binary"], default="regression",
        help="'regression' preserves the legacy joint 4D MSE head/checkpoints; "
             "'binary' uses separate XYZ regression and open/close-logit projections "
             "with BCE loss. Use binary for new gripper-focused training.",
    )
    ap.add_argument("--use-state", action="store_true",
                    help="Feed proprioception (current eef_pos, z-scored) to the head. "
                         "Helps 'how far to descend' but risks BC copycat -- try both.")
    ap.add_argument("--target", choices=["abs", "delta", "joystick"], default="abs",
                    help="Action target: 'abs' = absolute eef positions (v0); 'delta' = "
                         "pos[t+i]-pos[t] relative to the current frame. Delta composes "
                         "as current_TCP + prediction at deploy (servo-like; immune to "
                         "systematic absolute-position bias); 'joystick' = raw HDF5 "
                         "actions[t+i,0:3] plus actions[t+i,6] gripper. Joystick XYZ is "
                         "scaled and composed with the live TCP only at deployment.")
    ap.add_argument("--phase", choices=["1", "2", "joint"], default="joint",
                    help="1: train prior only (L_distill). 2: load --prior-ckpt, train "
                         "action head (L_act + 0.1*L_distill + 0.1*L_wm). joint: original "
                         "single-phase training.")
    ap.add_argument("--prior-ckpt", default="results/mini_lawam/prior_phase1.pt",
                    help="Phase 2: path to the phase-1 prior checkpoint to load.")
    ap.add_argument("--finetune-prior", action="store_true",
                    help="Phase 2: keep the loaded prior trainable (default: frozen).")
    ap.add_argument("--lambda-distill", type=float, default=None,
                    help="Override distill weight (default: 1.0 joint, 0.1 phase 2).")
    ap.add_argument("--lambda-wm", type=float, default=None,
                    help="Override wm/subgoal weight (default: 0.1).")
    ap.add_argument(
        "--lambda-gripper", type=float, default=1.0,
        help="Binary gripper BCE weight relative to normalized XYZ MSE (default: 1.0).",
    )
    ap.add_argument(
        "--gripper-target-offset", type=int, choices=[0, 1], default=None,
        help="Gripper label row relative to observation t, independent of XYZ: "
             "0=same row, 1=one row ahead. Default preserves the historical "
             "contract (joystick=0, abs/delta=1). Use 1 for the recommended "
             "joystick-binary release timing.",
    )
    ap.add_argument(
        "--include-tail-actions", action="store_true",
        help="Include anchors in the final action horizon of each demo. Short action "
             "chunks are right-padded/masked and the LaWM future image is clamped to "
             "the terminal frame. Recommended for phase-2 place/release training.",
    )
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", default=None,
                    help="Checkpoint path (default: results/mini_lawam/prior_phase1.pt "
                         "for phase 1, results/mini_lawam/ckpt.pt otherwise).")
    ap.add_argument("--csv-log", default="results/mini_lawam/train_log.csv",
                    help="Per-step metric log (always written). Plot via mini_lawam.plot_log.")
    ap.add_argument("--wandb", action="store_true",
                    help="Also log to Weights & Biases (online; requires `wandb login`).")
    ap.add_argument("--wandb-project", default="mini_lawam")
    ap.add_argument("--wandb-offline", action="store_true",
                    help="Log wandb offline instead of online (no login; sync later).")
    ap.add_argument("--run-name", default=None, help="wandb/CSV run name.")
    args = ap.parse_args()

    # Fail fast on wandb login BEFORE the expensive dataset/model setup.
    if args.wandb and not args.wandb_offline:
        import wandb
        if not wandb.api.api_key:
            raise SystemExit(
                "[wandb] online logging requested but you are not logged in.\n"
                "        Run `wandb login` (or set WANDB_API_KEY), or pass "
                "--wandb-offline to log locally."
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- phase resolution ---
    prior_only = args.phase == "1"                      # phase 1: L_distill only
    train_prior = args.phase != "2" or args.finetune_prior
    if args.out is None:
        args.out = ("results/mini_lawam/prior_phase1.pt" if prior_only
                    else "results/mini_lawam/ckpt.pt")
    lambda_distill = args.lambda_distill if args.lambda_distill is not None else (
        0.1 if args.phase == "2" else 1.0)
    lambda_wm = args.lambda_wm if args.lambda_wm is not None else 0.1
    gripper_target_offset = (
        int(args.gripper_target_offset)
        if args.gripper_target_offset is not None
        else (0 if args.target == "joystick" else 1)
    )
    print(f"phase={args.phase} | prior {'trains' if train_prior else 'FROZEN'} | "
          f"action head {'skipped' if prior_only else 'trains'} | "
          f"lambda_distill={lambda_distill} lambda_wm={lambda_wm} "
          f"lambda_gripper={args.lambda_gripper} "
          f"gripper_target_offset={gripper_target_offset} "
          f"include_tail_actions={args.include_tail_actions} | out={args.out}")

    if args.use_state and args.target != "abs":
        raise SystemExit(f"--use-state + --target {args.target} unsupported: the checkpoint "
                         "does not store absolute-position stats, so it cannot z-score "
                         "an absolute TCP state.")

    # One horizon for both the LaWM future pair and the action chunk.
    state_dim = 3 if args.use_state else 0   # proprioception = current eef_pos [x,y,z]
    cfg = MiniLaWAMConfig(use_wrist=args.use_wrist, head_type=args.head,
                          gripper_head=args.gripper_head,
                          use_state=args.use_state, state_dim=state_dim,
                          target_mode=args.target,
                          gripper_target_offset=gripper_target_offset,
                          include_tail_actions=args.include_tail_actions,
                          future_horizon=args.horizon, action_horizon=args.horizon,
                          lambda_gripper=args.lambda_gripper,
                          lambda_distill=lambda_distill, lambda_wm=lambda_wm)
    print(f"head={args.head} | use_wrist={args.use_wrist} | use_state={args.use_state} "
          f"| gripper_head={args.gripper_head} | target={args.target}")

    # gap = future horizon (LaWM pair, o_{t+future_horizon}); horizon = action chunk.
    ds = MiniLaWAMDataset(
        args.hdf5, gap=cfg.future_horizon, horizon=cfg.action_horizon,
        sample_stride=args.sample_stride, use_wrist=args.use_wrist,
        use_state=args.use_state, target_mode=args.target,
        gripper_target_offset=cfg.gripper_target_offset,
        include_tail_actions=cfg.include_tail_actions,
    )
    print(f"horizons: future(LaWM)={cfg.future_horizon}  action_chunk={cfg.action_horizon} "
          f"(gap between o_t and o_T = {cfg.future_horizon} frames)")
    print(f"dataset: {len(ds)} pairs | action stats mean={np.round(ds.action_mean,4)} "
          f"std={np.round(ds.action_std,4)}")
    train_loader, val_loader = make_loaders(ds, args.batch, args.workers, args.val_frac)

    model = MiniLaWAM(cfg).to(device)
    if prior_only:
        # Phase 1: action head is untouched (not in the optimizer, never run).
        model.action_head.requires_grad_(False)
    if args.phase == "2":
        prior_sd = torch.load(args.prior_ckpt, map_location="cpu", weights_only=False)
        model.prior.load_state_dict(prior_sd["prior"])
        print(f"[phase 2] loaded prior from {args.prior_ckpt} "
              f"(phase-1 step {prior_sd.get('step', '?')})")
        if not args.finetune_prior:
            model.prior.requires_grad_(False)
            model.prior.eval()

    def set_train_mode(training: bool):
        # Frozen prior stays in eval() so its GroupNorm/etc. behave as at load time.
        model.prior.train(training and train_prior)
        model.action_head.train(training and not prior_only)

    set_train_mode(True)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    # --- logging: CSV always, wandb optional ---
    os.makedirs(os.path.dirname(args.csv_log) or ".", exist_ok=True)
    csv_file = open(args.csv_log, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "split", "loss_total", "loss_act", "loss_xyz", "loss_gripper",
        "gripper_accuracy", "loss_distill", "loss_wm", "lr",
    ])
    csv_file.flush()

    def log_row(step, split, metrics, lr=""):
        csv_writer.writerow([
            step, split,
            metrics.get("loss_total", ""), metrics.get("loss_act", ""),
            metrics.get("loss_xyz", ""), metrics.get("loss_gripper", ""),
            metrics.get("gripper_accuracy", ""),
            metrics.get("loss_distill", ""), metrics.get("loss_wm", ""), lr,
        ])
        csv_file.flush()

    run = None
    if args.wandb:
        import wandb
        mode = "offline" if args.wandb_offline else "online"
        run = wandb.init(
            project=args.wandb_project, name=args.run_name, mode=mode,
            config={**cfg.__dict__, "steps": args.steps, "batch": args.batch,
                    "lr": args.lr, "hdf5": args.hdf5, "n_pairs": len(ds),
                    "phase": args.phase, "finetune_prior": args.finetune_prior},
        )
        print(f"[wandb] logging ({mode}) project={args.wandb_project}")

    # Phase 1 selects best on val loss_distill; phases 2/joint on val loss_act.
    best_key = "loss_distill" if prior_only else "loss_act"
    step, best_val = 0, float("inf")
    while step < args.steps:
        for batch in train_loader:
            o_t, o_T, actions, mask, gripper_targets, wrist, state = to_inputs(
                batch, device
            )
            out = model(
                o_t, o_T, actions, actions_mask=mask,
                gripper_targets=gripper_targets, wrist=wrist, state=state,
                prior_only=prior_only,
            )
            opt.zero_grad(set_to_none=True)
            out["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            step += 1

            if step % args.log_every == 0:
                lr = sched.get_last_lr()[0]
                train_m = {k: float(v) for k, v in out.items() if k != "pred"}
                print(f"step {step:>6} | total {train_m['loss_total']:.4f} "
                      f"act {train_m['loss_act']:.4f} distill {train_m['loss_distill']:.4f} "
                      f"wm {train_m['loss_wm']:.4f}"
                      + (
                          f" xyz {train_m['loss_xyz']:.4f} "
                          f"grip {train_m['loss_gripper']:.4f} "
                          f"grip_acc {train_m['gripper_accuracy'] * 100:.1f}%"
                          if "gripper_accuracy" in train_m else ""
                      )
                      + f" | lr {lr:.2e}")
                log_row(step, "train", train_m, lr)
                if run is not None:
                    run.log({**{f"train/{k}": v for k, v in train_m.items()},
                             "train/lr": lr}, step=step)
            if step % args.eval_every == 0:
                val = evaluate(model, val_loader, device,
                               prior_only=prior_only, set_train_mode=set_train_mode)
                print(f"  [val] " + " ".join(f"{k}={v:.4f}" for k, v in val.items()))
                log_row(step, "val", val)
                if run is not None:
                    run.log({f"val/{k}": v for k, v in val.items()}, step=step)
                if val.get(best_key, 1e9) < best_val:
                    best_val = val[best_key]
                    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                    ckpt = {
                        "prior": model.prior.state_dict(),
                        "cfg": cfg.__dict__,
                        "step": step,
                        "phase": args.phase,
                    }
                    if not prior_only:
                        # Full rollout checkpoint (same keys as before).
                        ckpt.update(
                            action_head=model.action_head.state_dict(),
                            action_mean=ds.action_mean,
                            action_std=ds.action_std,
                        )
                    torch.save(ckpt, args.out)
                    print(f"  [ckpt] saved best (val {best_key}={best_val:.4f}) -> {args.out}")
            if step >= args.steps:
                break
    csv_file.close()
    if run is not None:
        run.finish()
    print(f"done. metrics -> {args.csv_log}"
          + ("  (wandb run saved)" if run is not None else ""))


if __name__ == "__main__":
    main()
