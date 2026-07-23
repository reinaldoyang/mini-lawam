"""Modular rollout adapter for the mini_lawam BC policy.

This is the venue-INDEPENDENT core of deployment. It goes:

    raw camera frame (H,W,3 uint8)
        -> preprocess (resize 256, ImageNet-normalize)   [matches training exactly]
        -> model.predict()                                [z-scored action chunk]
        -> un-normalize (pred * action_std + action_mean) [physical units]
        -> 4D action chunk in PHYSICAL units: [eef_pos_base(3), gripper(1)]

It STOPS there. Turning a 4D action into an actual robot/sim command
(pose->servoL delta on real UR7e, or controller target in sim), choosing how many
chunk steps to execute before re-planning, and where frames come from are all
VENUE-SPECIFIC and intentionally left out -- see the `>>> SEAM` markers below.

Usage:
    policy = MiniLaWAMPolicy("results/mini_lawam/ckpt.pt", device="cuda")
    chunk = policy.act(frame_hwc_uint8)     # np.ndarray [H, 4], physical units
    # ... your sim/real code turns `chunk` into commands ...
"""

from typing import Optional

import numpy as np
import torch
from torchvision.transforms import v2

from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from mini_lawam.model import MiniLaWAM, MiniLaWAMConfig


class MiniLaWAMPolicy:
    """Loads a trained checkpoint and maps frames -> physical 4D action chunks."""

    def __init__(self, ckpt_path: str, device: Optional[str] = None,
                 image_hw=(256, 256), train_frame_hw=(168, 224)):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Rebuild the model from the saved config, load only the trained heads.
        cfg = MiniLaWAMConfig(**ckpt["cfg"])
        self.cfg = cfg
        self.model = MiniLaWAM(cfg).to(self.device)
        self.model.prior.load_state_dict(ckpt["prior"])
        self.model.action_head.load_state_dict(ckpt["action_head"])
        self.model.eval()

        # De-normalization stats (per-dim), saved at train time. [eef_pos(3), grip(1)]
        self.action_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)  # [4]
        self.action_std = np.asarray(ckpt["action_std"], dtype=np.float32)    # [4]

        # Same resize the Dataset applied before normalization (data.py:_frame).
        self.resize = v2.Resize(image_hw, antialias=True)
        self.image_hw = image_hw
        # Training images were RECORDED at train_frame_hw (record_real.py stores
        # 224x168), then upscaled to 256 by the Dataset. A live 640x480 camera
        # frame downscaled straight to 256 has different sharpness/aliasing, so
        # first drop live frames to the recorded size to match training
        # statistics exactly. None disables; dataset-sized inputs are a no-op.
        self.train_frame_hw = tuple(train_frame_hw) if train_frame_hw else None
        self.pre_resize = (v2.Resize(self.train_frame_hw, antialias=True)
                           if self.train_frame_hw else None)

    # ---- preprocessing: raw frame -> model input (must match training) ----
    def preprocess(self, frame_hwc_uint8: np.ndarray) -> torch.Tensor:
        """(H,W,3) uint8 -> o_t float [1,1,3,256,256], ImageNet-normalized.

        Mirrors MiniLaWAMDataset._frame (resize->uint8) + train.to_inputs
        (gpu_two_view_video_aug(training=False)) so features match training.
        """
        x = torch.from_numpy(np.ascontiguousarray(frame_hwc_uint8)).permute(2, 0, 1)  # [3,H,W]
        if self.pre_resize is not None and tuple(x.shape[-2:]) != self.train_frame_hw:
            x = self.pre_resize(x).to(torch.uint8)            # live cam -> recorded size
        x = self.resize(x).to(torch.uint8)                    # [3,256,256]
        frames_u8 = x.view(1, 1, 3, *self.image_hw).to(self.device)  # [B=1,T=1,3,256,256]
        vids, _ = gpu_two_view_video_aug(frames_u8, training=False)  # normalize on GPU
        return vids                                            # [1,1,3,256,256]

    def model_input_u8(self, frame_hwc_uint8: np.ndarray) -> np.ndarray:
        """The 256x256 uint8 image the policy ACTUALLY ingests (pre-normalization).

        Same resize path as preprocess() but stops before ImageNet-normalizing, so
        you can SEE exactly what the model sees (incl. the --train-frame-hw downscale).
        """
        x = torch.from_numpy(np.ascontiguousarray(frame_hwc_uint8)).permute(2, 0, 1)
        if self.pre_resize is not None and tuple(x.shape[-2:]) != self.train_frame_hw:
            x = self.pre_resize(x).to(torch.uint8)
        x = self.resize(x).to(torch.uint8)
        return x.permute(1, 2, 0).numpy()                      # HWC uint8 256x256

    # ---- the venue-independent output ----
    @torch.no_grad()
    def act(self, frame_hwc_uint8: np.ndarray,
            wrist_hwc_uint8: Optional[np.ndarray] = None,
            state_xyz: Optional[np.ndarray] = None) -> np.ndarray:
        """Frame -> physical 4D action chunk [H, 4] = [eef_pos_base(3), gripper(1)].

        Absolute EEF positions (meters, base frame) + raw gripper channel. This is
        the clean hand-off point. Downstream (venue-specific) code decides what to
        do with it. If the checkpoint was trained with use_wrist=True, you MUST pass
        `wrist_hwc_uint8` (the aux wrist_cam frame at the same timestep). If trained
        with use_state=True, pass `state_xyz` = current eef_pos [x,y,z] in meters
        (base frame); it is z-scored here with the same stats as training.
        """
        o_t = self.preprocess(frame_hwc_uint8)
        wrist = None
        if self.cfg.use_wrist:
            if wrist_hwc_uint8 is None:
                raise ValueError("checkpoint trained with use_wrist=True -> pass wrist_hwc_uint8")
            wrist = self.preprocess(wrist_hwc_uint8)
        state = None
        if getattr(self.cfg, "use_state", False):
            if state_xyz is None:
                raise ValueError("checkpoint trained with use_state=True -> pass state_xyz")
            s = (np.asarray(state_xyz, np.float32) - self.action_mean[:3]) / self.action_std[:3]
            state = torch.from_numpy(s).view(1, 3).to(self.device)
        pred = self.model.predict(o_t, wrist=wrist, state=state)    # [1,H,4], z-scored
        pred = pred[0].cpu().numpy().astype(np.float32)
        chunk = pred * self.action_std + self.action_mean  # un-normalize -> [H,4]
        if getattr(self.cfg, "target_mode", "abs") == "delta":
            # Delta targets: compose absolute positions from the CURRENT eef pos.
            # Each replan re-anchors at the true arm position -> servo-like loop.
            if state_xyz is None:
                raise ValueError("checkpoint trained with target_mode='delta' -> pass "
                                 "state_xyz (current eef xyz) to compose absolute targets")
            chunk[:, :3] += np.asarray(state_xyz, np.float32)
        return chunk

    # >>> SEAM (venue-specific): implement per setup, do NOT bake in here.
    #   - target -> command:  real UR7e: delta = eef_pos - current_TCP -> servoL;
    #                         sim:       set controller target pose / joint cmd.
    #   - gripper:            threshold the 4th channel at 0 (< 0 open, > 0 close).
    #   - execution cadence:  how many of the H steps to run before re-planning
    #                         (receding horizon), and the observation source loop.
    # You said you'll prompt with your integration plan -- these stay unimplemented
    # until then so no real/sim assumptions leak into the shared core.


if __name__ == "__main__":
    # Offline sanity check: pull real frames from the dataset, run act(), and
    # compare the FIRST predicted step against the ground-truth target at t+1.
    # This validates the whole inference path WITHOUT any sim/real robot.
    #   --mode single : one frame, print predicted vs GT chunk (quick eyeball).
    #   --mode eval   : batched error over many frames, train vs val (real verdict).
    import argparse
    import h5py

    from mini_lawam.data import MiniLaWAMDataset, _read_target

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/mini_lawam/ckpt.pt")
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--mode", choices=["single", "eval"], default="eval")
    # single-mode:
    ap.add_argument("--demo", default=None, help="[single] demo key; default = first")
    ap.add_argument("--t", type=int, default=0, help="[single] start frame index")
    # eval-mode (must match train.py to reproduce its train/val split):
    ap.add_argument("--n-samples", type=int, default=200,
                    help="[eval] frames sampled per split")
    ap.add_argument("--sample-stride", type=int, default=2, help="[eval] match train.py")
    ap.add_argument("--val-frac", type=float, default=0.05, help="[eval] match train.py")
    ap.add_argument("--split-seed", type=int, default=0, help="[eval] match train.py seed")
    args = ap.parse_args()

    policy = MiniLaWAMPolicy(args.ckpt)
    step = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("step")
    H = policy.cfg.action_horizon
    print(f"loaded ckpt (step {step}) | action_dim={policy.cfg.action_dim} horizon={H} "
          f"use_wrist={policy.cfg.use_wrist}")
    print(f"action_mean={np.round(policy.action_mean,4)} action_std={np.round(policy.action_std,4)}")

    if args.mode == "single":
        with h5py.File(args.hdf5, "r") as f:
            demo = args.demo or list(f["data"].keys())[0]
            g = f["data"][demo]
            frame = g["obs"]["table_cam"][args.t]
            wrist = g["obs"]["wrist_cam"][args.t] if policy.cfg.use_wrist else None
            # current eef pos: proprioception (use_state) and/or delta anchor
            need_xyz = (getattr(policy.cfg, "use_state", False)
                        or getattr(policy.cfg, "target_mode", "abs") == "delta")
            st = (g["obs"]["eef_pos_base"][args.t].astype(np.float32)
                  if need_xyz else None)
            gt_pos = g["obs"]["eef_pos_base"][args.t + 1].astype(np.float32)
            gt_grip = float(g["actions"][args.t + 1, 6])
        chunk = policy.act(frame, wrist, state_xyz=st)
        print(f"\ndemo={demo} t={args.t}  chunk shape={chunk.shape}")
        print(f"pred step0 : eef_pos={np.round(chunk[0,:3],4)}  gripper={chunk[0,3]:+.3f}")
        print(f"gt   step0 : eef_pos={np.round(gt_pos,4)}  gripper={gt_grip:+.3f}")
        print(f"pos L2 err : {np.linalg.norm(chunk[0,:3]-gt_pos):.4f} m")
        raise SystemExit(0)

    # ---- eval mode: reproduce train.py's split, then measure error per split ----
    ds = MiniLaWAMDataset(args.hdf5, gap=H, horizon=H, sample_stride=args.sample_stride)
    n = len(ds)
    perm = np.random.default_rng(args.split_seed).permutation(n)
    n_val = max(1, int(n * args.val_frac))
    splits = {"val": perm[:n_val], "train": perm[n_val:]}
    pick = np.random.default_rng(1)  # deterministic subsample of each split

    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        for split, idxs in splits.items():
            take = idxs if len(idxs) <= args.n_samples else \
                pick.choice(idxs, args.n_samples, replace=False)
            pos_l2, step0_l2, mae, grip_ok, cnt = 0.0, 0.0, np.zeros(4), 0, 0
            for i in take:
                demo, t = ds.index[int(i)]
                g = data[demo]
                frame = g["obs"]["table_cam"][t]                       # (H,W,3) u8
                wrist = g["obs"]["wrist_cam"][t] if policy.cfg.use_wrist else None
                need_xyz = (getattr(policy.cfg, "use_state", False)
                            or getattr(policy.cfg, "target_mode", "abs") == "delta")
                st = (g["obs"]["eef_pos_base"][t].astype(np.float32)
                      if need_xyz else None)
                gt = _read_target(g, t + 1, H, "eef_pos_base", 6)      # [H,4] physical
                pred = policy.act(frame, wrist, state_xyz=st)          # [H,4] physical
                pos_l2 += np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1).mean()
                step0_l2 += np.linalg.norm(pred[0, :3] - gt[0, :3])
                mae += np.abs(pred - gt).mean(axis=0)
                grip_ok += float((np.sign(pred[:, 3]) == np.sign(gt[:, 3])).mean())
                cnt += 1
            print(f"\n[{split}]  n={cnt}  (of {len(idxs)} in split)")
            print(f"  pos L2 err  (mean over horizon): {pos_l2/cnt*100:.2f} cm")
            print(f"  pos L2 err  (step 0 only)      : {step0_l2/cnt*100:.2f} cm")
            print(f"  per-dim MAE [x y z](m) grip    : {np.round(mae/cnt,4)}")
            print(f"  gripper sign accuracy          : {grip_ok/cnt*100:.1f}%")
