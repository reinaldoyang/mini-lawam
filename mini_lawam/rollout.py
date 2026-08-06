"""Modular rollout adapter for the mini_lawam BC policy.

This is the venue-INDEPENDENT core of deployment. It goes:

    raw camera frame (H,W,3 uint8)
        -> preprocess (resize 256, ImageNet-normalize)   [matches training exactly]
        -> model.predict()                                [z-scored action chunk]
        -> un-normalize XYZ; legacy grip regresses physical value, binary grip
           thresholds its logit to an exact -1/+1 command
        -> action chunk in checkpoint target units:
             abs/delta -> [absolute eef_pos_base(3), gripper(1)]
             joystick  -> [raw joystick XYZ command(3), optional RZ(1), gripper(1)]

It STOPS there. Turning an action chunk into an actual robot/sim command
(pose->servoL delta on real UR7e, or controller target in sim), choosing how many
chunk steps to execute before re-planning, and where frames come from are all
VENUE-SPECIFIC and intentionally left out -- see the `>>> SEAM` markers below.

Usage:
    policy = MiniLaWAMPolicy("results/mini_lawam/ckpt.pt", device="cuda")
    chunk = policy.act(frame_hwc_uint8)     # np.ndarray [H, action_dim], physical units
    # ... your sim/real code turns `chunk` into commands ...
"""

from typing import Optional

import numpy as np
import torch
from torchvision.transforms import v2

from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
from mini_lawam.model import MiniLaWAM, MiniLaWAMConfig


def decode_action_prediction(pred, action_mean, action_std,
                             gripper_head: str = "regression") -> np.ndarray:
    """Convert raw head output to physical motion plus exact/legacy grip.

    Regression mode denormalizes every channel. Binary mode treats the final
    channel as a close logit and maps it to the
    dataset convention exactly: logit < 0 -> -1 open, logit >= 0 -> +1 close.
    """
    pred = np.asarray(pred, dtype=np.float32)
    mean = np.asarray(action_mean, dtype=np.float32)
    std = np.asarray(action_std, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[1] < 4:
        raise ValueError(f"expected raw action prediction [H,D>=4], got {pred.shape}")
    action_dim = pred.shape[1]
    if mean.shape != (action_dim,) or std.shape != (action_dim,):
        raise ValueError(
            f"expected action mean/std shape ({action_dim},), got "
            f"{mean.shape}/{std.shape}"
        )
    if gripper_head == "regression":
        return pred * std + mean
    if gripper_head != "binary":
        raise ValueError(
            f"unknown gripper_head {gripper_head!r} "
            "(use 'regression' or 'binary')"
        )
    chunk = np.empty_like(pred)
    chunk[:, :-1] = pred[:, :-1] * std[:-1] + mean[:-1]
    chunk[:, -1] = np.where(pred[:, -1] >= 0.0, 1.0, -1.0)
    return chunk


class MiniLaWAMPolicy:
    """Loads a trained checkpoint and maps frames to physical action chunks."""

    # These fields determine the modules and tensor shapes constructed by
    # MiniLaWAM.__init__. Training/deployment metadata such as target_mode and
    # loss weights may differ between hot-swapped checkpoints.
    _ARCHITECTURE_FIELDS = (
        "lam_ckpt", "lam_yaml", "action_dim", "action_horizon",
        "use_state", "state_dim", "use_wrist", "head_type",
        "gripper_head", "include_rz", "attn_hidden", "attn_layers",
        "attn_heads",
    )

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

        # Per-dimension target statistics saved at train time. The final dimension
        # is gripper; motion is XYZ or optional XYZ+RZ depending on the checkpoint.
        self.action_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(ckpt["action_std"], dtype=np.float32)
        self._validate_action_stats(cfg, self.action_mean, self.action_std)
        self.ckpt_path = str(ckpt_path)

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

    @classmethod
    def _architecture_signature(cls, cfg: MiniLaWAMConfig):
        return tuple(getattr(cfg, field) for field in cls._ARCHITECTURE_FIELDS)

    @staticmethod
    def _validate_action_stats(cfg, action_mean, action_std):
        expected = (int(cfg.action_dim),)
        if action_mean.shape != expected or action_std.shape != expected:
            raise ValueError(
                f"checkpoint action stats must have shape {expected}, got "
                f"{action_mean.shape}/{action_std.shape}"
            )

    @staticmethod
    def checkpoint_config(ckpt_path: str):
        """Read a checkpoint's runtime/model config without constructing a model."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return MiniLaWAMConfig(**ckpt["cfg"])

    def assert_hot_swap_compatible(self, cfg: MiniLaWAMConfig):
        """Reject configs that cannot reuse this policy's constructed modules."""
        if self._architecture_signature(cfg) == self._architecture_signature(self.cfg):
            return
        changed = [
            field for field in self._ARCHITECTURE_FIELDS
            if getattr(cfg, field) != getattr(self.cfg, field)
        ]
        raise ValueError(
            "task checkpoint is not hot-swap compatible; architecture fields "
            f"differ: {', '.join(changed)}"
        )

    @staticmethod
    def _assert_state_dict_compatible(module, incoming, label):
        """Validate keys/shapes before load_state_dict can partially mutate a module."""
        current = module.state_dict()
        missing = sorted(set(current) - set(incoming))
        unexpected = sorted(set(incoming) - set(current))
        mismatched = sorted(
            key for key in set(current) & set(incoming)
            if tuple(current[key].shape) != tuple(incoming[key].shape)
        )
        if missing or unexpected or mismatched:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            if mismatched:
                details.append(f"shape_mismatch={mismatched}")
            raise ValueError(f"incompatible {label} state dict: {'; '.join(details)}")

    def reload_checkpoint(self, ckpt_path: str):
        """Hot-swap compatible trained weights without duplicating the LAM on GPU.

        The frozen vision/latent-action model is reused, while the task-specific
        prior, action head, normalization statistics, and runtime config are
        replaced. A clear error is raised before mutation when the checkpoint
        requires a different model architecture.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = MiniLaWAMConfig(**ckpt["cfg"])
        self.assert_hot_swap_compatible(cfg)
        self._assert_state_dict_compatible(self.model.prior, ckpt["prior"], "prior")
        self._assert_state_dict_compatible(
            self.model.action_head, ckpt["action_head"], "action_head"
        )

        self.model.prior.load_state_dict(ckpt["prior"])
        self.model.action_head.load_state_dict(ckpt["action_head"])
        self.cfg = cfg
        self.model.cfg = cfg
        action_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)
        action_std = np.asarray(ckpt["action_std"], dtype=np.float32)
        self._validate_action_stats(cfg, action_mean, action_std)
        self.action_mean = action_mean
        self.action_std = action_std
        self.ckpt_path = str(ckpt_path)
        self.model.eval()
        return cfg

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
            state_xyz: Optional[np.ndarray] = None,
            return_subgoal_change: bool = False):
        """Frame -> physical action chunk [H, action_dim].

        Output semantics are checkpoint-controlled:
          abs      -> absolute [eef_pos_base XYZ, raw gripper]
          delta    -> absolute [current XYZ + predicted EEF delta, raw gripper]
          joystick -> raw [joystick XYZ, optional RZ, raw gripper]

        Joystick scaling/composition is venue-specific and intentionally happens
        in rollout_ur7e, not here. If the checkpoint was trained with use_wrist=True,
        you MUST pass `wrist_hwc_uint8` at the same timestep. If trained with
        use_state=True, pass `state_xyz` = current eef_pos [x,y,z] in meters
        (base frame); it is z-scored here with the same stats as training.

        With `return_subgoal_change=True`, return `(chunk, change_grid)`, where
        `change_grid` is the per-patch DINO feature-change magnitude
        `||u_hat_T - u_t||` for the predicted subgoal. The grid is derived from
        the same inference pass, so visualization does not run DINO a second time.
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
        model_out = self.model.predict(
            o_t, wrist=wrist, state=state, return_subgoal=return_subgoal_change
        )
        subgoal_change = None
        if return_subgoal_change:
            pred, u_t_tokens, u_hat_tokens = model_out
            change = (u_hat_tokens[0] - u_t_tokens[0]).norm(dim=-1)
            grid = int(round(np.sqrt(change.numel())))
            if grid * grid != change.numel():
                raise RuntimeError(
                    f"DINO patch-token count {change.numel()} is not a square grid"
                )
            subgoal_change = change.reshape(grid, grid).cpu().numpy().astype(np.float32)
        else:
            pred = model_out
        pred = pred[0].cpu().numpy().astype(np.float32)
        chunk = decode_action_prediction(
            pred, self.action_mean, self.action_std,
            gripper_head=getattr(self.cfg, "gripper_head", "regression"),
        )
        if getattr(self.cfg, "target_mode", "abs") == "delta":
            # Delta targets: compose absolute positions from the CURRENT eef pos.
            # Each replan re-anchors at the true arm position -> servo-like loop.
            if state_xyz is None:
                raise ValueError("checkpoint trained with target_mode='delta' -> pass "
                                 "state_xyz (current eef xyz) to compose absolute targets")
            chunk[:, :3] += np.asarray(state_xyz, np.float32)
        if return_subgoal_change:
            return chunk, subgoal_change
        return chunk

    # >>> SEAM (venue-specific): implement per setup, do NOT bake in here.
    #   - target -> command:  real UR7e: delta = eef_pos - current_TCP -> servoL;
    #                         sim:       set controller target pose / joint cmd.
    #   - gripper:            threshold the final channel at 0 (< 0 open, > 0 close).
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

    from mini_lawam.data import (
        MiniLaWAMDataset,
        _read_gripper_target,
        _read_target,
        _read_target_delta,
        _read_target_joystick,
    )

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
    target_mode = getattr(policy.cfg, "target_mode", "abs")
    include_rz = bool(getattr(policy.cfg, "include_rz", False))
    gripper_target_offset = int(
        getattr(policy.cfg, "gripper_target_offset", -1)
    )
    if gripper_target_offset < 0:
        gripper_target_offset = 0 if target_mode == "joystick" else 1
    print(f"loaded ckpt (step {step}) | action_dim={policy.cfg.action_dim} horizon={H} "
          f"use_wrist={policy.cfg.use_wrist} target={target_mode} "
          f"include_rz={include_rz} "
          f"gripper_head={getattr(policy.cfg, 'gripper_head', 'regression')} "
          f"gripper_target_offset={gripper_target_offset}")
    print(f"action_mean={np.round(policy.action_mean,4)} action_std={np.round(policy.action_std,4)}")

    def read_gt(g, t):
        """Ground truth in the same output convention as policy.act()."""
        if target_mode == "joystick":
            gt = _read_target_joystick(g, t, H, 6, include_rz=include_rz)
        elif target_mode == "delta":
            gt = _read_target_delta(g, t, H, "eef_pos_base", 6)
            gt[:, :3] += g["obs"]["eef_pos_base"][t].astype(np.float32)
        else:
            gt = _read_target(g, t + 1, H, "eef_pos_base", 6)
        grip_available = int(g["actions"].shape[0]) - (t + gripper_target_offset)
        h = min(int(gt.shape[0]), max(0, grip_available))
        gt = gt[:h]
        gt[:, -1:] = _read_gripper_target(
            g, t, h, 6, gripper_target_offset
        )
        return gt

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
            gt = read_gt(g, args.t)
        chunk = policy.act(frame, wrist, state_xyz=st)
        print(f"\ndemo={demo} t={args.t}  chunk shape={chunk.shape}")
        xyz_name = "joystick_xyz" if target_mode == "joystick" else "eef_pos"
        rz_text = f"  rz={chunk[0,3]:+.4f}" if include_rz else ""
        gt_rz_text = f"  rz={gt[0,3]:+.4f}" if include_rz else ""
        print(f"pred step0 : {xyz_name}={np.round(chunk[0,:3],4)}  "
              f"{rz_text}  gripper={chunk[0,-1]:+.3f}")
        print(f"gt   step0 : {xyz_name}={np.round(gt[0,:3],4)}  "
              f"{gt_rz_text}  gripper={gt[0,-1]:+.3f}")
        print(f"XYZ L2 err : {np.linalg.norm(chunk[0,:3]-gt[0,:3]):.4f} "
              f"{'action units' if target_mode == 'joystick' else 'm'}")
        raise SystemExit(0)

    # ---- eval mode: reproduce train.py's split, then measure error per split ----
    ds = MiniLaWAMDataset(
        args.hdf5, gap=H, horizon=H, sample_stride=args.sample_stride,
        target_mode=target_mode, include_rz=include_rz,
        gripper_target_offset=gripper_target_offset,
        include_tail_actions=bool(
            getattr(policy.cfg, "include_tail_actions", False)
        ),
    )
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
            pos_l2, step0_l2 = 0.0, 0.0
            mae = np.zeros(policy.cfg.action_dim)
            grip_ok, cnt = 0, 0
            for i in take:
                demo, t = ds.index[int(i)]
                g = data[demo]
                frame = g["obs"]["table_cam"][t]                       # (H,W,3) u8
                wrist = g["obs"]["wrist_cam"][t] if policy.cfg.use_wrist else None
                need_xyz = (getattr(policy.cfg, "use_state", False)
                            or getattr(policy.cfg, "target_mode", "abs") == "delta")
                st = (g["obs"]["eef_pos_base"][t].astype(np.float32)
                      if need_xyz else None)
                gt = read_gt(g, t)                          # [H,action_dim] target units
                pred = policy.act(frame, wrist, state_xyz=st)
                pred = pred[:len(gt)]
                pos_l2 += np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1).mean()
                step0_l2 += np.linalg.norm(pred[0, :3] - gt[0, :3])
                mae += np.abs(pred - gt).mean(axis=0)
                grip_ok += float((np.sign(pred[:, -1]) == np.sign(gt[:, -1])).mean())
                cnt += 1
            print(f"\n[{split}]  n={cnt}  (of {len(idxs)} in split)")
            if target_mode == "joystick":
                print(f"  XYZ action L2 (mean horizon)   : {pos_l2/cnt:.4f}")
                print(f"  XYZ action L2 (step 0 only)    : {step0_l2/cnt:.4f}")
                labels = "[x y z rz] grip" if include_rz else "[x y z] grip"
                print(f"  per-dim MAE {labels:<17}: {np.round(mae/cnt,4)}")
            else:
                print(f"  pos L2 err  (mean over horizon): {pos_l2/cnt*100:.2f} cm")
                print(f"  pos L2 err  (step 0 only)      : {step0_l2/cnt*100:.2f} cm")
                print(f"  per-dim MAE [x y z](m) grip    : {np.round(mae/cnt,4)}")
            print(f"  gripper sign accuracy          : {grip_ok/cnt*100:.1f}%")
