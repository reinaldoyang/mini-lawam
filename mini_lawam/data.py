"""HDF5 pair dataloader for the minimal LaWAM BC policy.

Yields, per sample:
    frames_u8   : uint8 [2, 3, 256, 256]  = (o_t, o_{t+gap}) from table_cam, resized
    actions     : float [H, target_dim]   = normalized target chunk
    actions_mask: float [H, target_dim]    = 1 for valid steps, 0 for right-padding

Target = ABSOLUTE end-effector trajectory over the next H steps:
    [eef_pos (3), gripper_pos (1)]  from obs, at frames t+1 .. t+H.
This is cleaner/less quantized than the raw delta `actions` field. Targets are
z-scored per dim using dataset statistics; keep the stats to un-normalize at
deployment. At rollout you convert predicted absolute positions to whatever the
IsaacLab controller consumes (e.g. delta = pred - current).

Preprocessing contract: this Dataset only resizes to 256 and returns uint8. The
train loop applies ImageNet normalization on GPU via
gpu_two_view_video_aug(..., training=False), matching what the LAM expects.
"""

from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

POS_KEY_DEFAULT = "eef_pos_base"  # absolute EEF position (3), base frame
GRIP_ACTION_COL = 6               # gripper command column in raw `actions` (no gripper obs here)
WRIST_KEY_DEFAULT = "wrist_cam"   # arm-mounted aux view (action head only, current frame t)


def build_index(hdf5_path: str, gap: int, horizon: int, sample_stride: int = 1
                ) -> List[Tuple[str, int]]:
    """List (demo_key, t) where o_{t+gap} and targets t+1..t+H are all valid."""
    idx: List[Tuple[str, int]] = []
    with h5py.File(hdf5_path, "r") as f:
        for demo in f["data"].keys():
            T = int(f["data"][demo]["obs"]["table_cam"].shape[0])
            last = T - max(gap, horizon) - 1
            if last < 0:
                continue
            idx.extend((demo, t) for t in range(0, last + 1, sample_stride))
    return idx


def _read_target(g, t0: int, n: int, pos_key: str, grip_col: int) -> np.ndarray:
    """Absolute [eef_pos_base(3), action_gripper(1)] over frames [t0, t0+n)."""
    pos = g["obs"][pos_key][t0:t0 + n].astype(np.float32)          # [n,3]
    grip = g["actions"][t0:t0 + n, grip_col:grip_col + 1].astype(np.float32)  # [n,1]
    return np.concatenate([pos, grip], axis=1)                    # [n,4]


def compute_action_stats(hdf5_path: str, pos_key: str, grip_col: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dim mean/std over all [eef_pos_base, action_gripper] targets."""
    chunks = []
    with h5py.File(hdf5_path, "r") as f:
        for demo in f["data"].keys():
            g = f["data"][demo]
            T = int(g["obs"][pos_key].shape[0])
            chunks.append(_read_target(g, 0, T, pos_key, grip_col))
    alla = np.concatenate(chunks, axis=0)
    mean = alla.mean(axis=0)
    std = alla.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


class MiniLaWAMDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        gap: int = 32,
        horizon: int = 32,
        pos_key: str = POS_KEY_DEFAULT,
        grip_col: int = GRIP_ACTION_COL,
        sample_stride: int = 1,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        image_hw: Tuple[int, int] = (256, 256),
        use_wrist: bool = False,
        wrist_key: str = WRIST_KEY_DEFAULT,
        use_state: bool = False,
    ):
        self.hdf5_path = hdf5_path
        self.gap = int(gap)
        self.horizon = int(horizon)
        self.pos_key = pos_key
        self.grip_col = grip_col
        self.image_hw = image_hw
        self.use_wrist = use_wrist
        self.wrist_key = wrist_key
        self.use_state = use_state   # proprioception = current eef_pos at frame t
        self.resize = v2.Resize(image_hw, antialias=True)
        self.index = build_index(hdf5_path, self.gap, self.horizon, sample_stride)
        if action_mean is None or action_std is None:
            action_mean, action_std = compute_action_stats(hdf5_path, pos_key, grip_col)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        self._file: Optional[h5py.File] = None  # opened lazily per worker

    def __len__(self) -> int:
        return len(self.index)

    def _f(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def _frame(self, cam, t: int) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(cam[t])).permute(2, 0, 1)  # [3,H,W] u8
        return self.resize(x).to(torch.uint8)                                # [3,256,256]

    def __getitem__(self, i: int):
        demo, t = self.index[i]
        g = self._f()["data"][demo]
        cam = g["obs"]["table_cam"]
        frames = torch.stack([self._frame(cam, t), self._frame(cam, t + self.gap)], 0)  # [2,3,256,256]

        # Absolute EEF target trajectory over the next H steps (t+1 .. t+H).
        raw = _read_target(g, t + 1, self.horizon, self.pos_key, self.grip_col)  # [h,4]
        h = raw.shape[0]
        raw = (raw - self.action_mean) / self.action_std
        dim = self.action_mean.shape[0]
        actions = np.zeros((self.horizon, dim), dtype=np.float32)
        mask = np.zeros((self.horizon, dim), dtype=np.float32)
        actions[:h] = raw
        mask[:h] = 1.0

        out = {
            "frames_u8": frames,
            "actions": torch.from_numpy(actions),
            "actions_mask": torch.from_numpy(mask),
        }
        if self.use_wrist:
            # Aux view: wrist_cam at the CURRENT frame t only (not the pair).
            out["wrist_u8"] = self._frame(g["obs"][self.wrist_key], t)  # [3,256,256]
        if self.use_state:
            # Proprioception: current eef_pos at frame t, z-scored with the position
            # part of the action stats (same units/frame as the target positions).
            cur = g["obs"][self.pos_key][t].astype(np.float32)          # [3]
            state = (cur - self.action_mean[:3]) / self.action_std[:3]
            out["state"] = torch.from_numpy(state.astype(np.float32))   # [3]
        return out


def split_o_t_o_T(vids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """[B,2,3,256,256] normalized -> (o_t, o_T) each [B,1,3,256,256]."""
    return vids[:, 0:1], vids[:, 1:2]
