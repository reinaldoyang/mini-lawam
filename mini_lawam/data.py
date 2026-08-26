"""HDF5 pair dataloader for the minimal LaWAM BC policy.

Yields, per sample:
    frames_u8   : uint8 [2, 3, 256, 256]  = (o_t, o_{t+gap}) from table_cam, resized
    actions     : float [H, target_dim]   = normalized target chunk
    actions_mask: float [H, target_dim]    = 1 for valid steps, 0 for right-padding
    gripper_targets: float [H, 1]          = raw binary class (0=open, 1=close)

The target is selected by ``target_mode``:
    abs      : [eef_pos[t+i+1] (3), raw_gripper[t+i+1] (1)]
    delta    : [eef_pos[t+i+1] - eef_pos[t] (3), raw_gripper[t+i+1] (1)]
    joystick : [raw_actions[t+i, 0:3], raw_actions[t+i, 6]]

With ``include_rz=True``, joystick targets become
``[raw_actions[t+i, 0:3], raw_actions[t+i, 5], raw_actions[t+i, 6]]``.
With ``include_ry=True`` as well, RY is inserted before RZ:
``[raw_actions[t+i, 0:3], raw_actions[t+i, 4], raw_actions[t+i, 5], raw_actions[t+i, 6]]``.

``gripper_target_offset`` can override the gripper clock independently of XYZ.
For example, joystick XYZ can stay at ``actions[t+i, 0:3]`` while a value of 1
trains the separate binary gripper output on ``actions[t+i+1, 6]``. Rotation
column 3 (RX) is always omitted. Columns 4 (RY) and 5 (RZ) are optional for
joystick targets; both are omitted by default so legacy four-dimensional
checkpoints remain unchanged.

Targets are z-scored per dimension using dataset statistics; the checkpoint
keeps those statistics so deployment can restore the original physical scale.

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
RY_ACTION_COL = 4                 # base-Y rotation command in raw joystick `actions`
RZ_ACTION_COL = 5                 # base-Z rotation command in raw joystick `actions`
WRIST_KEY_DEFAULT = "wrist_cam"   # arm-mounted aux view (action head only, current frame t)


def build_index(hdf5_path: str, gap: int, horizon: int, sample_stride: int = 1,
                include_tail_actions: bool = False,
                ) -> List[Tuple[str, int]]:
    """List training anchors.

    Legacy mode requires a full future frame and action horizon. Tail mode keeps
    anchors through T-2, clamps the LaWM future image to the terminal frame, and
    lets the existing action mask right-pad short chunks. This is important for
    terminal discrete events such as place-and-release.
    """
    idx: List[Tuple[str, int]] = []
    with h5py.File(hdf5_path, "r") as f:
        for demo in f["data"].keys():
            T = int(f["data"][demo]["obs"]["table_cam"].shape[0])
            last = T - 2 if include_tail_actions else T - max(gap, horizon) - 1
            if last < 0:
                continue
            idx.extend((demo, t) for t in range(0, last + 1, sample_stride))
    return idx


def _read_target(g, t0: int, n: int, pos_key: str, grip_col: int) -> np.ndarray:
    """Absolute [eef_pos_base(3), action_gripper(1)] over frames [t0, t0+n)."""
    pos = g["obs"][pos_key][t0:t0 + n].astype(np.float32)          # [n,3]
    grip = g["actions"][t0:t0 + n, grip_col:grip_col + 1].astype(np.float32)  # [n,1]
    return np.concatenate([pos, grip], axis=1)                    # [n,4]


def _read_target_delta(g, t: int, n: int, pos_key: str, grip_col: int) -> np.ndarray:
    """Delta [(pos[t+i]-pos[t])(3), action_gripper(1)] for i=1..n.

    Position targets are RELATIVE to the current frame t (per-chunk anchor), so
    at deployment the chunk composes as current_TCP + delta -- servo-like,
    immune to systematic bias in absolute-position regression. Gripper stays raw.
    """
    anchor = g["obs"][pos_key][t].astype(np.float32)               # [3]
    pos = g["obs"][pos_key][t + 1:t + 1 + n].astype(np.float32)    # [n,3]
    grip = g["actions"][t + 1:t + 1 + n, grip_col:grip_col + 1].astype(np.float32)
    return np.concatenate([pos - anchor, grip], axis=1)            # [n,4]


def _read_target_joystick(g, t: int, n: int, grip_col: int,
                          include_rz: bool = False,
                          include_ry: bool = False) -> np.ndarray:
    """Raw joystick motion + gripper commands for indices [t, t+n).

    The source HDF5 action layout is [XYZ(3), rotation(3), gripper(1)]. Mini-LaWAM
    normally keeps its legacy [XYZ, gripper] target. With ``include_ry=True``
    and/or ``include_rz=True``, action columns 4 (RY) and/or 5 (RZ) are inserted
    before the gripper, in that order, yielding e.g. [XYZ, RY, RZ, gripper].
    """
    raw = g["actions"][t:t + n].astype(np.float32)
    required_col = max(
        2, grip_col,
        RY_ACTION_COL if include_ry else 0,
        RZ_ACTION_COL if include_rz else 0,
    )
    if raw.ndim != 2 or raw.shape[1] <= required_col:
        raise ValueError(
            f"expected HDF5 actions with XYZ columns 0:3 and gripper column "
            f"{grip_col}"
            + (f" and RY column {RY_ACTION_COL}" if include_ry else "")
            + (f" and RZ column {RZ_ACTION_COL}" if include_rz else "")
            + f", got shape {raw.shape}"
        )
    columns = [raw[:, :3]]
    if include_ry:
        columns.append(raw[:, RY_ACTION_COL:RY_ACTION_COL + 1])
    if include_rz:
        columns.append(raw[:, RZ_ACTION_COL:RZ_ACTION_COL + 1])
    columns.append(raw[:, grip_col:grip_col + 1])
    return np.concatenate(columns, axis=1)


def _read_gripper_target(g, t: int, n: int, grip_col: int,
                         offset: int) -> np.ndarray:
    """Raw gripper command rows [t+offset, t+offset+n), shape [n,1]."""
    start = int(t) + int(offset)
    grip = g["actions"][start:start + int(n), grip_col:grip_col + 1].astype(np.float32)
    if grip.shape != (int(n), 1):
        raise IndexError(
            f"gripper target offset {offset} from t={t} requires {n} rows, "
            f"got shape {grip.shape}"
        )
    return grip


def compute_action_stats(hdf5_path: str, pos_key: str, grip_col: int,
                         target_mode: str = "abs", horizon: int = 24,
                         include_rz: bool = False,
                         include_ry: bool = False,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dim mean/std of the targets.

    abs     : over all [eef_pos_base, action_gripper] frames.
    delta   : over all chunk deltas pos[t+i]-pos[t], i=1..horizon (positions),
              with gripper stats from the raw gripper channel.
    joystick: over raw [action_xyz, (optional action_ry), (optional action_rz),
              action_gripper] rows.
    """
    if (include_rz or include_ry) and target_mode != "joystick":
        raise ValueError(
            "include_rz/include_ry are only supported with target_mode='joystick'"
        )
    chunks = []
    with h5py.File(hdf5_path, "r") as f:
        for demo in f["data"].keys():
            g = f["data"][demo]
            T = int(
                g["actions"].shape[0]
                if target_mode == "joystick"
                else g["obs"][pos_key].shape[0]
            )
            if target_mode == "abs":
                chunks.append(_read_target(g, 0, T, pos_key, grip_col))
            elif target_mode == "delta":
                pos = g["obs"][pos_key][...].astype(np.float32)
                grip = g["actions"][:, grip_col:grip_col + 1].astype(np.float32)
                for i in range(1, horizon + 1):
                    if T <= i:
                        break
                    d = pos[i:] - pos[:-i]                       # [T-i,3]
                    chunks.append(np.concatenate([d, grip[i:]], axis=1))
            elif target_mode == "joystick":
                chunks.append(_read_target_joystick(
                    g, 0, T, grip_col, include_rz=include_rz, include_ry=include_ry
                ))
            else:
                raise ValueError(
                    f"unknown target_mode {target_mode!r}; "
                    "expected 'abs', 'delta', or 'joystick'"
                )
    alla = np.concatenate(chunks, axis=0)
    mean = alla.mean(axis=0)
    std = alla.std(axis=0)
    # Column order after XYZ(0:3): RY (if enabled) then RZ (if enabled).
    ry_col = 3 if include_ry else None
    rz_col = 3 + int(include_ry) if include_rz else None
    if ry_col is not None and std[ry_col] < 1e-6:
        raise ValueError(
            "--include-ry requested, but HDF5 action column 4 has no RY "
            "variation; use a VR dataset with nonzero RY commands"
        )
    if rz_col is not None and std[rz_col] < 1e-6:
        raise ValueError(
            "--include-rz requested, but HDF5 action column 5 has no RZ "
            "variation; use a VR dataset with nonzero RZ commands"
        )
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
        target_mode: str = "abs",    # "abs", cumulative EEF "delta", or raw "joystick"
        include_rz: bool = False,     # joystick only: append raw action RZ before grip
        include_ry: bool = False,     # joystick only: append raw action RY before RZ/grip
        gripper_target_offset: Optional[int] = None,
        include_tail_actions: bool = False,
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
        if target_mode not in ("abs", "delta", "joystick"):
            raise ValueError(
                f"unknown target_mode {target_mode!r}; "
                "expected 'abs', 'delta', or 'joystick'"
            )
        self.target_mode = target_mode
        self.include_rz = bool(include_rz)
        self.include_ry = bool(include_ry)
        if (self.include_rz or self.include_ry) and self.target_mode != "joystick":
            raise ValueError(
                "include_rz/include_ry are only supported with target_mode='joystick'"
            )
        if gripper_target_offset is None:
            # Preserve the historical contracts unless training explicitly
            # decouples gripper timing from the XYZ target representation.
            gripper_target_offset = 0 if target_mode == "joystick" else 1
        if int(gripper_target_offset) not in (0, 1):
            raise ValueError(
                "gripper_target_offset must be 0 (same row) or 1 (one row ahead)"
            )
        self.gripper_target_offset = int(gripper_target_offset)
        self.include_tail_actions = bool(include_tail_actions)
        self.resize = v2.Resize(image_hw, antialias=True)
        self.index = build_index(
            hdf5_path, self.gap, self.horizon, sample_stride,
            include_tail_actions=self.include_tail_actions,
        )
        if action_mean is None or action_std is None:
            action_mean, action_std = compute_action_stats(
                hdf5_path, pos_key, grip_col,
                target_mode=target_mode, horizon=self.horizon,
                include_rz=self.include_rz, include_ry=self.include_ry)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        expected_dim = 4 + int(self.include_rz) + int(self.include_ry)
        if self.action_mean.shape != (expected_dim,) or self.action_std.shape != (expected_dim,):
            raise ValueError(
                f"expected action stats shape ({expected_dim},), got "
                f"{self.action_mean.shape}/{self.action_std.shape}"
            )
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
        future_t = min(t + self.gap, int(cam.shape[0]) - 1)
        frames = torch.stack([self._frame(cam, t), self._frame(cam, future_t)], 0)  # [2,3,256,256]

        # EEF modes use future rows t+1..t+H; joystick uses commands t..t+H-1.
        if self.target_mode == "delta":
            raw = _read_target_delta(g, t, self.horizon, self.pos_key, self.grip_col)
        elif self.target_mode == "joystick":
            # Same-index alignment: observation[t] -> raw joystick action[t].
            raw = _read_target_joystick(
                g, t, self.horizon, self.grip_col,
                include_rz=self.include_rz, include_ry=self.include_ry
            )
        else:
            raw = _read_target(g, t + 1, self.horizon, self.pos_key, self.grip_col)  # [h,4]
        # The chosen gripper offset can have fewer valid tail rows than XYZ.
        grip_available = int(g["actions"].shape[0]) - (t + self.gripper_target_offset)
        h = min(int(raw.shape[0]), max(0, grip_available))
        raw = raw[:h]
        # Gripper timing is an explicit, target-mode-independent contract.
        # In the recommended joystick-binary run, XYZ stays at action[t+i] while
        # gripper uses action[t+i+1], teaching chunk[0] to predict the next command.
        raw[:, 3:4] = _read_gripper_target(
            g, t, h, self.grip_col, self.gripper_target_offset
        )
        # Preserve the raw discrete class before z-scoring the legacy 4D action
        # target. Binary-head training consumes this field directly; regression
        # checkpoints continue to use normalized actions[..., 3] unchanged.
        raw_gripper = raw[:, 3].copy()
        normalized = (raw - self.action_mean) / self.action_std
        dim = self.action_mean.shape[0]
        actions = np.zeros((self.horizon, dim), dtype=np.float32)
        mask = np.zeros((self.horizon, dim), dtype=np.float32)
        gripper_targets = np.zeros((self.horizon, 1), dtype=np.float32)
        actions[:h] = normalized
        mask[:h] = 1.0
        gripper_targets[:h, 0] = (raw_gripper > 0.0).astype(np.float32)

        out = {
            "frames_u8": frames,
            "actions": torch.from_numpy(actions),
            "actions_mask": torch.from_numpy(mask),
            "gripper_targets": torch.from_numpy(gripper_targets),
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
