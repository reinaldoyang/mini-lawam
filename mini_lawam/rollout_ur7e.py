#!/usr/bin/env python3
"""Real-world UR7e rollout for the mini_lawam BC policy (the venue SEAM).

Pattern follows ur7e_ramen_il/scripts/real_world/lapa_latent_real_servoL.py:
  - background 500 Hz servo thread (real_servo_utils.start_servo_pipeline)
    interpolates toward a shared target TCP; we only update the target.
  - background RealSense reader caches the latest table_cam frame.
  - pygame control: S=start rollout, E=end rollout, H=home, Q=quit.
  - DRY-RUN by default; add --execute to actually send servoL/gripper.

mini_lawam-specific differences vs the LAPA baseline:
  - policy output is an [H, action_dim] chunk whose checkpoint-stored target mode is:
      abs/delta -> absolute [eef_pos_base(3), gripper(1)] targets
      joystick  -> raw recorded [joystick_xyz(3), optional rz(1), gripper(1)] commands
    Joystick motion commands are multiplied by --action-scale and composed with
    the measured TCP at each execution tick. Roll/pitch stay locked; optional RZ
    is applied only for a compatible checkpoint when --enable-rz is passed.
  - receding horizon: execute the first --exec-steps (default 8) waypoints of
    each chunk at --control-hz (default 20, matching training), then re-plan
    from a fresh frame.
  - only table_cam feeds the policy (wrist_cam unused by design).

Safety for absolute targets:
  - workspace box clamp (--ws-min/--ws-max), per-axis.
  - per-replan reach clamp: move at most --max-reach toward the prediction.
  - the servo thread's interp_alpha/max_pos_step caps actual robot speed.

Run (from the LaWAM repo root, so relative LAM paths resolve):
  # dry-run, print chunks only:
  python -m mini_lawam.rollout_ur7e --table-cam-serial <SN>
  # execute on robot:
  python -m mini_lawam.rollout_ur7e --table-cam-serial <SN> \
      --robot-ip 140.96.93.130 --execute --use-gripper-control
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_REALWORLD_DIR = "/home/iclu200/reinaldoyang/ur7e_ramen_il/scripts/real_world"

TABLE_CAM_KEY = "table_cam"
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30
ROBOTIQ_SOCKET_PORT = 63352

# Home joint pose the robot returns to before each rollout. This MUST match the
# training dataset's joint_pos[0], or the first observation is out-of-distribution.
#   multi_egg_30_moved_256 : [0.4076, -1.4255, -1.7052, -1.5821, 1.5703, 1.9768]
#   old multi_egg_114ep    : [0, -pi/2, -pi/2, -pi/2, pi/2, pi/2]
# Override at runtime with --home-q. Default below is the moved-256 dataset.
HOME_Q = [0.4076, -1.4255, -1.7052, -1.5821, 1.5703, 1.9768]

# Fixed locked TCP orientation (axis-angle rotvec, base frame): measured from the
# dataset's eef_quat_base across ALL demo frames (constant within 0.24 deg).
# = 180 deg about base Y, i.e. tool pointing straight down.
DEMO_LOCKED_ROTVEC = np.array([0.0036, 3.14094, -0.00024], dtype=np.float64)
TRACE_TRIAL_RE = re.compile(r"^trial_(\d+)(?:_|$)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="results/mini_lawam/ckpt_114ep_wrist.pt")
    p.add_argument(
        "--task-profile", action="append", nargs=3,
        metavar=("KEY", "CKPT", "CLOSE_MM"),
        help="Bind a numeric key (1-9) to a checkpoint and gripper closing width. "
             "Repeat for multiple tasks; press the key while IDLE to switch. "
             "When omitted, --ckpt and --gripper-close-mm retain their usual behavior.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--realworld-dir", default=DEFAULT_REALWORLD_DIR,
                   help="ur7e_ramen_il/scripts/real_world (real_servo_utils + rtde_gripper)")

    # cadence / horizon
    p.add_argument("--control-hz", type=float, default=20.0,
                   help="waypoint rate; MUST match training rate (20 Hz)")
    p.add_argument("--exec-steps", type=int, default=8,
                   help="chunk steps executed before re-planning (receding horizon). "
                        "Ignored when --temporal-ensemble is set (re-plans every step).")
    p.add_argument("--temporal-ensemble", action="store_true",
                   help="ACT-style temporal ensembling: re-plan EVERY step and execute a "
                        "weighted average of all overlapping chunk predictions for the "
                        "current timestep. Cancels the per-chunk oscillation (smooth motion) "
                        "and keeps the arm on the predicted path (in-distribution feedback). "
                        "Needs inference << control period (15ms vs 50ms here -> fine).")
    p.add_argument("--te-m", type=float, default=0.1,
                   help="Temporal-ensemble weight decay: newest prediction weight 1, a "
                        "prediction made 'age' steps ago gets exp(-te_m*age). Larger = more "
                        "responsive/less smooth; smaller = smoother/laggier. ACT uses ~0.01.")
    p.add_argument("--delta-scale", type=float, default=1.0,
                   help="Deployment gain for XYZ deltas from a delta-target checkpoint. "
                        "1.0 preserves the learned motion; values >1 command larger "
                        "translations. Applied before temporal ensembling and safety clamps. "
                        "Does not affect the gripper; non-default values require a "
                        "delta-target checkpoint.")
    p.add_argument("--action-scale", type=float, default=0.3,
                   help="Deployment gain for raw XYZ and optional RZ commands from a "
                        "joystick-target checkpoint (default: 0.3). Applied after de-normalization and "
                        "before temporal ensembling/TCP composition. Does not affect "
                        "the gripper and is ignored by abs/delta checkpoints.")
    p.add_argument("--max-steps", type=int, default=2000, help="max control steps per rollout")
    p.add_argument("--num-rollouts", type=int, default=10)
    p.add_argument("--startup-wait-sec", type=float, default=1.0)

    # robot
    p.add_argument("--execute", action="store_true", help="actually send servoL/gripper")
    p.add_argument("--robot-ip", default="140.96.93.130")
    p.add_argument("--servo-hz", type=float, default=500.0)
    p.add_argument("--servol-speed", type=float, default=0.25)
    p.add_argument("--servol-acc", type=float, default=0.25)
    p.add_argument("--servol-lookahead", type=float, default=0.08)
    p.add_argument("--servol-gain", type=float, default=300.0)
    p.add_argument("--servol-interp-alpha", type=float, default=0.25)
    p.add_argument("--servol-max-pos-step", type=float, default=0.0015)
    p.add_argument("--servol-max-rot-step", type=float, default=0.02)
    p.add_argument("--home-movej-speed", type=float, default=0.6)
    p.add_argument("--home-movej-acc", type=float, default=1.2)
    p.add_argument("--home-q", type=float, nargs=6, default=HOME_Q,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                   help="Home joint pose (rad) to return to before each rollout. "
                        "MUST match the training dataset's joint_pos[0]. Default = "
                        "moved-256 dataset; pass the old home for old checkpoints.")

    # safety (absolute targets)
    # Defaults = multi_egg demo eef_pos_base range (+3cm margin, HARD z-floor at
    # the lowest z any demo ever reached, 0.158 m). Retighten if your table differs.
    p.add_argument("--ws-min", type=float, nargs=3, default=[-0.072, -0.164, 0.158],
                   metavar=("X", "Y", "Z"), help="workspace box min (base frame, m)")
    p.add_argument("--ws-max", type=float, nargs=3, default=[0.523, 0.612, 0.518],
                   metavar=("X", "Y", "Z"), help="workspace box max (base frame, m)")
    p.add_argument("--max-reach", type=float, default=0.06,
                   help="max distance (m) toward the predicted target per re-plan")
    p.add_argument("--locked-rotvec", type=float, nargs=3, default=None,
                   metavar=("RX", "RY", "RZ"),
                   help="override the fixed TCP orientation (axis-angle rotvec). "
                        "Default: DEMO_LOCKED_ROTVEC, measured from the demos.")
    p.add_argument(
        "--enable-rz", action="store_true",
        help="Apply the policy's predicted joystick RZ deltas around base Z while "
             "keeping roll/pitch locked. Requires a checkpoint trained with "
             "--target joystick --include-rz. Default: disabled (fully locked).",
    )

    # smoothing (kills prediction jitter between re-plans)
    p.add_argument("--target-ema", type=float, default=1.0,
                   help="EMA weight on the NEW prediction in [0,1]; 1.0 = off, "
                        "0.3 = heavy smoothing across waypoints")
    p.add_argument("--target-deadband", type=float, default=0.0,
                   help="ignore target changes smaller than this (m); 0 = off. "
                        "e.g. 0.004 suppresses ~4mm prediction jitter")

    # gripper
    p.add_argument("--use-gripper-control", action="store_true")
    p.add_argument("--gripper-open-mm", type=float, default=52.0)
    # Partial close (matches lapa_latent_real_servoL.py). 0.0 = fully shut =
    # crushes the object / collides fingertips. Raise for a bigger object.
    p.add_argument("--gripper-close-mm", type=float, default=23.0)
    p.add_argument("--gripper-speed", type=int, default=100)
    p.add_argument("--gripper-force", type=int, default=50)
    p.add_argument("--gripper-threshold", type=float, default=0.0,
                   help="final action channel <= thr => open, > thr => close "
                        "(open=-1, close=+1)")
    p.add_argument("--gripper-open-lead-steps", type=int, default=1,
                   help="When the commanded gripper is already closed, allow an open "
                        "prediction this many chunk steps ahead to trigger release now. "
                        "The release is latched open for the rest of the rollout. "
                        "Default 1 = 50 ms lookahead at 20 Hz; 0 disables lookahead.")

    # camera
    p.add_argument("--train-frame-hw", type=int, nargs=2, default=[168, 224],
                   metavar=("H", "W"),
                   help="Recorded size of the TRAINING images (record_real.py stores "
                        "224x168). Live camera frames are first resized to this, then "
                        "to 256x256 -- the exact training pipeline. Pass 0 0 to disable.")
    p.add_argument("--table-cam-serial", default="", help="RealSense serial for table_cam")
    p.add_argument("--wrist-cam-serial", default="",
                   help="RealSense serial for wrist_cam (REQUIRED if the checkpoint "
                        "was trained with use_wrist=True)")
    # per-camera exposure/gain (0.1 ms units, matching record_real.py). None = auto.
    p.add_argument("--table-exposure", type=float, default=None,
                   help="table_cam manual exposure in 0.1ms units (omit = auto).")
    p.add_argument("--table-gain", type=float, default=None,
                   help="table_cam manual gain (only applies with --table-exposure).")
    p.add_argument("--wrist-exposure", type=float, default=None,
                   help="wrist_cam manual exposure in 0.1ms units (omit = auto).")
    p.add_argument("--wrist-gain", type=float, default=None,
                   help="wrist_cam manual gain (only applies with --wrist-exposure).")
    p.add_argument("--show-camera", action="store_true",
                   help="live window: the 256x256 model-input view(s)")
    p.add_argument("--show-subgoal", action="store_true",
                   help="add a live table-camera overlay of predicted DINO feature change "
                        "||u_hat_T-u_t|| (requires --show-camera). Red/yellow patches are "
                        "where the predicted subgoal differs most from the observation.")
    p.add_argument("--subgoal-alpha", type=float, default=0.55,
                   help="opacity of the live subgoal heatmap in [0,1] (default: 0.55)")
    p.add_argument("--subgoal-update-steps", type=int, default=8,
                   help="refresh the displayed subgoal heatmap every N policy inferences "
                        "and hold it between updates (default: 8, about 2.5 Hz when the "
                        "policy runs at 20 Hz). Does not change policy inference/control.")
    p.add_argument("--video-scale", type=float, default=1.0, help="display window scale")
    p.add_argument("--offline-image", default=None,
                   help="HDF5 path or image file: dry-run inference without a camera. "
                        "For HDF5, uses data/demo_0/obs/table_cam[0].")

    # UI / debug
    p.add_argument("--pygame-window-w", type=int, default=560)
    p.add_argument("--pygame-window-h", type=int, default=150)
    p.add_argument("--trace-dir", default=None,
                   help="save per-rollout summary JSON and first table-camera frame here. "
                        "Trial numbering resumes from the highest trial_<N> already present.")
    p.add_argument("--save-frames", type=int, default=0,
                   help="Save the live table/wrist frames fed to the policy every N "
                        "control steps into <trace-dir>/frames_trial_<N>_<timestamp>/ "
                        "(0 = off). "
                        "Lets failures be analyzed offline on the exact inputs.")
    return p


def next_trace_trial_number(trace_dir) -> int:
    """Return max existing ``trial_<N>_...`` number + 1.

    A rollout normally writes both a PNG and JSON, so counting files would skip
    numbers. Parsing and deduplicating trial IDs also avoids overwriting a
    partial trace left by an interrupted run. With an empty/missing trace
    directory, persisted numbering starts at 1. Without tracing, retain the
    historical in-session zero-based number.
    """
    if not trace_dir:
        return 0
    root = Path(trace_dir)
    if not root.exists():
        return 1
    numbers = []
    for child in root.iterdir():
        match = TRACE_TRIAL_RE.match(child.name)
        if match is not None:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def format_rollout_stem(trial: int, started_wall: float) -> str:
    """Requested trace basename: trial_10_20260724T133010."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(started_wall))
    return f"trial_{int(trial)}_{stamp}"


# ----------------------------------------------------------------------------
# Camera: single-cam background reader (slim version of LAPA's pair reader)
# ----------------------------------------------------------------------------
class RealSenseTableReader:
    def __init__(self, serial: str, name: str = "table_cam",
                 exposure=None, gain=None):
        import pyrealsense2 as rs
        self.rs = rs
        self.serial = str(serial)
        self.name = name
        # exposure in 0.1 ms units (same convention as record_real.py). None = auto.
        self.exposure = exposure
        self.gain = gain
        self.lock = threading.Lock()
        self.last_frame = None       # RGB uint8 [H,W,3]
        self.last_time = None
        self.timeouts = 0
        self.running = False
        self.pipeline = None
        self.thread = None

    def _apply_color_options(self, profile):
        """Set manual exposure/gain (or auto) on the color sensor, like record_real.py."""
        rs = self.rs
        color_sensor = None
        for sensor in profile.get_device().query_sensors():
            nm = sensor.get_info(rs.camera_info.name)
            if "RGB" in nm or "Color" in nm:
                color_sensor = sensor
                break
        if color_sensor is None:
            print(f"[RS] {self.name} {self.serial}: no color sensor, skip exposure setup")
            return
        if self.exposure is None:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            print(f"[RS] {self.name} {self.serial}: AUTO exposure")
        else:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, float(self.exposure))
            msg = f"[RS] {self.name} {self.serial}: MANUAL exposure={self.exposure}"
            if self.gain is not None:
                color_sensor.set_option(rs.option.gain, float(self.gain))
                msg += f" gain={self.gain}"
            print(msg)
        time.sleep(0.2)   # let the setting take effect before warmup frames

    def _start_pipeline(self, hardware_reset=False):
        rs = self.rs
        if hardware_reset:
            ctx = rs.context()
            for dev in ctx.query_devices():
                try:
                    if dev.get_info(rs.camera_info.serial_number) == self.serial:
                        print(f"[RS] hardware_reset serial={self.serial}")
                        dev.hardware_reset()
                        time.sleep(3.0)
                except Exception:
                    pass
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        # rgb8 to match record_real.py (the training-data source).
        config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.rgb8, CAM_FPS)
        profile = pipeline.start(config)
        self.pipeline = pipeline
        self._apply_color_options(profile)
        for _ in range(10):  # warmup
            try:
                fs = pipeline.wait_for_frames(timeout_ms=1000)
                cf = fs.get_color_frame()
                if cf:
                    with self.lock:
                        self.last_frame = np.asanyarray(cf.get_data()).copy()
                        self.last_time = time.time()
            except Exception:
                pass
        print(f"[RS] started {self.name} serial={self.serial} {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")

    def _loop(self):
        while self.running:
            try:
                fs = self.pipeline.wait_for_frames(timeout_ms=1000)
                cf = fs.get_color_frame()
                if not cf:
                    self.timeouts += 1
                    continue
                with self.lock:
                    self.last_frame = np.asanyarray(cf.get_data()).copy()
                    self.last_time = time.time()
                self.timeouts = 0
            except Exception as exc:
                self.timeouts += 1
                print(f"[RS] read failed ({self.timeouts}): {exc}")
                if self.timeouts >= 5:
                    try:
                        self.pipeline.stop()
                    except Exception:
                        pass
                    time.sleep(0.8)
                    try:
                        self._start_pipeline(hardware_reset=True)
                    except Exception as e2:
                        print(f"[RS] restart failed: {e2}")
                        time.sleep(1.0)
                    self.timeouts = 0

    def latest(self):
        """Non-blocking: newest cached RGB frame or None (for the display)."""
        with self.lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def start(self):
        if not self.serial:
            raise ValueError("--table-cam-serial is required with a camera")
        self.running = True
        self._start_pipeline()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def read_rgb(self) -> np.ndarray:
        for _ in range(20):
            with self.lock:
                if self.last_frame is not None:
                    return self.last_frame.copy()
            time.sleep(0.05)
        raise RuntimeError("no cached table_cam frame")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Gripper (latched, same encoding as training: open=-1, close=+1)
# ----------------------------------------------------------------------------
class LatchedGripper:
    def __init__(self, RobotiqGripper, robot_ip, open_mm, close_mm, speed, force):
        def pct_to_raw(v):
            v = float(v)
            return int(np.clip(round(v * 255.0 / 100.0 if v <= 100.0 else v), 0, 255))

        self._g = RobotiqGripper()
        self._g.connect(str(robot_ip), ROBOTIQ_SOCKET_PORT)
        self.open_mm = float(open_mm)
        self.close_mm = float(close_mm)
        self.open_raw = self._mm_to_raw(self.open_mm)
        self.close_raw = self._mm_to_raw(self.close_mm)
        self.speed_raw = pct_to_raw(speed)
        self.force_raw = pct_to_raw(force)
        self.state = None
        if hasattr(self._g, "activate"):
            print("[GRIPPER] activating")
            self._g.activate(auto_calibrate=False)

    def _mm_to_raw(self, width_mm):
        open_ref = max(self.open_mm, 1e-6)
        return int(np.clip(round(255.0 * (1.0 - float(width_mm) / open_ref)), 0, 255))

    def set_close_mm(self, close_mm):
        """Change the width used by the next close command without moving now."""
        close_mm = float(close_mm)
        if not np.isfinite(close_mm) or not 0.0 <= close_mm <= self.open_mm:
            raise ValueError(
                f"gripper close width must be in [0, {self.open_mm:g}] mm, got {close_mm}"
            )
        self.close_mm = close_mm
        self.close_raw = self._mm_to_raw(close_mm)
        # If the prior task ended closed, ensure its cached command cannot suppress
        # the first close command at the newly selected width.
        if self.state == "close":
            self.state = None
        print(f"[GRIPPER] configured close width={self.close_mm:g} mm "
              f"(raw={self.close_raw}); no movement until the next command")

    def command(self, cmd: str):
        if cmd == self.state:
            return
        raw = self.open_raw if cmd == "open" else self.close_raw
        self._g.move(raw, self.speed_raw, self.force_raw)
        self.state = cmd

    def stop(self):
        try:
            self._g.disconnect()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# pygame status window (S/E/H/Q), same operator flow as LAPA/record_real
# ----------------------------------------------------------------------------
def init_pygame(args, two_rows=False):
    import pygame
    pygame.init()
    pygame.display.set_caption("mini_lawam UR7e rollout")
    if args.show_camera:
        vs = float(args.video_scale)
        mv = int(256 * vs)
        n = 1 + int(bool(two_rows)) + int(bool(args.show_subgoal))
        w = 16 + n * (mv + 10) + 6
        h = 96 + mv + 30
        screen = pygame.display.set_mode((max(w, args.pygame_window_w), h))
    else:
        screen = pygame.display.set_mode((args.pygame_window_w, args.pygame_window_h))
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()
    return pygame, screen, font, clock


def draw_status(pygame, screen, font, mode, extra="", frame_rgb=None,
                wrist_rgb=None, subgoal_rgb=None, video_scale=1.0,
                task_keys=""):
    """Status text + policy inputs and optional predicted-subgoal overlay.

    `subgoal_rgb` is the table input overlaid with per-patch
    `||u_hat_T-u_t||`; it is already 256x256 RGB.
    """
    screen.fill((20, 20, 20))
    controls = "S=start  E=end  H=home  Q=quit"
    if task_keys:
        controls += f"  {task_keys}=task"
    for i, line in enumerate([f"Mode: {mode}", controls, extra]):
        if line:
            screen.blit(font.render(line, True, (0, 255, 0)), (16, 18 + 26 * i))
    vs = float(video_scale)
    mv = int(256 * vs)
    small = pygame.font.SysFont(None, 22)
    x, y0 = 16, 96
    panels = [
        (frame_rgb, "table_cam input"),
        (wrist_rgb, "wrist_cam input"),
        (subgoal_rgb, "pred subgoal change (red=high)"),
    ]
    for rgb, name in panels:
        if rgb is None:
            continue
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))  # (W,H,3)
        model = pygame.transform.smoothscale(surf, (mv, mv))  # what the policy ingests
        screen.blit(model, (x, y0))
        screen.blit(small.render(name, True, (0, 255, 0)),
                    (x, y0 + mv + 6))
        x += mv + 10
    pygame.display.flip()


def poll_cmd(pygame):
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            return "quit"
        if ev.type == pygame.KEYDOWN:
            cmd = {pygame.K_s: "start", pygame.K_e: "end",
                   pygame.K_h: "home", pygame.K_q: "quit"}.get(ev.key)
            if cmd is not None:
                return cmd
            if getattr(ev, "unicode", "") in "123456789":
                return f"task:{ev.unicode}"
    return None


def build_task_profiles(raw_profiles, default_ckpt, default_close_mm, open_mm):
    """Parse repeatable CLI task profiles while preserving single-task behavior."""
    open_mm = float(open_mm)
    if not np.isfinite(open_mm) or open_mm <= 0.0:
        raise ValueError(f"--gripper-open-mm must be finite and > 0, got {open_mm}")

    entries = raw_profiles or [("1", default_ckpt, str(default_close_mm))]
    profiles = {}
    for raw_key, ckpt, raw_close_mm in entries:
        key = str(raw_key)
        if len(key) != 1 or key not in "123456789":
            raise ValueError(f"task profile KEY must be one digit from 1 to 9, got {key!r}")
        if key in profiles:
            raise ValueError(f"duplicate task profile key {key!r}")
        close_mm = float(raw_close_mm)
        if not np.isfinite(close_mm) or not 0.0 <= close_mm <= open_mm:
            raise ValueError(
                f"task {key} CLOSE_MM must be in [0, {open_mm:g}], got {close_mm}"
            )
        profiles[key] = {
            "key": key,
            "ckpt": str(ckpt),
            "close_mm": close_mm,
        }
    return profiles, bool(raw_profiles)


# ----------------------------------------------------------------------------
# Safety helpers for ABSOLUTE position targets
# ----------------------------------------------------------------------------
def clamp_abs_target(pred_xyz, current_xyz, ws_min, ws_max, max_reach):
    """Workspace-box clamp + limit distance traveled toward the prediction."""
    tgt = np.clip(np.asarray(pred_xyz, dtype=np.float64),
                  np.asarray(ws_min, dtype=np.float64),
                  np.asarray(ws_max, dtype=np.float64))
    delta = tgt - np.asarray(current_xyz, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist > float(max_reach) and dist > 1e-9:
        tgt = np.asarray(current_xyz, dtype=np.float64) + delta * (float(max_reach) / dist)
    return tgt


def scale_delta_chunk(chunk, anchor_xyz, delta_scale):
    """Scale a composed absolute chunk's XYZ displacement around its TCP anchor."""
    scaled = np.asarray(chunk).copy()
    anchor = np.asarray(anchor_xyz, dtype=scaled.dtype)
    scaled[:, :3] = anchor + float(delta_scale) * (scaled[:, :3] - anchor)
    return scaled


def scale_joystick_chunk(chunk, action_scale, include_rz=False):
    """Scale raw joystick motion (XYZ[+RZ]); preserve the final gripper channel."""
    scaled = np.asarray(chunk).copy()
    expected_dim = 5 if include_rz else 4
    if scaled.ndim != 2 or scaled.shape[1] != expected_dim:
        raise ValueError(
            f"joystick chunk must have shape [H,{expected_dim}], got {scaled.shape}"
        )
    scaled[:, :-1] *= float(action_scale)
    return scaled


def compose_target_xyz(pred_xyz, current_xyz, target_mode):
    """Convert a policy XYZ row into an absolute TCP target.

    Joystick rows are incremental commands (already deployment-scaled);
    abs/delta policy rows are already absolute by this boundary.
    """
    pred = np.asarray(pred_xyz, dtype=np.float64)
    if target_mode == "joystick":
        return np.asarray(current_xyz, dtype=np.float64) + pred
    if target_mode in ("abs", "delta"):
        return pred
    raise ValueError(f"unsupported target_mode={target_mode!r}")


def compose_locked_rotvec_with_rz(locked_rotvec, rz_offset):
    """Apply an accumulated base-Z rotation to a locked tool-down orientation."""
    from scipy.spatial.transform import Rotation

    locked = np.asarray(locked_rotvec, dtype=np.float64).reshape(3)
    rz_rotation = Rotation.from_rotvec(
        np.asarray([0.0, 0.0, float(rz_offset)], dtype=np.float64)
    )
    return (rz_rotation * Rotation.from_rotvec(locked)).as_rotvec()


def select_gripper_with_open_lookahead(
    chunk,
    step_index,
    last_cmd,
    release_latched,
    threshold=0.0,
    open_lead_steps=1,
):
    """Select a gripper row while allowing only the release event to look ahead.

    Closing always uses the immediate row. Once the commanded gripper is closed,
    an open prediction in [step_index, step_index + open_lead_steps] triggers
    release. Release then stays latched open for this single-pick rollout so an
    unchanged pre-release image cannot command close again on the next frame.

    Returns ``(grip_value, release_latched, source_index)``. ``source_index`` is
    -1 when an existing release latch supplied the open command.
    """
    rows = np.asarray(chunk)
    if rows.ndim != 2 or rows.shape[1] < 4:
        raise ValueError(f"action chunk must have shape [H,>=4], got {rows.shape}")
    idx = int(step_index)
    if idx < 0 or idx >= rows.shape[0]:
        raise IndexError(f"gripper step_index {idx} outside chunk horizon {rows.shape[0]}")
    lead = int(open_lead_steps)
    if lead < 0:
        raise ValueError(f"open_lead_steps must be >= 0, got {lead}")

    if release_latched:
        return -1.0, True, -1

    immediate = float(rows[idx, -1])
    if last_cmd != "close":
        return immediate, False, idx

    end = min(rows.shape[0] - 1, idx + lead)
    window = rows[idx:end + 1, -1]
    open_offsets = np.flatnonzero(window <= float(threshold))
    if open_offsets.size:
        source_index = idx + int(open_offsets[0])
        return float(rows[source_index, -1]), True, source_index
    return immediate, False, idx


def make_subgoal_overlay(model_input_rgb, change_grid, alpha=0.55):
    """Colorize a DINO patch-change grid and blend it over the 256px model input."""
    import cv2

    base = np.asarray(model_input_rgb, dtype=np.uint8)
    heat = np.nan_to_num(np.asarray(change_grid, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)
    if heat.ndim != 2:
        raise ValueError(f"subgoal change map must be 2D, got {heat.shape}")
    lo, hi = np.percentile(heat, [5.0, 95.0])
    if hi <= lo + 1e-8:
        normalized = np.zeros_like(heat)
    else:
        normalized = np.clip((heat - lo) / (hi - lo), 0.0, 1.0)
    resized = cv2.resize(normalized, (base.shape[1], base.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    heat_bgr = cv2.applyColorMap(
        np.clip(resized * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    heat_rgb = heat_bgr[:, :, ::-1].copy()
    return cv2.addWeighted(base, 1.0 - float(alpha), heat_rgb, float(alpha), 0.0)


def should_update_subgoal(enabled, inference_index, update_steps):
    """Whether this inference should refresh the held GUI subgoal overlay."""
    return bool(enabled) and int(inference_index) % int(update_steps) == 0


def format_duration_hms(duration_sec: float) -> str:
    """Format seconds as HH:MM:SS.mmm, matching the rollout summary schema."""
    total_ms = int(round(float(duration_sec) * 1000.0))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def load_offline_frame(path: str, cam: str = "table_cam") -> np.ndarray:
    p = Path(path)
    if p.suffix in (".hdf5", ".h5"):
        import h5py
        with h5py.File(p, "r") as f:
            demo = list(f["data"].keys())[0]
            return np.asarray(f["data"][demo]["obs"][cam][0])
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    if not np.isfinite(args.delta_scale) or args.delta_scale < 0.0:
        raise ValueError("--delta-scale must be a finite value >= 0")
    if not np.isfinite(args.action_scale) or args.action_scale < 0.0:
        raise ValueError("--action-scale must be a finite value >= 0")
    if args.gripper_open_lead_steps < 0:
        raise ValueError("--gripper-open-lead-steps must be >= 0")
    if args.show_subgoal and not args.show_camera:
        raise ValueError("--show-subgoal requires --show-camera")
    if not np.isfinite(args.subgoal_alpha) or not 0.0 <= args.subgoal_alpha <= 1.0:
        raise ValueError("--subgoal-alpha must be a finite value in [0,1]")
    if args.subgoal_update_steps < 1:
        raise ValueError("--subgoal-update-steps must be >= 1")
    if args.execute and args.offline_image:
        raise ValueError("--execute cannot be combined with --offline-image")
    if args.execute and not args.table_cam_serial:
        raise ValueError("--execute requires --table-cam-serial")
    task_profiles, task_switching = build_task_profiles(
        args.task_profile, args.ckpt, args.gripper_close_mm, args.gripper_open_mm
    )
    active_profile_key = next(iter(task_profiles))
    initial_profile = task_profiles[active_profile_key]
    active_ckpt = initial_profile["ckpt"]
    active_close_mm = initial_profile["close_mm"]

    # Import the battle-tested hardware utils from the ur7e_ramen_il codebase.
    rw_dir = str(Path(args.realworld_dir).resolve())
    if rw_dir not in sys.path:
        sys.path.insert(0, rw_dir)
    from real_servo_utils import (  # noqa: E402
        move_robot_home, start_servo_pipeline, stop_servo_pipeline,
        update_shared_servo_target,
    )
    from rtde_gripper.robotiq_gripper import RobotiqGripper  # noqa: E402

    from mini_lawam.rollout import MiniLaWAMPolicy  # loads LAM; run from repo root

    print(f"[INFO] execute={args.execute} ({'ROBOT COMMANDS ENABLED' if args.execute else 'dry-run'})")
    train_hw = None if args.train_frame_hw[0] <= 0 else tuple(args.train_frame_hw)
    policy = MiniLaWAMPolicy(active_ckpt, device=args.device, train_frame_hw=train_hw)
    if task_switching:
        print("[TASK] configured profiles:")
        for key, profile in task_profiles.items():
            cfg = MiniLaWAMPolicy.checkpoint_config(profile["ckpt"])
            policy.assert_hot_swap_compatible(cfg)
            profile_target_mode = getattr(cfg, "target_mode", "abs")
            if profile_target_mode not in ("abs", "delta", "joystick"):
                raise ValueError(
                    f"task {key} has unsupported target_mode={profile_target_mode!r}"
                )
            if profile_target_mode != "delta" and args.delta_scale != 1.0:
                raise ValueError(
                    f"task {key} target_mode={profile_target_mode!r} is incompatible "
                    "with non-default --delta-scale"
                )
            print(f"[TASK]   key {key}: ckpt={profile['ckpt']}  "
                  f"close={profile['close_mm']:g} mm")
        print(f"[TASK] active key {active_profile_key}")
    if train_hw:
        print(f"[POLICY] live frames matched to training resolution {train_hw} (h, w) "
              "before the 256x256 resize")
    H = policy.cfg.action_horizon
    k = int(min(max(1, args.exec_steps), H))
    dt = 1.0 / float(args.control_hz)
    print(f"[INFO] chunk horizon H={H}, exec_steps k={k}, control {args.control_hz:.0f} Hz "
          f"(re-plan every {k * dt:.2f}s)")
    print(f"[INFO] workspace box: min={args.ws_min} max={args.ws_max}, max_reach={args.max_reach} m")

    reader = None
    offline_frame = None
    wrist_offline_frame = None
    wrist_reader = None
    rtde_c = rtde_r = servo_state = servo_thread = None
    gripper = None
    locked_rotvec = (
        np.asarray(args.locked_rotvec, dtype=np.float64)
        if args.locked_rotvec is not None
        else DEMO_LOCKED_ROTVEC.copy()
    )
    locked_source = "CLI override" if args.locked_rotvec is not None else "fixed, from demos"
    print(f"[ROT] locked_rotvec ({locked_source}) = {np.round(locked_rotvec, 4)}")
    pygame = screen = font = clock = None

    need_wrist = bool(policy.cfg.use_wrist)
    target_mode = getattr(policy.cfg, "target_mode", "abs")
    include_rz = bool(getattr(policy.cfg, "include_rz", False))
    # current eef xyz is needed as proprioception (use_state) and/or as the
    # composition anchor for cumulative EEF-delta targets. Joystick commands are
    # composed with the live TCP later, at each execution tick.
    need_state = bool(getattr(policy.cfg, "use_state", False)) or target_mode == "delta"
    if target_mode != "delta" and args.delta_scale != 1.0:
        raise ValueError("--delta-scale only applies to a checkpoint with target_mode='delta'")
    if target_mode not in ("abs", "delta", "joystick"):
        raise ValueError(f"unsupported checkpoint target_mode={target_mode!r}")
    if args.enable_rz and not include_rz:
        raise ValueError(
            "--enable-rz requires a checkpoint trained with "
            "--target joystick --include-rz"
        )
    print(f"[INFO] checkpoint use_wrist={need_wrist} "
          f"use_state={getattr(policy.cfg, 'use_state', False)} target={target_mode} "
          f"include_rz={include_rz} enable_rz={args.enable_rz} "
          f"gripper_head={getattr(policy.cfg, 'gripper_head', 'regression')} "
          f"delta_scale={args.delta_scale:g}"
          + (f" action_scale={args.action_scale:g}" if target_mode == "joystick" else ""))
    print(f"[INFO] gripper open lookahead={args.gripper_open_lead_steps} step(s) "
          "(release latches open for the rest of each rollout)")
    if args.show_subgoal:
        print("[VIZ] live predicted-subgoal feature-change overlay enabled "
              f"(alpha={args.subgoal_alpha:g}, refresh every "
              f"{args.subgoal_update_steps} policy inference(s))")

    latest_subgoal_overlay = None
    subgoal_inference_index = 0

    def update_runtime_policy_config():
        """Refresh checkpoint-controlled rollout semantics after a hot swap."""
        nonlocal target_mode, need_state, include_rz
        target_mode = getattr(policy.cfg, "target_mode", "abs")
        include_rz = bool(getattr(policy.cfg, "include_rz", False))
        need_state = bool(getattr(policy.cfg, "use_state", False)) or target_mode == "delta"
        if target_mode != "delta" and args.delta_scale != 1.0:
            raise ValueError(
                "--delta-scale only applies to a checkpoint with target_mode='delta'"
            )
        if target_mode not in ("abs", "delta", "joystick"):
            raise ValueError(f"unsupported checkpoint target_mode={target_mode!r}")
        if args.enable_rz and not include_rz:
            raise ValueError(
                "--enable-rz requires every task checkpoint to be trained with "
                "--target joystick --include-rz"
            )

    def activate_task_profile(key):
        """Hot-swap model weights and gripper configuration while IDLE."""
        nonlocal active_profile_key, active_ckpt, active_close_mm
        nonlocal latest_subgoal_overlay, subgoal_inference_index
        profile = task_profiles.get(str(key))
        if profile is None:
            print(f"[TASK] no profile is assigned to key {key}")
            return False
        if str(key) == active_profile_key:
            print(f"[TASK] key {key} already active: {Path(active_ckpt).name}, "
                  f"close={active_close_mm:g} mm")
            return True

        print(f"[TASK] loading key {key}: {profile['ckpt']}")
        policy.reload_checkpoint(profile["ckpt"])
        update_runtime_policy_config()
        active_profile_key = str(key)
        active_ckpt = profile["ckpt"]
        active_close_mm = profile["close_mm"]
        if gripper is not None:
            gripper.set_close_mm(active_close_mm)
        latest_subgoal_overlay = None
        subgoal_inference_index = 0
        print(f"[TASK] active key {active_profile_key}: {Path(active_ckpt).name}, "
              f"close={active_close_mm:g} mm, target={target_mode}")
        return True

    def cur_state():
        if not need_state:
            return None
        if not args.execute:
            return np.zeros(3, dtype=np.float64)   # dry-run: predictions print as deltas
        return np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)[:3]

    def predict_chunk(frame, wrist_frame):
        """Run inference and apply the target-mode-specific deployment gain."""
        nonlocal latest_subgoal_overlay, subgoal_inference_index
        anchor_xyz = cur_state()
        update_subgoal = should_update_subgoal(
            args.show_subgoal, subgoal_inference_index, args.subgoal_update_steps
        )
        subgoal_inference_index += 1
        policy_out = policy.act(
            frame, wrist_frame, state_xyz=anchor_xyz,
            return_subgoal_change=update_subgoal,
        )
        if update_subgoal:
            chunk, change_grid = policy_out
            model_input = policy.model_input_u8(frame)
            latest_subgoal_overlay = make_subgoal_overlay(
                model_input, change_grid, alpha=args.subgoal_alpha
            )
        else:
            chunk = policy_out
        if target_mode == "delta" and args.delta_scale != 1.0:
            chunk = scale_delta_chunk(chunk, anchor_xyz, args.delta_scale)
        elif target_mode == "joystick":
            chunk = scale_joystick_chunk(
                chunk, args.action_scale, include_rz=include_rz
            )
        return chunk

    try:
        if args.offline_image:
            offline_frame = load_offline_frame(args.offline_image)
            print(f"[INFO] offline frame {offline_frame.shape} from {args.offline_image}")
            if need_wrist:
                wrist_offline_frame = load_offline_frame(args.offline_image, cam="wrist_cam")
        elif args.table_cam_serial:
            reader = RealSenseTableReader(args.table_cam_serial, name="table_cam",
                                          exposure=args.table_exposure, gain=args.table_gain)
            reader.start()
            if need_wrist:
                if not args.wrist_cam_serial:
                    raise ValueError("checkpoint has use_wrist=True -> --wrist-cam-serial required")
                wrist_reader = RealSenseTableReader(args.wrist_cam_serial, name="wrist_cam",
                                                    exposure=args.wrist_exposure,
                                                    gain=args.wrist_gain)
                wrist_reader.start()
                print(f"[RS] wrist_cam serial={args.wrist_cam_serial} (aux view)")
        else:
            raise ValueError("need --table-cam-serial or --offline-image")

        def live_frame():
            """Newest table frame for the pygame display (None if --show-camera off)."""
            if not args.show_camera:
                return None
            if offline_frame is not None:
                return offline_frame
            return reader.latest() if reader is not None else None

        def live_wrist():
            """Newest wrist frame for the display (None unless the policy uses wrist)."""
            if not args.show_camera or not need_wrist:
                return None
            if wrist_offline_frame is not None:
                return wrist_offline_frame
            return wrist_reader.latest() if wrist_reader is not None else None

        if args.execute:
            import rtde_control
            import rtde_receive
            print(f"[RTDE] connecting to {args.robot_ip}")
            rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
            rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
            servo_state, servo_thread = start_servo_pipeline(
                rtde_r=rtde_r, rtde_c=rtde_c, control_hz=args.servo_hz,
                speed=args.servol_speed, acc=args.servol_acc,
                lookahead_time=args.servol_lookahead, gain=args.servol_gain,
                interp_alpha=args.servol_interp_alpha,
                max_pos_step=args.servol_max_pos_step,
                max_rot_step=args.servol_max_rot_step,
            )
            if args.use_gripper_control:
                gripper = LatchedGripper(RobotiqGripper, args.robot_ip,
                                         args.gripper_open_mm, active_close_mm,
                                         args.gripper_speed, args.gripper_force)

        pygame, screen, font, clock = init_pygame(args, two_rows=need_wrist)

        trial = next_trace_trial_number(args.trace_dir)
        session_rollouts = 0
        quit_all = False
        if args.trace_dir:
            print(f"[TRACE] next persistent trial number={trial} "
                  f"(scanned {Path(args.trace_dir)})")
        while session_rollouts < args.num_rollouts and not quit_all:
            latest_subgoal_overlay = None
            subgoal_inference_index = 0
            # ---- idle: wait for S / H / Q ----
            task_keys = "/".join(task_profiles) if task_switching else ""
            task_status = (f"task {active_profile_key}: {Path(active_ckpt).stem}  "
                           f"close={active_close_mm:g}mm")
            print("[IDLE] S=start  H=home  Q=quit"
                  + (f"  {task_keys}=task" if task_keys else ""))
            while True:
                draw_status(pygame, screen, font, "IDLE", task_status,
                            frame_rgb=live_frame(), wrist_rgb=live_wrist(),
                            subgoal_rgb=latest_subgoal_overlay,
                            video_scale=args.video_scale, task_keys=task_keys)
                cmd = poll_cmd(pygame)
                if cmd is not None and cmd.startswith("task:"):
                    if task_switching:
                        activate_task_profile(cmd.split(":", 1)[1])
                        task_status = (
                            f"task {active_profile_key}: {Path(active_ckpt).stem}  "
                            f"close={active_close_mm:g}mm"
                        )
                    continue
                if cmd == "start":
                    rollout_started_wall = time.time()
                    rollout_started_perf = time.perf_counter()
                    break
                if cmd == "home" and args.execute:
                    if gripper is not None:
                        gripper.command("open")
                    move_robot_home(rtde_c, rtde_r, servo_state,
                                    args.home_movej_speed, args.home_movej_acc,
                                    home_q=args.home_q)
                if cmd == "quit":
                    quit_all = True
                    break
                clock.tick(30)
            if quit_all:
                break

            # ---- rollout ----
            print(f"\n[ROLLOUT] trial={trial}"
                  + f"  locked_rotvec={np.round(locked_rotvec, 4)}"
                  + ("  RZ=enabled" if args.enable_rz else "  RZ=locked"))
            time.sleep(max(0.0, args.startup_wait_sec))

            rollout_stem = format_rollout_stem(trial, rollout_started_wall)
            first_table_frame = None
            trace = {"ckpt": str(Path(active_ckpt).resolve()), "execute": args.execute,
                     "task_profile_key": active_profile_key if task_switching else None,
                     "gripper_close_mm": active_close_mm,
                     "target_mode": target_mode,
                     "include_rz": include_rz,
                     "enable_rz": args.enable_rz,
                     "delta_scale": args.delta_scale,
                     "action_scale": args.action_scale if target_mode == "joystick" else None,
                     "gripper_open_lead_steps": args.gripper_open_lead_steps,
                     "exec_steps": k, "control_hz": args.control_hz, "steps": []}
            result = "completed"
            step = 0
            smooth = {"ema_xyz": None, "last_cmd_xyz": None}  # per-rollout smoothing state
            grip_runtime = {"last_cmd": None, "release_latched": False}
            rotation_runtime = {"rz_accum": 0.0}

            def select_grip_value(chunk, step_index):
                """Apply open-only lookahead and report the release transition once."""
                was_latched = grip_runtime["release_latched"]
                grip_val, release_latched, source_index = \
                    select_gripper_with_open_lookahead(
                        chunk=chunk,
                        step_index=step_index,
                        last_cmd=grip_runtime["last_cmd"],
                        release_latched=was_latched,
                        threshold=args.gripper_threshold,
                        open_lead_steps=args.gripper_open_lead_steps,
                    )
                grip_runtime["release_latched"] = release_latched
                if release_latched and not was_latched:
                    lead = source_index - int(step_index)
                    print(f"[GRIPPER] release triggered from chunk[{source_index}] "
                          f"(open lookahead={lead} step{'s' if lead != 1 else ''})")
                return grip_val

            def apply_waypoint(pred_xyz, pred_rz, grip_val, step, sub):
                """One control tick: smooth -> clamp -> servo target + gripper + trace."""
                pred_xyz = np.asarray(pred_xyz, dtype=np.float64)
                pred_rz = float(pred_rz)
                if not np.isfinite(pred_rz):
                    raise ValueError(f"non-finite predicted RZ at step {step}: {pred_rz}")
                grip_cmd = "open" if grip_val <= args.gripper_threshold else "close"
                # For joystick checkpoints pred_xyz is now a scaled incremental
                # command. Compose it from the live TCP at the moment this row is
                # executed. abs/delta policy outputs are already absolute targets.
                if target_mode == "joystick":
                    current_xyz = (
                        np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)[:3]
                        if args.execute else np.zeros(3, dtype=np.float64)
                    )
                    command_xyz = compose_target_xyz(pred_xyz, current_xyz, target_mode)
                else:
                    command_xyz = compose_target_xyz(pred_xyz, None, target_mode)
                beta = float(args.target_ema)
                smooth["ema_xyz"] = command_xyz if smooth["ema_xyz"] is None else \
                    beta * command_xyz + (1.0 - beta) * smooth["ema_xyz"]
                smoothed_xyz = smooth["ema_xyz"]
                if (args.target_deadband > 0.0 and smooth["last_cmd_xyz"] is not None
                        and np.linalg.norm(smoothed_xyz - smooth["last_cmd_xyz"])
                        < args.target_deadband):
                    smoothed_xyz = smooth["last_cmd_xyz"]
                if args.enable_rz:
                    rotation_runtime["rz_accum"] += pred_rz
                    target_rotvec = compose_locked_rotvec_with_rz(
                        locked_rotvec, rotation_runtime["rz_accum"]
                    )
                else:
                    target_rotvec = locked_rotvec
                if args.execute:
                    cur = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
                    tgt_xyz = clamp_abs_target(smoothed_xyz, cur[:3],
                                               args.ws_min, args.ws_max, args.max_reach)
                    update_shared_servo_target(servo_state,
                                               np.concatenate([tgt_xyz, target_rotvec]))
                    if gripper is not None:
                        try:
                            gripper.command(grip_cmd)
                        except Exception as exc:
                            print(f"[GRIPPER] failed: {exc}")
                else:
                    tgt_xyz = smoothed_xyz
                smooth["last_cmd_xyz"] = np.asarray(tgt_xyz, dtype=np.float64)
                grip_runtime["last_cmd"] = grip_cmd
                trace["steps"].append({
                    "step": step, "sub": sub,
                    "pred_xyz": pred_xyz.tolist(),
                    "pred_xyz_kind": (
                        "scaled_joystick_delta" if target_mode == "joystick"
                        else "absolute_target"
                    ),
                    "tgt_xyz": np.asarray(tgt_xyz, dtype=float).tolist(),
                    "pred_rz_delta": pred_rz if include_rz else None,
                    "rz_accum": rotation_runtime["rz_accum"] if args.enable_rz else 0.0,
                    "tgt_rotvec": np.asarray(target_rotvec, dtype=float).tolist(),
                    "grip": float(grip_val), "grip_cmd": grip_cmd,
                })
                return grip_cmd

            def read_frames():
                nonlocal first_table_frame
                fr = offline_frame if offline_frame is not None else reader.read_rgb()
                wf = None
                if need_wrist:
                    wf = (wrist_offline_frame if wrist_offline_frame is not None
                          else wrist_reader.read_rgb())
                if first_table_frame is None:
                    first_table_frame = fr.copy()
                return fr, wf

            frames_dir = None
            if args.save_frames > 0 and args.trace_dir:
                frames_dir = Path(args.trace_dir) / f"frames_{rollout_stem}"
                frames_dir.mkdir(parents=True, exist_ok=True)
                print(f"[FRAMES] saving every {args.save_frames} steps -> {frames_dir}")

            def dump_frames(step, fr, wf):
                if frames_dir is None or step % args.save_frames != 0:
                    return
                from PIL import Image
                Image.fromarray(fr).save(frames_dir / f"{step:05d}_table.jpg", quality=92)
                if wf is not None:
                    Image.fromarray(wf).save(frames_dir / f"{step:05d}_wrist.jpg", quality=92)

            ensemble = {}   # temporal-ensemble buffer: abs step -> list of chunk rows (oldest..newest)
            stop_cmd = None
            while step < args.max_steps:
                t0 = time.time()
                draw_status(pygame, screen, font, "ROLLOUT",
                            f"step {step}  E=end H=home Q=quit",
                            frame_rgb=live_frame(), wrist_rgb=live_wrist(),
                            subgoal_rgb=latest_subgoal_overlay,
                            video_scale=args.video_scale)
                cmd = poll_cmd(pygame)
                if cmd is not None and cmd.startswith("task:"):
                    print("[TASK] switch ignored during rollout; press E, then select "
                          "the task from the IDLE screen")
                if cmd in ("end", "home", "quit"):
                    stop_cmd = cmd
                    break

                if args.temporal_ensemble:
                    # Re-plan every step; execute a weighted average over all overlapping
                    # chunks' predictions for THIS timestep (newest weighted most).
                    frame, wrist_frame = read_frames()
                    dump_frames(step, frame, wrist_frame)
                    t_inf = time.time()
                    chunk = predict_chunk(frame, wrist_frame)
                    inf_ms = (time.time() - t_inf) * 1e3
                    for j in range(H):
                        ensemble.setdefault(step + j, []).append(chunk[j])
                    preds = np.asarray(ensemble.pop(step, [chunk[0]]))
                    n = len(preds)
                    age = np.arange(n - 1, -1, -1)                        # newest -> 0
                    w = np.exp(-float(args.te_m) * age)
                    w /= w.sum()
                    avg = (preds * w[:, None]).sum(0)
                    # Ensemble-average motion targets/commands, but use the newest
                    # chunk for gripper timing. Only release may look ahead.
                    grip_val = select_grip_value(chunk, step_index=0)
                    pred_rz = avg[3] if include_rz else 0.0
                    grip_cmd = apply_waypoint(
                        avg[:3], pred_rz, grip_val, step, sub=0
                    )
                    if step % 8 == 0:
                        print("gripper chunk:", chunk[:, -1])
                        mv_H = np.linalg.norm(chunk[-1, :3] - chunk[0, :3]) * 1e3
                        xyz_label = "avg_joy_delta" if target_mode == "joystick" else "avg_xyz"
                        rz_text = f"  rz_delta={pred_rz:+.4f}" if include_rz else ""
                        print(f"[STEP {step}] inf={inf_ms:.0f}ms te_n={n}  "
                              f"{xyz_label}={np.round(avg[:3], 4)}  "
                              f"{rz_text}"
                              f"grip={grip_val:+.2f}->{grip_cmd}  "
                              f"chunk_span[0->{H - 1}]={mv_H:.0f}mm")
                    step += 1
                    sleep_t = dt - (time.time() - t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    continue

                # --- default: receding horizon, execute k waypoints per re-plan ---
                frame, wrist_frame = read_frames()
                dump_frames(step, frame, wrist_frame)
                t_inf = time.time()
                chunk = predict_chunk(frame, wrist_frame)
                inf_ms = (time.time() - t_inf) * 1e3
                for i in range(k):
                    t0 = time.time()
                    if args.show_camera:
                        draw_status(pygame, screen, font, "ROLLOUT",
                                    f"step {step}  E=end H=home Q=quit",
                                    frame_rgb=live_frame(), wrist_rgb=live_wrist(),
                                    subgoal_rgb=latest_subgoal_overlay,
                                    video_scale=args.video_scale)
                    cmd = poll_cmd(pygame)
                    if cmd is not None and cmd.startswith("task:"):
                        print("[TASK] switch ignored during rollout; press E, then select "
                              "the task from the IDLE screen")
                    if cmd in ("end", "home", "quit"):
                        stop_cmd = cmd
                        break
                    grip_val = select_grip_value(chunk, step_index=i)
                    pred_rz = chunk[i, 3] if include_rz else 0.0
                    grip_cmd = apply_waypoint(
                        chunk[i, :3], pred_rz, grip_val, step, sub=i
                    )
                    if i == 0:
                        print("gripper chunk:", chunk[:, -1])
                        mv_k = np.linalg.norm(chunk[k - 1, :3] - chunk[0, :3]) * 1e3
                        mv_H = np.linalg.norm(chunk[-1, :3] - chunk[0, :3]) * 1e3
                        xyz_label = "joy_delta" if target_mode == "joystick" else "pred_xyz"
                        rz_text = f"  rz_delta={pred_rz:+.4f}" if include_rz else ""
                        print(f"[STEP {step}] inf={inf_ms:.0f}ms  "
                              f"{xyz_label}={np.round(chunk[i, :3], 4)}  "
                              f"{rz_text}"
                              f"grip={grip_val:+.2f}->{grip_cmd}  "
                              f"chunk_span[0->{k - 1}]={mv_k:.0f}mm "
                              f"[0->{H - 1}]={mv_H:.0f}mm")
                    step += 1
                    if step >= args.max_steps:
                        break
                    sleep_t = dt - (time.time() - t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                if stop_cmd is not None:
                    break

            if stop_cmd is not None:
                result = {"end": "ended", "home": "go_home", "quit": "quit"}[stop_cmd]

            trace["result"] = result
            rollout_finished_wall = time.time()
            duration_sec = round(time.perf_counter() - rollout_started_perf, 3)
            print(f"[ROLLOUT] result={result} duration={duration_sec:.3f}s")
            if args.trace_dir:
                out = Path(args.trace_dir)
                out.mkdir(parents=True, exist_ok=True)
                first_frame_path = None
                if first_table_frame is not None:
                    from PIL import Image
                    first_frame_fp = out / f"{rollout_stem}_table_cam_first.png"
                    Image.fromarray(first_table_frame).save(first_frame_fp)
                    first_frame_path = first_frame_fp.as_posix()
                    print(f"[FIRST FRAME] {first_frame_fp}")
                summary = {
                    "trial": trial,
                    "ckpt": trace["ckpt"],
                    "task_profile_key": trace["task_profile_key"],
                    "gripper_close_mm": trace["gripper_close_mm"],
                    "target_mode": target_mode,
                    "include_rz": include_rz,
                    "enable_rz": args.enable_rz,
                    "final_rz_accum": rotation_runtime["rz_accum"],
                    "delta_scale": args.delta_scale,
                    "action_scale": (
                        args.action_scale if target_mode == "joystick" else None
                    ),
                    "gripper_open_lead_steps": args.gripper_open_lead_steps,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                                time.localtime(rollout_started_wall)),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                                 time.localtime(rollout_finished_wall)),
                    "duration_sec": duration_sec,
                    "duration_hms": format_duration_hms(duration_sec),
                    "steps_executed": len(trace["steps"]),
                    "first_table_frame": first_frame_path,
                }
                fp = out / f"{rollout_stem}_summary.json"
                fp.write_text(json.dumps(summary, indent=2) + "\n")
                print(f"[SUMMARY] {fp}")

            if result == "go_home" and args.execute:
                if gripper is not None:
                    gripper.command("open")
                move_robot_home(rtde_c, rtde_r, servo_state,
                                args.home_movej_speed, args.home_movej_acc,
                                home_q=args.home_q)
            if result == "quit":
                quit_all = True
            trial += 1
            session_rollouts += 1

    finally:
        stop_servo_pipeline(servo_state, servo_thread)
        if gripper is not None:
            gripper.stop()
        if reader is not None:
            reader.stop()
        if wrist_reader is not None:
            wrist_reader.stop()
        if rtde_c is not None:
            try:
                rtde_c.stopScript()
            except Exception:
                pass
        if pygame is not None:
            try:
                pygame.quit()
            except Exception:
                pass
        print("[INFO] rollout closed")


if __name__ == "__main__":
    main()
