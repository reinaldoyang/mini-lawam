#!/usr/bin/env python3
"""Real-world UR7e rollout for the mini_lawam BC policy (the venue SEAM).

Pattern follows ur7e_ramen_il/scripts/real_world/lapa_latent_real_servoL.py:
  - background 500 Hz servo thread (real_servo_utils.start_servo_pipeline)
    interpolates toward a shared target TCP; we only update the target.
  - background RealSense reader caches the latest table_cam frame.
  - pygame control: S=start rollout, E=end rollout, H=home, Q=quit.
  - DRY-RUN by default; add --execute to actually send servoL/gripper.

mini_lawam-specific differences vs the LAPA baseline:
  - policy output is an ABSOLUTE [H=32, 4] chunk: [eef_pos_base(3), gripper(1)]
    (same base frame as rtde getActualTCPPose position). No delta composition:
        target_tcp = [pred_xyz, locked_rotvec]
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
import sys
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_REALWORLD_DIR = "/home/iclu200/reinaldoyang/ur7e_ramen_il/scripts/real_world"

TABLE_CAM_KEY = "table_cam"
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30
ROBOTIQ_SOCKET_PORT = 63352

# Same home joint pose as record_real.py / real_servo_utils.move_robot_home.
HOME_Q = [0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]

# Fixed locked TCP orientation (axis-angle rotvec, base frame): measured from the
# dataset's eef_quat_base across ALL demo frames (constant within 0.24 deg).
# = 180 deg about base Y, i.e. tool pointing straight down.
DEMO_LOCKED_ROTVEC = np.array([0.0036, 3.14094, -0.00024], dtype=np.float64)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="results/mini_lawam/ckpt_114ep_wrist.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--realworld-dir", default=DEFAULT_REALWORLD_DIR,
                   help="ur7e_ramen_il/scripts/real_world (real_servo_utils + rtde_gripper)")

    # cadence / horizon
    p.add_argument("--control-hz", type=float, default=20.0,
                   help="waypoint rate; MUST match training rate (20 Hz)")
    p.add_argument("--exec-steps", type=int, default=8,
                   help="chunk steps executed before re-planning (receding horizon)")
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
                   help="chunk[:,3] <= thr => open, > thr => close (open=-1, close=+1)")

    # camera
    p.add_argument("--table-cam-serial", default="", help="RealSense serial for table_cam")
    p.add_argument("--wrist-cam-serial", default="",
                   help="RealSense serial for wrist_cam (REQUIRED if the checkpoint "
                        "was trained with use_wrist=True)")
    p.add_argument("--show-camera", action="store_true",
                   help="live window: raw table_cam + the 256x256 model-input view")
    p.add_argument("--video-scale", type=float, default=1.0, help="display window scale")
    p.add_argument("--offline-image", default=None,
                   help="HDF5 path or image file: dry-run inference without a camera. "
                        "For HDF5, uses data/demo_0/obs/table_cam[0].")

    # UI / debug
    p.add_argument("--pygame-window-w", type=int, default=560)
    p.add_argument("--pygame-window-h", type=int, default=150)
    p.add_argument("--trace-dir", default=None, help="save per-rollout JSON traces here")
    return p


# ----------------------------------------------------------------------------
# Camera: single-cam background reader (slim version of LAPA's pair reader)
# ----------------------------------------------------------------------------
class RealSenseTableReader:
    def __init__(self, serial: str):
        import pyrealsense2 as rs
        self.rs = rs
        self.serial = str(serial)
        self.lock = threading.Lock()
        self.last_frame = None       # RGB uint8 [H,W,3]
        self.last_time = None
        self.timeouts = 0
        self.running = False
        self.pipeline = None
        self.thread = None

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
        pipeline.start(config)
        self.pipeline = pipeline
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
        print(f"[RS] started table_cam serial={self.serial} {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")

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
        def mm_to_raw(v, open_ref):
            open_ref = max(float(open_ref), 1e-6)
            return int(np.clip(round(255.0 * (1.0 - float(v) / open_ref)), 0, 255))

        def pct_to_raw(v):
            v = float(v)
            return int(np.clip(round(v * 255.0 / 100.0 if v <= 100.0 else v), 0, 255))

        self._g = RobotiqGripper()
        self._g.connect(str(robot_ip), ROBOTIQ_SOCKET_PORT)
        self.open_raw = mm_to_raw(open_mm, open_mm)
        self.close_raw = mm_to_raw(close_mm, open_mm)
        self.speed_raw = pct_to_raw(speed)
        self.force_raw = pct_to_raw(force)
        self.state = None
        if hasattr(self._g, "activate"):
            print("[GRIPPER] activating")
            self._g.activate(auto_calibrate=False)

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
def init_pygame(args):
    import pygame
    pygame.init()
    pygame.display.set_caption("mini_lawam UR7e rollout")
    if args.show_camera:
        vs = float(args.video_scale)
        raw_w, raw_h, mv = int(480 * vs), int(360 * vs), int(256 * vs)
        w = 16 + raw_w + 10 + mv + 16
        h = 96 + max(raw_h, mv) + 30
        screen = pygame.display.set_mode((max(w, args.pygame_window_w), h))
    else:
        screen = pygame.display.set_mode((args.pygame_window_w, args.pygame_window_h))
    font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()
    return pygame, screen, font, clock


def draw_status(pygame, screen, font, mode, extra="", frame_rgb=None, video_scale=1.0):
    """Status text + (optionally) the live camera: raw view | 256x256 model input."""
    screen.fill((20, 20, 20))
    for i, line in enumerate([f"Mode: {mode}", "S=start  E=end  H=home  Q=quit", extra]):
        if line:
            screen.blit(font.render(line, True, (0, 255, 0)), (16, 18 + 26 * i))
    if frame_rgb is not None:
        vs = float(video_scale)
        raw_w, raw_h, mv = int(480 * vs), int(360 * vs), int(256 * vs)
        # pygame surfaces are (W,H,3)
        surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        raw = pygame.transform.smoothscale(surf, (raw_w, raw_h))
        model = pygame.transform.smoothscale(surf, (mv, mv))  # what the policy ingests
        y0 = 96
        screen.blit(raw, (16, y0))
        screen.blit(model, (16 + raw_w + 10, y0))
        small = pygame.font.SysFont(None, 22)
        screen.blit(small.render("table_cam raw", True, (0, 255, 0)),
                    (16, y0 + raw_h + 6))
        screen.blit(small.render("model input 256x256", True, (0, 255, 0)),
                    (16 + raw_w + 10, y0 + mv + 6))
    pygame.display.flip()


def poll_cmd(pygame):
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            return "quit"
        if ev.type == pygame.KEYDOWN:
            return {pygame.K_s: "start", pygame.K_e: "end",
                    pygame.K_h: "home", pygame.K_q: "quit"}.get(ev.key)
    return None


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
    if args.execute and args.offline_image:
        raise ValueError("--execute cannot be combined with --offline-image")
    if args.execute and not args.table_cam_serial:
        raise ValueError("--execute requires --table-cam-serial")

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
    policy = MiniLaWAMPolicy(args.ckpt, device=args.device)
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
    locked_rotvec = None
    pygame = screen = font = clock = None

    need_wrist = bool(policy.cfg.use_wrist)
    print(f"[INFO] checkpoint use_wrist={need_wrist}")

    try:
        if args.offline_image:
            offline_frame = load_offline_frame(args.offline_image)
            print(f"[INFO] offline frame {offline_frame.shape} from {args.offline_image}")
            if need_wrist:
                wrist_offline_frame = load_offline_frame(args.offline_image, cam="wrist_cam")
        elif args.table_cam_serial:
            reader = RealSenseTableReader(args.table_cam_serial)
            reader.start()
            if need_wrist:
                if not args.wrist_cam_serial:
                    raise ValueError("checkpoint has use_wrist=True -> --wrist-cam-serial required")
                wrist_reader = RealSenseTableReader(args.wrist_cam_serial)
                wrist_reader.start()
                print(f"[RS] wrist_cam serial={args.wrist_cam_serial} (aux view)")
        else:
            raise ValueError("need --table-cam-serial or --offline-image")

        def live_frame():
            """Newest frame for the pygame display (None if --show-camera off)."""
            if not args.show_camera:
                return None
            if offline_frame is not None:
                return offline_frame
            return reader.latest() if reader is not None else None

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
                                         args.gripper_open_mm, args.gripper_close_mm,
                                         args.gripper_speed, args.gripper_force)

            # Locked orientation: FIXED constant (the orientation every demo used),
            # combined with the policy's predicted xyz at each waypoint.
            if args.locked_rotvec is not None:
                locked_rotvec = np.asarray(args.locked_rotvec, dtype=np.float64)
                print(f"[ROT] locked_rotvec (CLI override) = {np.round(locked_rotvec, 4)}")
            else:
                locked_rotvec = DEMO_LOCKED_ROTVEC.copy()
                print(f"[ROT] locked_rotvec (fixed, from demos) = {np.round(locked_rotvec, 4)}")

        pygame, screen, font, clock = init_pygame(args)

        trial, quit_all = 0, False
        while trial < args.num_rollouts and not quit_all:
            # ---- idle: wait for S / H / Q ----
            print("[IDLE] S=start  H=home  Q=quit")
            while True:
                draw_status(pygame, screen, font, "IDLE", "waiting for S / H / Q",
                            frame_rgb=live_frame(), video_scale=args.video_scale)
                cmd = poll_cmd(pygame)
                if cmd == "start":
                    break
                if cmd == "home" and args.execute:
                    if gripper is not None:
                        gripper.command("open")
                    move_robot_home(rtde_c, rtde_r, servo_state,
                                    args.home_movej_speed, args.home_movej_acc)
                if cmd == "quit":
                    quit_all = True
                    break
                clock.tick(30)
            if quit_all:
                break

            # ---- rollout ----
            print(f"\n[ROLLOUT] trial={trial}"
                  + (f"  locked_rotvec={np.round(locked_rotvec, 4)}" if args.execute else ""))
            time.sleep(max(0.0, args.startup_wait_sec))

            trace = {"ckpt": str(Path(args.ckpt).resolve()), "execute": args.execute,
                     "exec_steps": k, "control_hz": args.control_hz, "steps": []}
            result = "completed"
            step = 0
            ema_xyz = None      # EMA state across waypoints (reset per rollout)
            last_cmd_xyz = None  # last commanded target (for the deadband)
            while step < args.max_steps:
                # keyboard control
                draw_status(pygame, screen, font, "ROLLOUT",
                            f"step {step}  E=end H=home Q=quit",
                            frame_rgb=live_frame(), video_scale=args.video_scale)
                cmd = poll_cmd(pygame)
                if cmd in ("end", "home", "quit"):
                    result = {"end": "ended", "home": "go_home", "quit": "quit"}[cmd]
                    break

                # 1) fresh frame(s) -> absolute action chunk [H,4]
                frame = offline_frame if offline_frame is not None else reader.read_rgb()
                wrist_frame = None
                if need_wrist:
                    wrist_frame = (wrist_offline_frame if wrist_offline_frame is not None
                                   else wrist_reader.read_rgb())
                t_inf = time.time()
                chunk = policy.act(frame, wrist_frame)          # physical units
                inf_ms = (time.time() - t_inf) * 1e3

                # 2) execute the first k waypoints at control_hz (receding horizon)
                stop_cmd = None
                for i in range(k):
                    t0 = time.time()
                    # react to E/H/Q within one waypoint (50 ms), not per re-plan
                    if args.show_camera:
                        draw_status(pygame, screen, font, "ROLLOUT",
                                    f"step {step}  E=end H=home Q=quit",
                                    frame_rgb=live_frame(),
                                    video_scale=args.video_scale)
                    cmd = poll_cmd(pygame)
                    if cmd in ("end", "home", "quit"):
                        stop_cmd = cmd
                        break
                    pred_xyz = chunk[i, :3].astype(np.float64)
                    grip_val = float(chunk[i, 3])
                    grip_cmd = "open" if grip_val <= args.gripper_threshold else "close"

                    # --- smoothing: EMA across waypoints, then deadband ---
                    beta = float(args.target_ema)
                    ema_xyz = pred_xyz if ema_xyz is None else \
                        beta * pred_xyz + (1.0 - beta) * ema_xyz
                    smoothed_xyz = ema_xyz
                    if (args.target_deadband > 0.0 and last_cmd_xyz is not None
                            and np.linalg.norm(smoothed_xyz - last_cmd_xyz)
                            < args.target_deadband):
                        smoothed_xyz = last_cmd_xyz  # hold: change too small to act on

                    if args.execute:
                        cur = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
                        tgt_xyz = clamp_abs_target(smoothed_xyz, cur[:3],
                                                   args.ws_min, args.ws_max, args.max_reach)
                        target_tcp = np.concatenate([tgt_xyz, locked_rotvec])
                        update_shared_servo_target(servo_state, target_tcp)
                        if gripper is not None:
                            try:
                                gripper.command(grip_cmd)
                            except Exception as exc:
                                print(f"[GRIPPER] failed: {exc}")
                    else:
                        tgt_xyz = smoothed_xyz  # dry-run: no clamping reference available
                    last_cmd_xyz = np.asarray(tgt_xyz, dtype=np.float64)

                    if i == 0:
                        # intended motion WITHIN the predicted chunk (the key
                        # health signal: ~0 => policy predicts "stay put")
                        mv_k = np.linalg.norm(chunk[k - 1, :3] - chunk[0, :3]) * 1e3
                        mv_H = np.linalg.norm(chunk[-1, :3] - chunk[0, :3]) * 1e3
                        print(f"[STEP {step}] inf={inf_ms:.0f}ms  "
                              f"pred_xyz={np.round(pred_xyz, 4)}  grip={grip_val:+.2f}->{grip_cmd}  "
                              f"chunk_move[0->{k - 1}]={mv_k:.0f}mm [0->{H - 1}]={mv_H:.0f}mm")
                    trace["steps"].append({
                        "step": step, "sub": i,
                        "pred_xyz": np.asarray(pred_xyz, dtype=float).tolist(),
                        "tgt_xyz": np.asarray(tgt_xyz, dtype=float).tolist(),
                        "grip": grip_val, "grip_cmd": grip_cmd,
                    })
                    step += 1
                    if step >= args.max_steps:
                        break
                    sleep_t = dt - (time.time() - t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                if stop_cmd is not None:
                    result = {"end": "ended", "home": "go_home", "quit": "quit"}[stop_cmd]
                    break

            trace["result"] = result
            if args.trace_dir:
                out = Path(args.trace_dir)
                out.mkdir(parents=True, exist_ok=True)
                fp = out / f"trial_{trial:03d}_{time.strftime('%Y%m%d%H%M%S')}.json"
                fp.write_text(json.dumps(trace, indent=2))
                print(f"[TRACE] {fp}")

            if result == "go_home" and args.execute:
                if gripper is not None:
                    gripper.command("open")
                move_robot_home(rtde_c, rtde_r, servo_state,
                                args.home_movej_speed, args.home_movej_acc)
            if result == "quit":
                quit_all = True
            trial += 1

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
