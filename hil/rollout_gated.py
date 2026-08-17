#!/usr/bin/env python3
"""Run Mini-LaWAM with the learned Stage 1 correction and Stage 2 gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .actions import action_from_target, clamp_target_pose, target_from_action
from .collect_corrections import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HOME_Q,
    DEFAULT_LOCKED_ROTVEC,
    DEFAULT_REALWORLD_DIR,
    DEFAULT_WS_MAX,
    DEFAULT_WS_MIN,
)
from .gated_policy import GatedResidualPolicy, compose_gated_action
from .policy import MiniLaWAMBasePolicy
from .robot import DualRealSense, RobotRuntime


DEFAULT_STAGE2_CHECKPOINT = "results/hil/mini_lawam_stage2_gate/residual_gate_stage2.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    model = parser.add_argument_group("Mini-LaWAM base policy")
    model.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    model.add_argument("--device", default="cuda")
    model.add_argument("--train-frame-hw", type=int, nargs=2, default=[240, 320], metavar=("H", "W"))
    model.add_argument("--action-scale", type=float, default=1.0)
    model.add_argument("--delta-scale", type=float, default=1.0)
    model.add_argument("--enable-rz", action="store_true")
    model.add_argument("--temporal-ensemble", action="store_true")
    model.add_argument("--te-m", type=float, default=0.2)
    model.add_argument("--target-ema", type=float, default=1.0)
    model.add_argument("--target-deadband", type=float, default=0.0)
    model.add_argument("--gripper-open-lead-steps", type=int, default=0)

    correction = parser.add_argument_group("gated residual policy")
    correction.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2_CHECKPOINT)
    correction.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="Gate-on probability; default uses the Stage 2 checkpoint evaluation threshold.",
    )
    correction.add_argument(
        "--gate-hysteresis",
        type=float,
        default=0.05,
        help="Gate turns off at threshold minus this value.",
    )
    correction.add_argument(
        "--correction-source-frame-hw",
        type=int,
        nargs=2,
        default=[168, 224],
        metavar=("H", "W"),
        help="Original HIL collection image size before any converted-dataset resize.",
    )
    correction.add_argument(
        "--correction-frame-hw",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("H", "W"),
        help="Image size used to train Stage 1/2.",
    )

    camera = parser.add_argument_group("cameras and display")
    camera.add_argument("--table-cam-serial", required=True)
    camera.add_argument("--wrist-cam-serial", required=True)
    camera.add_argument("--table-exposure", type=float, default=180.0)
    camera.add_argument("--table-gain", type=float, default=16.0)
    camera.add_argument("--wrist-exposure", type=float, default=100.0)
    camera.add_argument("--wrist-gain", type=float, default=16.0)
    camera.add_argument("--camera-max-age", type=float, default=0.5)
    camera.add_argument("--show-camera", action="store_true")
    camera.add_argument("--show-subgoal", action="store_true")
    camera.add_argument("--subgoal-alpha", type=float, default=0.55)
    camera.add_argument("--subgoal-update-steps", type=int, default=8)
    camera.add_argument("--video-scale", type=float, default=1.0)
    camera.add_argument("--pygame-window-w", type=int, default=680)
    camera.add_argument("--pygame-window-h", type=int, default=150)
    camera.add_argument("--no-display", action="store_true")

    robot = parser.add_argument_group("UR7e")
    robot.add_argument("--execute", action="store_true")
    robot.add_argument("--robot-ip", default="140.96.93.7")
    robot.add_argument("--realworld-dir", default=DEFAULT_REALWORLD_DIR)
    robot.add_argument("--control-hz", type=float, default=20.0)
    robot.add_argument("--servo-hz", type=float, default=500.0)
    robot.add_argument("--servol-speed", type=float, default=0.25)
    robot.add_argument("--servol-acc", type=float, default=0.25)
    robot.add_argument("--servol-lookahead", type=float, default=0.08)
    robot.add_argument("--servol-gain", type=float, default=300.0)
    robot.add_argument("--servol-interp-alpha", type=float, default=0.25)
    robot.add_argument("--servol-max-pos-step", type=float, default=0.002)
    robot.add_argument("--servol-max-rot-step", type=float, default=0.005)
    robot.add_argument("--home-q", type=float, nargs=6, default=DEFAULT_HOME_Q)
    robot.add_argument("--home-movej-speed", type=float, default=0.6)
    robot.add_argument("--home-movej-acc", type=float, default=1.2)
    robot.add_argument("--locked-rotvec", type=float, nargs=3, default=DEFAULT_LOCKED_ROTVEC)
    robot.add_argument("--ws-min", type=float, nargs=3, default=DEFAULT_WS_MIN)
    robot.add_argument("--ws-max", type=float, nargs=3, default=DEFAULT_WS_MAX)
    robot.add_argument(
        "--max-reach",
        "--max-target-lead",
        dest="max_target_lead",
        type=float,
        default=0.015,
    )

    gripper = parser.add_argument_group("Robotiq gripper")
    gripper.add_argument("--use-gripper-control", action="store_true")
    gripper.add_argument("--gripper-open-mm", type=float, default=52.0)
    gripper.add_argument("--gripper-close-mm", type=float, default=23.0)
    gripper.add_argument("--gripper-speed", type=float, default=100.0)
    gripper.add_argument("--gripper-force", type=float, default=50.0)
    gripper.add_argument("--gripper-threshold", type=float, default=0.0)

    run = parser.add_argument_group("rollout")
    run.add_argument("--max-steps", type=int, default=2000)
    run.add_argument("--num-rollouts", type=int, default=1)
    run.add_argument("--startup-wait-sec", type=float, default=1.0)
    run.add_argument("--auto-start", action="store_true", help="Start without waiting for the S key.")
    run.add_argument("--home-after-rollout", action="store_true")
    run.add_argument("--log-every", type=int, default=8)
    run.add_argument(
        "--trace-dir",
        default=None,
        help=(
            "Save a summary JSON and the first table-camera frame for each rollout. "
            "Trial numbering resumes from the highest trial_N already present."
        ),
    )
    run.add_argument(
        "--save-frames",
        type=int,
        default=0,
        help=(
            "Save table-camera frames every N control steps under --trace-dir "
            "(0 disables periodic frame capture)."
        ),
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("ckpt", "stage2_checkpoint"):
        if not Path(getattr(args, name)).expanduser().is_file():
            parser.error(f"--{name.replace('_', '-')} does not exist: {getattr(args, name)}")
    positive = (
        "control_hz",
        "servo_hz",
        "camera_max_age",
        "max_target_lead",
        "servol_speed",
        "servol_acc",
        "servol_lookahead",
        "servol_gain",
        "servol_interp_alpha",
        "servol_max_pos_step",
        "servol_max_rot_step",
        "log_every",
        "max_steps",
        "num_rollouts",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.startup_wait_sec < 0.0 or not np.isfinite(args.startup_wait_sec):
        parser.error("--startup-wait-sec must be finite and non-negative")
    if not np.isfinite(args.action_scale) or args.action_scale < 0.0:
        parser.error("--action-scale must be finite and non-negative")
    if not np.isfinite(args.delta_scale) or args.delta_scale < 0.0:
        parser.error("--delta-scale must be finite and non-negative")
    if not np.isfinite(args.te_m) or args.te_m < 0.0:
        parser.error("--te-m must be finite and non-negative")
    if not np.isfinite(args.target_ema) or not 0.0 <= args.target_ema <= 1.0:
        parser.error("--target-ema must be in [0,1]")
    if not np.isfinite(args.target_deadband) or args.target_deadband < 0.0:
        parser.error("--target-deadband must be finite and non-negative")
    if not np.isfinite(args.subgoal_alpha) or not 0.0 <= args.subgoal_alpha <= 1.0:
        parser.error("--subgoal-alpha must be in [0,1]")
    if args.subgoal_update_steps < 1:
        parser.error("--subgoal-update-steps must be positive")
    if any(value <= 0 for value in (*args.correction_source_frame_hw, *args.correction_frame_hw)):
        parser.error("correction source/model frame sizes must be positive")
    if not (all(value > 0 for value in args.train_frame_hw) or all(value <= 0 for value in args.train_frame_hw)):
        parser.error("--train-frame-hw must contain two positive values or 0 0")
    if args.gate_threshold is not None and not 0.0 <= args.gate_threshold <= 1.0:
        parser.error("--gate-threshold must be in [0,1]")
    effective_threshold = 0.5 if args.gate_threshold is None else args.gate_threshold
    if not 0.0 <= args.gate_hysteresis <= effective_threshold:
        parser.error("--gate-hysteresis must be non-negative and no greater than the gate threshold")
    if np.any(np.asarray(args.ws_min) >= np.asarray(args.ws_max)):
        parser.error("every --ws-min component must be smaller than --ws-max")
    if args.show_subgoal and (not args.show_camera or args.no_display):
        parser.error("--show-subgoal requires --show-camera and cannot use --no-display")
    if args.no_display and not args.auto_start:
        parser.error("--no-display requires --auto-start; stop a headless rollout with Ctrl-C")
    if args.gripper_open_lead_steps < 0:
        parser.error("--gripper-open-lead-steps must be non-negative")
    if args.save_frames < 0:
        parser.error("--save-frames must be non-negative")
    if args.save_frames > 0 and not args.trace_dir:
        parser.error("--save-frames requires --trace-dir")
    if not np.all(np.isfinite(args.locked_rotvec)):
        parser.error("--locked-rotvec must contain finite values")


class CorrectionImagePreprocessor:
    """Reproduce collection storage resize followed by converted-dataset resize."""

    def __init__(self, source_hw: Sequence[int], model_hw: Sequence[int]) -> None:
        from torchvision.transforms import v2

        self.source_hw = (int(source_hw[0]), int(source_hw[1]))
        self.model_hw = (int(model_hw[0]), int(model_hw[1]))
        self.resize = v2.Resize(self.model_hw, antialias=True)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        value = np.asarray(image, dtype=np.uint8)
        if value.shape[:2] != self.source_hw:
            value = cv2.resize(
                value,
                (self.source_hw[1], self.source_hw[0]),
                interpolation=cv2.INTER_AREA,
            )
        if self.source_hw != self.model_hw:
            tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)
            value = self.resize(tensor).to(torch.uint8).permute(1, 2, 0).numpy()
        return np.ascontiguousarray(value, dtype=np.uint8)


def _init_ui(args):
    if args.no_display:
        return None, None, None, None
    from mini_lawam.rollout_ur7e import init_pygame

    return init_pygame(args, two_rows=True)


def _draw_ui(pygame, screen, font, args, mode: str, extra: str, table=None, wrist=None, subgoal=None) -> None:
    if pygame is None:
        return
    from mini_lawam.rollout_ur7e import draw_status

    draw_status(
        pygame,
        screen,
        font,
        mode,
        extra,
        frame_rgb=table if args.show_camera else None,
        wrist_rgb=wrist if args.show_camera else None,
        subgoal_rgb=subgoal,
        video_scale=args.video_scale,
    )


def _poll_ui(pygame) -> Optional[str]:
    if pygame is None:
        return None
    from mini_lawam.rollout_ur7e import poll_cmd

    return poll_cmd(pygame)


def _move_home(robot: RobotRuntime, pygame) -> None:
    """Move home with the servo stopped and discard queued key-repeat events."""
    robot.move_home()
    if pygame is not None:
        pygame.event.clear(pygame.KEYDOWN)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    train_hw = None if args.train_frame_hw[0] <= 0 else tuple(args.train_frame_hw)

    base_policy = MiniLaWAMBasePolicy(
        args.ckpt,
        device=args.device,
        train_frame_hw=train_hw,
        action_scale=args.action_scale,
        delta_scale=args.delta_scale,
        enable_rz=args.enable_rz,
        gripper_threshold=args.gripper_threshold,
        workspace_min=args.ws_min,
        workspace_max=args.ws_max,
        max_target_lead=args.max_target_lead,
        temporal_ensemble=args.temporal_ensemble,
        temporal_ensemble_decay=args.te_m,
        gripper_open_lead_steps=args.gripper_open_lead_steps,
        target_ema=args.target_ema,
        target_deadband=args.target_deadband,
        show_subgoal=args.show_subgoal,
        subgoal_alpha=args.subgoal_alpha,
        subgoal_update_steps=args.subgoal_update_steps,
    )
    if args.enable_rz and not base_policy.include_rz:
        parser.error("--enable-rz requires a base checkpoint trained with RZ")
    correction_policy = GatedResidualPolicy(
        args.stage2_checkpoint,
        device=args.device,
        gate_threshold=args.gate_threshold,
        gate_hysteresis=args.gate_hysteresis,
    )
    correction_policy.assert_base_policy_compatible(args.ckpt, target_mode=base_policy.target_mode)
    correction_images = CorrectionImagePreprocessor(
        args.correction_source_frame_hw,
        args.correction_frame_hw,
    )

    cameras = DualRealSense(
        table_serial=args.table_cam_serial,
        wrist_serial=args.wrist_cam_serial,
        table_exposure=args.table_exposure,
        table_gain=args.table_gain,
        wrist_exposure=args.wrist_exposure,
        wrist_gain=args.wrist_gain,
    )
    robot = RobotRuntime(args)
    pygame = screen = font = clock = None
    print(f"[INFO] execute={args.execute}; base={args.ckpt}; stage2={args.stage2_checkpoint}")
    print(
        f"[GATE] on={correction_policy.gate_threshold:.3f} "
        f"off={correction_policy.gate_off_threshold:.3f} "
        f"temporal={correction_policy.config.temporal_context} "
        f"correction_frames={tuple(args.correction_source_frame_hw)}->{tuple(args.correction_frame_hw)}"
    )

    try:
        cameras.start()
        robot.connect()
        pygame, screen, font, clock = _init_ui(args)
        from mini_lawam.rollout_ur7e import (
            format_duration_hms,
            format_rollout_stem,
            next_trace_trial_number,
        )

        trace_trial = next_trace_trial_number(args.trace_dir)
        if args.trace_dir:
            print(
                f"[TRACE] next persistent trial number={trace_trial} "
                f"(scanned {Path(args.trace_dir)})"
            )
        quit_all = False
        rollout_index = 0
        while rollout_index < args.num_rollouts and not quit_all:
            if not args.auto_start:
                print("[IDLE] S=start H=home Q=quit")
                while True:
                    table_idle, wrist_idle = cameras.read_pair(max_age=args.camera_max_age)
                    _draw_ui(pygame, screen, font, args, "IDLE", "gated residual rollout", table_idle, wrist_idle)
                    command = _poll_ui(pygame)
                    if command == "start":
                        break
                    if command == "home":
                        _move_home(robot, pygame)
                    if command == "quit":
                        quit_all = True
                        break
                    assert clock is not None
                    clock.tick(30)
                if quit_all:
                    break

            rollout_started_wall = time.time()
            rollout_started_perf = time.perf_counter()
            rollout_stem = format_rollout_stem(trace_trial, rollout_started_wall)
            first_table_frame = None
            frames_dir = None
            if args.save_frames > 0 and args.trace_dir:
                frames_dir = Path(args.trace_dir) / f"frames_{rollout_stem}"
                frames_dir.mkdir(parents=True, exist_ok=True)
                print(f"[FRAMES] saving every {args.save_frames} steps -> {frames_dir}")

            time.sleep(args.startup_wait_sec)
            base_policy.reset()
            correction_policy.reset()
            command_pose = robot.actual_pose()
            initial_tcp = np.asarray(command_pose, dtype=np.float64).copy()
            robot.start_servo()
            robot.queue_target(command_pose)
            print(
                f"[ROLLOUT] started index={rollout_index} trace_trial={trace_trial} "
                f"from TCP={np.round(command_pose, 4)}"
            )
            stop_command: Optional[str] = None
            previous_gate = False
            steps_executed = 0
            gate_active_steps = 0
            period = 1.0 / float(args.control_hz)

            for step in range(args.max_steps):
                started = time.monotonic()
                command = _poll_ui(pygame)
                if command in ("end", "home", "quit"):
                    stop_command = command
                    break

                table_rgb, wrist_rgb = cameras.read_pair(max_age=args.camera_max_age)
                if first_table_frame is None:
                    first_table_frame = np.asarray(table_rgb).copy()
                if frames_dir is not None and step % args.save_frames == 0:
                    from PIL import Image

                    Image.fromarray(table_rgb).save(
                        frames_dir / f"{step:05d}_table.jpg",
                        quality=92,
                    )
                robot_observation = robot.observation()
                actual_pose = robot_observation["tcp_pose"]
                observed_gripper = float(robot.gripper_state)
                base = base_policy.predict(
                    table_rgb=table_rgb,
                    wrist_rgb=wrist_rgb,
                    actual_pose=actual_pose,
                    command_pose=command_pose,
                )
                correction = correction_policy.predict(
                    table_rgb=correction_images(table_rgb),
                    wrist_rgb=correction_images(wrist_rgb),
                    robot_observation=robot_observation,
                    gripper_state=observed_gripper,
                    base_action=base.action,
                )

                composed_action = compose_gated_action(base.action, correction, enable_rz=args.enable_rz)
                if correction.active:
                    requested_target = target_from_action(command_pose, composed_action)
                    executed_target = clamp_target_pose(
                        requested_target,
                        actual_pose,
                        args.ws_min,
                        args.ws_max,
                        args.max_target_lead,
                    )
                    selected_gripper = correction.gripper_state
                    base_policy.commit_manual_target(executed_target)
                    executed_action = action_from_target(command_pose, executed_target, selected_gripper)
                else:
                    executed_target = base.target_pose
                    selected_gripper = base.gripper_state
                    executed_action = base.action.copy()

                robot.queue_target(executed_target)
                robot.command_gripper(selected_gripper)
                command_pose = np.asarray(executed_target, dtype=np.float64).copy()
                steps_executed += 1
                gate_active_steps += int(correction.active)

                if correction.active != previous_gate:
                    print(
                        f"[GATE] {'CORRECTION ON' if correction.active else 'BASE ONLY'} "
                        f"p={correction.gate_probability:.3f}"
                    )
                    previous_gate = correction.active
                if step % args.log_every == 0:
                    print(
                        f"[STEP {step:05d}] gate={int(correction.active)} "
                        f"p={correction.gate_probability:.3f} "
                        f"res={np.array2string(correction.arm_residual, precision=4, suppress_small=True)} "
                        f"exec={np.array2string(executed_action, precision=4, suppress_small=True)} "
                        f"grip={'close' if selected_gripper > 0 else 'open'}"
                    )
                status = (
                    f"step={step} gate={'ON' if correction.active else 'OFF'} "
                    f"p={correction.gate_probability:.3f}"
                )
                _draw_ui(
                    pygame,
                    screen,
                    font,
                    args,
                    "ROLLOUT",
                    status,
                    table_rgb,
                    wrist_rgb,
                    base.subgoal_overlay,
                )

                remaining = period - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
                elif step % args.log_every == 0:
                    print(f"[TIMING] step exceeded {period * 1000.0:.1f} ms control budget")

            robot.stop_servo()
            result = {
                "end": "ended",
                "home": "go_home",
                "quit": "quit",
            }.get(stop_command, "max_steps")
            rollout_finished_wall = time.time()
            duration_sec = round(time.perf_counter() - rollout_started_perf, 3)
            print(
                f"[ROLLOUT] ended index={rollout_index} reason={stop_command or 'max_steps'} "
                f"duration={duration_sec:.3f}s"
            )
            # Complete safety-critical motion before any nonessential trace I/O.
            if stop_command == "home" or args.home_after_rollout:
                _move_home(robot, pygame)
            if args.trace_dir:
                from PIL import Image

                trace_dir = Path(args.trace_dir)
                trace_dir.mkdir(parents=True, exist_ok=True)
                first_table_path = None
                if first_table_frame is not None:
                    table_path = trace_dir / f"{rollout_stem}_table_cam_first.png"
                    Image.fromarray(first_table_frame).save(table_path)
                    first_table_path = table_path.as_posix()
                    print(f"[FIRST FRAME] {table_path}")
                summary = {
                    "trial": trace_trial,
                    "ckpt": str(Path(args.ckpt).expanduser().resolve()),
                    "stage2_checkpoint": str(
                        Path(args.stage2_checkpoint).expanduser().resolve()
                    ),
                    "execute": bool(args.execute),
                    "result": result,
                    "started_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S%z",
                        time.localtime(rollout_started_wall),
                    ),
                    "finished_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S%z",
                        time.localtime(rollout_finished_wall),
                    ),
                    "duration_sec": duration_sec,
                    "duration_hms": format_duration_hms(duration_sec),
                    "steps_executed": steps_executed,
                    "gate_active_steps": gate_active_steps,
                    "gate_active_fraction": (
                        gate_active_steps / steps_executed if steps_executed else 0.0
                    ),
                    "gate_on_threshold": correction_policy.gate_threshold,
                    "gate_off_threshold": correction_policy.gate_off_threshold,
                    "temporal_context": correction_policy.config.temporal_context,
                    "initial_tcp": initial_tcp.tolist(),
                    "final_command_tcp": np.asarray(command_pose, dtype=float).tolist(),
                    "first_table_frame": first_table_path,
                    "periodic_frames_dir": (
                        frames_dir.as_posix() if frames_dir is not None else None
                    ),
                }
                summary_path = trace_dir / f"{rollout_stem}_summary.json"
                summary_path.write_text(
                    json.dumps(summary, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"[SUMMARY] {summary_path}")
            if stop_command == "quit":
                quit_all = True
            rollout_index += 1
            trace_trial += 1
    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    finally:
        try:
            robot.close()
        finally:
            cameras.close()
            if pygame is not None:
                try:
                    pygame.quit()
                except Exception:
                    pass
        print("[INFO] gated rollout closed")


if __name__ == "__main__":
    main()
