#!/usr/bin/env python3
"""Collect VR human corrections on top of an autonomous Mini-LaWAM rollout.

The Meta Quest side grip is a momentary ownership switch:

* released: Mini-LaWAM controls the arm and gripper;
* held: Quest motion and the front-trigger gripper state fully replace policy
  commands;
* released again: Mini-LaWAM resumes from the last human command target.

Run this module from the LaWAM repository root.  Robot commands are disabled
unless the literal ``--execute`` flag is supplied.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .actions import action_from_target, clamp_target_pose, gripper_state
from .constants import GRIPPER_CLOSE, GRIPPER_OPEN
from .data import CorrectionEpisode, CorrectionHDF5Writer
from .policy import MiniLaWAMBasePolicy
from .robot import DualRealSense, RobotRuntime
from .vr import DEFAULT_MAPPING_MATRIX, QuestReceiver, VRClutch, VRControlOutput


DEFAULT_CHECKPOINT = (
    "results/mini_lawam/checkpoint/vr_controller/"
    "ckpt_new_vr_teleop_egg_rz_103ep_256_attn_rz_binary_grip_t1.pt"
)
DEFAULT_REALWORLD_DIR = "/home/iclu200/reinaldoyang/ur7e_ramen_il/scripts/real_world"
DEFAULT_HOME_Q = [0.4076, -1.4255, -1.7052, -1.5821, 1.5703, 1.9768]
DEFAULT_LOCKED_ROTVEC = [0.0036, 3.14094, -0.00024]
DEFAULT_WS_MIN = [-0.165, -0.164, 0.158]
DEFAULT_WS_MAX = [0.54, 0.63, 0.518]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    model = parser.add_argument_group("Mini-LaWAM")
    model.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    model.add_argument("--device", default="cuda")
    model.add_argument(
        "--train-frame-hw",
        type=int,
        nargs=2,
        default=[240, 320],
        metavar=("H", "W"),
        help="Original recording resolution before Mini-LaWAM's 256x256 resize; use 0 0 to disable.",
    )
    model.add_argument("--action-scale", type=float, default=1.0, help="Deployment gain for joystick XYZ and RZ.")
    model.add_argument("--delta-scale", type=float, default=1.0, help="Deployment gain for a delta-target checkpoint.")
    model.add_argument("--enable-rz", action="store_true", help="Allow policy and Quest base-Z rotation.")
    model.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="Match rollout_ur7e_vr by averaging all overlapping chunk predictions.",
    )
    model.add_argument(
        "--te-m",
        type=float,
        default=0.1,
        help="Temporal-ensemble age decay; newest prediction has weight one.",
    )
    model.add_argument("--target-ema", type=float, default=1.0, help="Weight on the newest XYZ target.")
    model.add_argument("--target-deadband", type=float, default=0.0, help="Ignore smaller XYZ target changes.")
    model.add_argument(
        "--gripper-open-lead-steps",
        type=int,
        default=1,
        help="Open-only gripper lookahead; release remains latched for the episode.",
    )

    quest = parser.add_argument_group("Meta Quest")
    quest.add_argument("--quest-host", default="127.0.0.1")
    quest.add_argument("--quest-port", type=int, default=5555)
    quest.add_argument("--quest-connect-timeout", type=float, default=5.0)
    quest.add_argument("--quest-watchdog", type=float, default=0.2)
    quest.add_argument(
        "--vr-position-scale",
        "--position-scale",
        dest="vr_position_scale",
        type=float,
        default=1.2,
        help="Robot metres per mapped Quest metre during takeover (recorder default: 1.2).",
    )
    quest.add_argument(
        "--vr-rz-scale",
        "--rz-scale",
        dest="vr_rz_scale",
        type=float,
        default=-1.0,
        help="Robot base-Z radians per Quest yaw radian.",
    )
    quest.add_argument(
        "--vr-max-linear-speed",
        "--max-linear-speed",
        dest="vr_max_linear_speed",
        type=float,
        default=0.2,
        help="Maximum VR takeover target speed in m/s (recorder default: 0.2).",
    )
    quest.add_argument(
        "--vr-max-angular-speed",
        "--max-angular-speed",
        dest="vr_max_angular_speed",
        type=float,
        default=0.5,
        help="Maximum VR takeover angular speed in rad/s.",
    )
    quest.add_argument(
        "--vr-mapping-matrix",
        type=float,
        nargs=9,
        default=DEFAULT_MAPPING_MATRIX.reshape(-1).tolist(),
        metavar=("M00", "M01", "M02", "M10", "M11", "M12", "M20", "M21", "M22"),
    )

    camera = parser.add_argument_group("cameras and display")
    camera.add_argument("--table-cam-serial", required=True)
    camera.add_argument("--wrist-cam-serial", required=True)
    camera.add_argument("--table-exposure", type=float, default=180.0)
    camera.add_argument("--table-gain", type=float, default=16.0)
    camera.add_argument("--wrist-exposure", type=float, default=100.0)
    camera.add_argument("--wrist-gain", type=float, default=16.0)
    camera.add_argument("--camera-max-age", type=float, default=0.5)
    camera.add_argument("--image-height", type=int, default=168)
    camera.add_argument("--image-width", type=int, default=224)
    camera.add_argument("--display-scale", type=float, default=2.0)
    camera.add_argument(
        "--show-camera",
        action="store_true",
        help="Rollout-compatible flag; the HIL camera window is shown unless --no-display is used.",
    )
    camera.add_argument(
        "--show-subgoal",
        action="store_true",
        help="Show the predicted DINO feature-change overlay in the table-camera panel.",
    )
    camera.add_argument("--subgoal-alpha", type=float, default=0.55)
    camera.add_argument("--subgoal-update-steps", type=int, default=8)
    camera.add_argument("--no-display", action="store_true", help="Use Quest face buttons only; do not open pygame.")

    robot = parser.add_argument_group("UR7e")
    robot.add_argument("--execute", action="store_true", help="Actually command the robot; default is a dry run.")
    robot.add_argument("--robot-ip", default="140.96.93.7")
    robot.add_argument("--realworld-dir", default=DEFAULT_REALWORLD_DIR, help="Directory containing rtde_gripper.")
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
        help="Maximum commanded XYZ target lead from the measured TCP.",
    )

    gripper = parser.add_argument_group("Robotiq gripper")
    gripper.add_argument("--use-gripper-control", action="store_true")
    gripper.add_argument("--gripper-open-mm", type=float, default=52.0)
    gripper.add_argument("--gripper-close-mm", type=float, default=23.0)
    gripper.add_argument("--gripper-speed", type=float, default=100.0)
    gripper.add_argument("--gripper-force", type=float, default=50.0)
    gripper.add_argument("--gripper-threshold", type=float, default=0.0)

    output = parser.add_argument_group("correction dataset")
    output.add_argument("--output-dir", default="dataset/hil_mini_lawam_vr")
    output.add_argument("--output-file", default=None)
    output.add_argument("--hdf5-compression", choices=["lzf", "gzip", "none"], default="lzf")
    output.add_argument("--hdf5-write-batch-size", type=int, default=32)
    output.add_argument("--seed", type=int, default=101)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not Path(args.ckpt).is_file():
        parser.error(f"Mini-LaWAM checkpoint does not exist: {args.ckpt}")
    positive = (
        "control_hz",
        "servo_hz",
        "quest_watchdog",
        "camera_max_age",
        "max_target_lead",
        "vr_position_scale",
        "vr_max_linear_speed",
        "vr_max_angular_speed",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.image_height <= 0 or args.image_width <= 0:
        parser.error("--image-height and --image-width must be positive")
    if args.hdf5_write_batch_size <= 0:
        parser.error("--hdf5-write-batch-size must be positive")
    if np.any(np.asarray(args.ws_min) >= np.asarray(args.ws_max)):
        parser.error("every --ws-min component must be smaller than --ws-max")
    if not np.isfinite(args.vr_rz_scale):
        parser.error("--vr-rz-scale/--rz-scale must be finite")
    if not np.all(np.isfinite(args.vr_mapping_matrix)):
        parser.error("--vr-mapping-matrix must contain finite values")
    if not np.isfinite(args.action_scale) or args.action_scale < 0.0:
        parser.error("--action-scale must be finite and non-negative")
    if not np.isfinite(args.te_m) or args.te_m < 0.0:
        parser.error("--te-m must be finite and non-negative")
    if not np.isfinite(args.target_ema) or not 0.0 <= args.target_ema <= 1.0:
        parser.error("--target-ema must be in [0,1]")
    if not np.isfinite(args.target_deadband) or args.target_deadband < 0.0:
        parser.error("--target-deadband must be finite and non-negative")
    if args.gripper_open_lead_steps < 0:
        parser.error("--gripper-open-lead-steps must be non-negative")
    if not np.isfinite(args.subgoal_alpha) or not 0.0 <= args.subgoal_alpha <= 1.0:
        parser.error("--subgoal-alpha must be in [0,1]")
    if args.subgoal_update_steps < 1:
        parser.error("--subgoal-update-steps must be positive")
    if args.show_subgoal and (args.no_display or not args.show_camera):
        parser.error("--show-subgoal requires --show-camera and cannot use --no-display")


def resolve_output_path(args: argparse.Namespace) -> Path:
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if args.output_file is None:
        return directory / f"hil_corrections_{int(time.time())}.hdf5"
    requested = Path(args.output_file)
    return requested if requested.is_absolute() else directory / requested


def resize_for_storage(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    source = np.asarray(image, dtype=np.uint8)
    if source.shape[:2] != (height, width):
        source = cv2.resize(source, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(source, dtype=np.uint8)


def init_display(args):
    if args.no_display:
        return None, None, None
    import pygame

    pygame.init()
    width = max(1, int(round(args.image_width * args.display_scale)))
    height = max(1, int(round(args.image_height * args.display_scale)))
    screen = pygame.display.set_mode((width * 2, height))
    pygame.display.set_caption("Mini-LaWAM VR HIL corrections | wrist | table")
    return pygame, screen, pygame.font.SysFont(None, 26)


def poll_keyboard(pygame) -> list[str]:
    if pygame is None:
        return []
    commands: list[str] = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            commands.append("quit")
        elif event.type == pygame.KEYDOWN:
            mapping = {
                pygame.K_ESCAPE: "quit",
                pygame.K_q: "quit",
                pygame.K_z: "start",
                pygame.K_s: "start",
                pygame.K_v: "save",
                pygame.K_x: "discard",
                pygame.K_h: "home",
            }
            command = mapping.get(event.key)
            if command is not None:
                commands.append(command)
    return commands


def draw_display(pygame, screen, font, args, wrist_rgb, table_rgb, status: str) -> None:
    if pygame is None:
        return
    width = max(1, int(round(args.image_width * args.display_scale)))
    height = max(1, int(round(args.image_height * args.display_scale)))

    def surface(image):
        item = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        return pygame.transform.smoothscale(item, (width, height))

    screen.blit(surface(wrist_rgb), (0, 0))
    screen.blit(surface(table_rgb), (width, 0))
    background = pygame.Surface((width * 2, 34), pygame.SRCALPHA)
    background.fill((0, 0, 0, 180))
    screen.blit(background, (0, 0))
    screen.blit(font.render(status, True, (80, 255, 80)), (8, 7))
    pygame.display.flip()


def binary_gripper_label(value: float, threshold: float) -> int:
    """Encode the executed gripper command as OPEN=0 or CLOSE=1."""
    return GRIPPER_CLOSE if gripper_state(value, threshold) > 0.0 else GRIPPER_OPEN


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    np.random.seed(args.seed)
    output_path = resolve_output_path(args)
    train_hw = None if args.train_frame_hw[0] <= 0 else tuple(args.train_frame_hw)

    policy = MiniLaWAMBasePolicy(
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
    print(
        f"[POLICY] target={policy.target_mode} include_rz={policy.include_rz} "
        f"enable_rz={args.enable_rz} wrist={policy.use_wrist} horizon={policy.action_horizon} "
        f"temporal_ensemble={policy.temporal_ensemble} te_m={args.te_m:g}"
    )
    print(f"[INFO] execute={args.execute}; output={output_path}")

    cameras = DualRealSense(
        table_serial=args.table_cam_serial,
        wrist_serial=args.wrist_cam_serial,
        table_exposure=args.table_exposure,
        table_gain=args.table_gain,
        wrist_exposure=args.wrist_exposure,
        wrist_gain=args.wrist_gain,
    )
    robot = RobotRuntime(args)
    quest = QuestReceiver(args.quest_host, args.quest_port)
    clutch = VRClutch(
        mapping_matrix=args.vr_mapping_matrix,
        position_scale=args.vr_position_scale,
        rz_scale=args.vr_rz_scale,
        max_linear_speed=args.vr_max_linear_speed,
        max_angular_speed=args.vr_max_angular_speed,
        locked_rotvec=args.locked_rotvec,
        enable_rz=args.enable_rz,
    )
    episode = CorrectionEpisode()
    writer: Optional[CorrectionHDF5Writer] = None
    pygame = screen = font = None
    recording = False
    command_pose: Optional[np.ndarray] = None
    step_index = 0
    watchdog_latched = False
    running = True

    def start_episode() -> None:
        nonlocal recording, command_pose, step_index
        if recording:
            print("[WARN] an episode is already recording")
            return
        episode.reset()
        policy.reset()
        clutch.reset(require_release=True)
        quest.drain_samples()
        command_pose = robot.actual_pose()
        robot.start_servo()
        robot.queue_target(command_pose)
        recording = True
        step_index = 0
        print("[EPISODE] started; release then hold a side grip to take over")

    def finish_episode(*, save: bool, outcome: str, go_home: bool) -> None:
        nonlocal recording, command_pose
        robot.stop_servo()
        if save and len(episode):
            assert writer is not None
            name, count, interventions = writer.write_episode(episode, outcome=outcome)
            print(f"[EPISODE] saved {name}: steps={count} interventions={interventions} outcome={outcome}")
        elif save:
            print("[WARN] episode is empty; nothing saved")
        else:
            print(f"[EPISODE] discarded {len(episode)} buffered steps")
        episode.reset()
        recording = False
        command_pose = None
        clutch.reset(require_release=True)
        if go_home:
            robot.move_home()

    try:
        cameras.start()
        robot.connect()
        quest.start(connect_timeout=args.quest_connect_timeout)
        quest.wait_for_first_pose(timeout=args.quest_connect_timeout)
        writer = CorrectionHDF5Writer(output_path, args, policy)
        pygame, screen, font = init_display(args)

        print("\n=== Mini-LaWAM VR HIL correction collector ===")
        print("Quest: A=start, Y=save+home, X=discard+home, B=home")
        print("Keyboard: Z/S=start, V=save+home, X=discard+home, H=home, Q/Esc=quit")
        print("Hold one side grip for full VR takeover; release it for policy control.")
        print("While taking over, the front trigger toggles the gripper.")
        period = 1.0 / float(args.control_hz)

        while running:
            loop_started = time.monotonic()
            table_rgb, wrist_rgb = cameras.read_pair(max_age=args.camera_max_age)
            controls = quest.drain_controls() + poll_keyboard(pygame)
            for control in controls:
                if control == "quit":
                    running = False
                    break
                if control == "start":
                    start_episode()
                elif control == "save":
                    if recording:
                        finish_episode(save=True, outcome="saved", go_home=True)
                    else:
                        print("[WARN] no active episode to save")
                elif control == "discard":
                    if recording:
                        finish_episode(save=False, outcome="discarded", go_home=True)
                    else:
                        print("[WARN] no active episode to discard")
                elif control == "home":
                    if recording:
                        print("[WARN] home ignored while recording; save or discard first")
                    else:
                        robot.move_home()
            if not running:
                break

            latest = quest.latest()
            if latest is None:
                raise RuntimeError("Quest has not published a controller pose")
            quest_age = time.monotonic() - latest.received_at
            if quest_age > float(args.quest_watchdog):
                if recording and not watchdog_latched:
                    print(f"[SAFETY] Quest stream stale for {quest_age:.3f}s; stopping and saving partial episode")
                    finish_episode(save=bool(len(episode)), outcome="aborted_vr_watchdog", go_home=False)
                watchdog_latched = True
                status = f"QUEST STALE {quest_age:.2f}s | restart stream before recording"
                draw_display(pygame, screen, font, args, wrist_rgb, table_rgb, status)
                remaining = period - (time.monotonic() - loop_started)
                if remaining > 0.0:
                    time.sleep(remaining)
                continue
            watchdog_latched = False

            if not recording:
                quest.drain_samples()
                draw_display(
                    pygame,
                    screen,
                    font,
                    args,
                    wrist_rgb,
                    table_rgb,
                    "IDLE | A or Z=start | side grip=takeover during episode",
                )
                remaining = period - (time.monotonic() - loop_started)
                if remaining > 0.0:
                    time.sleep(remaining)
                continue

            assert command_pose is not None
            robot_obs = robot.observation()
            actual_pose = robot_obs["tcp_pose"]
            observed_gripper_state = float(robot.gripper_state)

            samples = quest.drain_samples()
            if not samples:
                samples = [latest]
            output: Optional[VRControlOutput] = None
            takeover_started = False
            takeover_released = False
            for sample in samples:
                output = clutch.update(
                    sample.pose,
                    sample.received_at,
                    actual_pose=actual_pose,
                    current_gripper_state=robot.gripper_state,
                )
                takeover_started = takeover_started or output.started
                takeover_released = takeover_released or output.released
            assert output is not None
            if takeover_started:
                # Remove any policy target lead before human ownership begins.
                command_pose = actual_pose.copy()
                # Publish the hold target before model inference so the 500 Hz
                # worker stops pursuing the previous autonomous target as soon
                # as the 20 Hz loop observes the side-grip edge.
                robot.queue_target(command_pose)

            base = policy.predict(
                table_rgb=table_rgb,
                wrist_rgb=wrist_rgb,
                actual_pose=actual_pose,
                command_pose=command_pose,
            )

            human_action = np.zeros(7, dtype=np.float32)
            human_action[6] = observed_gripper_state
            if output.active:
                if output.target_pose is None:
                    raise RuntimeError("VR clutch is active without a target pose")
                human_target = clamp_target_pose(
                    output.target_pose,
                    actual_pose,
                    args.ws_min,
                    args.ws_max,
                    args.max_target_lead,
                )
                clutch.commit_target(human_target)
                human_action = action_from_target(command_pose, human_target, output.gripper_state)
                executed_target = human_target
                executed_action = human_action.copy()
                selected_gripper = output.gripper_state
                policy.commit_manual_target(human_target)
            else:
                executed_target = base.target_pose
                executed_action = base.action.copy()
                selected_gripper = base.gripper_state

            robot.queue_target(executed_target)
            robot.command_gripper(selected_gripper)

            intervention = bool(output.active)
            residual = np.zeros(7, dtype=np.float32)
            gripper_label = binary_gripper_label(executed_action[6], args.gripper_threshold)
            if intervention:
                residual = executed_action - base.action

            table_store = resize_for_storage(table_rgb, args.image_height, args.image_width)
            wrist_store = resize_for_storage(wrist_rgb, args.image_height, args.image_width)
            episode.append(
                obs={
                    "table_cam": table_store,
                    "wrist_cam": wrist_store,
                    "eef_pos_base": robot_obs["eef_pos_base"],
                    "eef_quat_base": robot_obs["eef_quat_base"],
                    "joint_pos": robot_obs["joint_pos"],
                    "gripper_state": np.asarray([observed_gripper_state], dtype=np.float32),
                    "base_policy_action": base.action,
                    "quest_controller": latest.pose.as_array(),
                },
                base_action=base.action,
                human_action=human_action,
                executed_action=executed_action,
                residual_target=residual,
                manual_control=intervention,
                intervention=intervention,
                gripper_label=gripper_label,
                timestamp=time.time(),
            )
            command_pose = np.asarray(executed_target, dtype=np.float64).copy()

            if takeover_started:
                print("[CONTROL] VR TAKEOVER — side grip held")
            if output.gripper_toggled:
                print(f"[CONTROL] VR gripper -> {int(output.gripper_state):+d}")
            if takeover_released:
                print("[CONTROL] POLICY RESUMED — side grip released")
            if step_index % 10 == 0:
                owner = "VR" if intervention else "POLICY"
                print(
                    f"[STEP {step_index:05d}] owner={owner} grip={int(robot.gripper_state):+d} "
                    f"base={np.array2string(base.action, precision=4, suppress_small=True)} "
                    f"exec={np.array2string(executed_action, precision=4, suppress_small=True)}"
                )
            status = (
                f"{'VR TAKEOVER' if intervention else 'POLICY'} | step={step_index} "
                f"corrections={sum(episode.intervene_mask)} grip={int(robot.gripper_state):+d}"
            )
            table_display = base.subgoal_overlay if base.subgoal_overlay is not None else table_rgb
            draw_display(pygame, screen, font, args, wrist_rgb, table_display, status)
            step_index += 1

            remaining = period - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    except BaseException as exc:
        if recording:
            try:
                robot.stop_servo()
                if writer is not None and len(episode):
                    name, count, interventions = writer.write_episode(episode, outcome="aborted_runtime_error")
                    print(f"[SAFETY] saved partial {name}: steps={count} interventions={interventions}")
                    episode.reset()
                recording = False
            except BaseException as save_exc:
                print(f"[ERROR] failed to save partial episode: {save_exc}", file=sys.stderr)
        raise
    finally:
        if recording and len(episode):
            print(f"[WARN] discarding {len(episode)} unsaved steps during shutdown")
        try:
            robot.close()
        finally:
            quest.close()
            cameras.close()
            if writer is not None:
                writer.close()
            if pygame is not None:
                try:
                    pygame.quit()
                except Exception:
                    pass
        print("[INFO] HIL collector closed")


if __name__ == "__main__":
    main()
