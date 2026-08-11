"""Camera, RTDE, servo-thread, and gripper plumbing for the HIL collector.

Heavy hardware imports are intentionally deferred until the corresponding
objects are started, so ``python -m hil.collect_corrections --help`` and the
unit tests work on machines without the robot stack.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .actions import quat_wxyz_from_rotvec


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
ROBOTIQ_SOCKET_PORT = 63352


class RealSenseReader:
    """Background RGB reader for one RealSense camera."""

    def __init__(self, serial: str, name: str, *, exposure=None, gain=None) -> None:
        self.serial = str(serial)
        self.name = str(name)
        self.exposure = exposure
        self.gain = gain
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_time: Optional[float] = None
        self._error: Optional[BaseException] = None
        self._running = False
        self._pipeline = None
        self._thread: Optional[threading.Thread] = None

    def _configure(self, rs, profile) -> None:
        color_sensor = None
        for sensor in profile.get_device().query_sensors():
            name = sensor.get_info(rs.camera_info.name)
            if "RGB" in name or "Color" in name:
                color_sensor = sensor
                break
        if color_sensor is None:
            raise RuntimeError(f"{self.name} {self.serial} has no RGB sensor")
        if self.exposure is None:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            print(f"[RS] {self.name}: auto exposure")
        else:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, float(self.exposure))
            if self.gain is not None:
                color_sensor.set_option(rs.option.gain, float(self.gain))
            print(f"[RS] {self.name}: exposure={self.exposure} gain={self.gain}")

    def start(self) -> None:
        if not self.serial:
            raise ValueError(f"camera serial is required for {self.name}")
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            rs.format.rgb8,
            CAMERA_FPS,
        )
        print(f"[RS] connecting {self.name} serial={self.serial}")
        profile = pipeline.start(config)
        self._pipeline = pipeline
        self._configure(rs, profile)
        for _ in range(10):
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if color:
                with self._lock:
                    self._frame = np.asanyarray(color.get_data()).copy()
                    self._frame_time = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-reader", daemon=True)
        self._thread.start()
        print(f"[RS] ready {self.name} {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}")

    def _loop(self) -> None:
        try:
            while self._running:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                with self._lock:
                    self._frame = np.asanyarray(color.get_data()).copy()
                    self._frame_time = time.monotonic()
        except BaseException as exc:
            with self._lock:
                if self._running:
                    self._error = exc

    def latest(self, *, max_age: float = 0.5) -> np.ndarray:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"{self.name} acquisition failed: {self._error}") from self._error
            if self._frame is None or self._frame_time is None:
                raise RuntimeError(f"no frame available from {self.name}")
            age = time.monotonic() - self._frame_time
            if age > float(max_age):
                raise RuntimeError(f"{self.name} frame is stale ({age:.3f}s)")
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None


class DualRealSense:
    def __init__(
        self,
        *,
        table_serial: str,
        wrist_serial: str,
        table_exposure=None,
        table_gain=None,
        wrist_exposure=None,
        wrist_gain=None,
    ) -> None:
        self.table = RealSenseReader(
            table_serial,
            "table_cam",
            exposure=table_exposure,
            gain=table_gain,
        )
        self.wrist = RealSenseReader(
            wrist_serial,
            "wrist_cam",
            exposure=wrist_exposure,
            gain=wrist_gain,
        )

    def start(self) -> None:
        self.table.start()
        try:
            self.wrist.start()
        except BaseException:
            self.table.stop()
            raise

    def read_pair(self, *, max_age: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        return self.table.latest(max_age=max_age), self.wrist.latest(max_age=max_age)

    def close(self) -> None:
        self.wrist.stop()
        self.table.stop()


@dataclass
class SharedServoTarget:
    lock: threading.Lock = field(default_factory=threading.Lock)
    target: Optional[np.ndarray] = None
    running: bool = True
    error: Optional[BaseException] = None


def _clamp_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= float(maximum) or norm < 1e-12:
        return value
    return value * (float(maximum) / norm)


class ServoWorker:
    """Interpolate shared TCP targets and issue servoL at the robot cadence."""

    def __init__(
        self,
        rtde_c,
        initial_pose: Sequence[float],
        *,
        hz: float,
        speed: float,
        acceleration: float,
        lookahead: float,
        gain: float,
        interpolation_alpha: float,
        max_position_step: float,
        max_rotation_step: float,
    ) -> None:
        self.rtde_c = rtde_c
        self.hz = float(hz)
        self.speed = float(speed)
        self.acceleration = float(acceleration)
        self.lookahead = float(lookahead)
        self.gain = float(gain)
        self.alpha = float(interpolation_alpha)
        self.max_position_step = float(max_position_step)
        self.max_rotation_step = float(max_rotation_step)
        self.state = SharedServoTarget(target=np.asarray(initial_pose, dtype=np.float64).copy())
        self._thread = threading.Thread(target=self._run, name="hil-servo", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, target_pose: Sequence[float]) -> None:
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError(f"servo target must contain six finite values, got {target!r}")
        self.check_health()
        with self.state.lock:
            self.state.target = target.copy()

    def check_health(self) -> None:
        with self.state.lock:
            error = self.state.error
        if error is not None:
            raise RuntimeError(f"servo worker failed: {error}") from error

    def _run(self) -> None:
        period = 1.0 / self.hz
        next_time = time.perf_counter()
        commanded: Optional[np.ndarray] = None
        try:
            while True:
                with self.state.lock:
                    running = self.state.running
                    target = None if self.state.target is None else self.state.target.copy()
                if not running:
                    return
                if target is not None:
                    if commanded is None:
                        commanded = target.copy()
                    else:
                        position_step = _clamp_norm((target[:3] - commanded[:3]) * self.alpha, self.max_position_step)
                        rotation_step = _clamp_norm((target[3:6] - commanded[3:6]) * self.alpha, self.max_rotation_step)
                        commanded = commanded.copy()
                        commanded[:3] += position_step
                        commanded[3:6] += rotation_step
                    accepted = self.rtde_c.servoL(
                        commanded.tolist(),
                        self.speed,
                        self.acceleration,
                        period,
                        self.lookahead,
                        self.gain,
                    )
                    if accepted is False:
                        raise RuntimeError("RTDE servoL rejected the command")
                next_time += period
                remaining = next_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                else:
                    next_time = time.perf_counter()
        except BaseException as exc:
            with self.state.lock:
                self.state.error = exc
        finally:
            try:
                self.rtde_c.servoStop()
            except Exception:
                pass

    def stop(self) -> None:
        with self.state.lock:
            self.state.running = False
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("servo worker did not stop within two seconds")
        self.check_health()


class LatchedGripper:
    def __init__(self, driver_cls, robot_ip: str, *, open_mm: float, close_mm: float, speed: float, force: float):
        self.open_mm = float(open_mm)
        self.close_mm = float(close_mm)
        self._driver = driver_cls()
        self._driver.connect(str(robot_ip), ROBOTIQ_SOCKET_PORT)
        self._state: Optional[float] = None
        self._speed = int(np.clip(round(float(speed) * 255.0 / 100.0), 0, 255))
        self._force = int(np.clip(round(float(force) * 255.0 / 100.0), 0, 255))
        if hasattr(self._driver, "activate"):
            self._driver.activate(auto_calibrate=False)

    def _width_raw(self, width_mm: float) -> int:
        return int(np.clip(round(255.0 * (1.0 - width_mm / max(self.open_mm, 1e-6))), 0, 255))

    def command(self, state: float) -> None:
        exact = 1.0 if float(state) > 0.0 else -1.0
        if exact == self._state:
            return
        width = self.close_mm if exact > 0.0 else self.open_mm
        self._driver.move(self._width_raw(width), self._speed, self._force)
        self._state = exact

    def close(self) -> None:
        try:
            self._driver.disconnect()
        except Exception:
            pass


class RobotRuntime:
    """Real or synthetic UR state with episode-scoped servo arming."""

    def __init__(self, args) -> None:
        self.args = args
        self.execute = bool(args.execute)
        self.rtde_r = None
        self.rtde_c = None
        self.servo: Optional[ServoWorker] = None
        self.gripper: Optional[LatchedGripper] = None
        self.gripper_state = -1.0
        self._dry_pose = np.concatenate(
            (
                (np.asarray(args.ws_min, dtype=np.float64) + np.asarray(args.ws_max, dtype=np.float64)) / 2.0,
                np.asarray(args.locked_rotvec, dtype=np.float64),
            )
        )

    def connect(self) -> None:
        if not self.execute:
            print(f"[ROBOT] dry-run synthetic TCP={np.round(self._dry_pose, 4)}")
            return
        import rtde_control
        import rtde_receive

        print(f"[RTDE] connecting to {self.args.robot_ip}")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.args.robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.args.robot_ip)
        self.actual_pose()
        if self.args.use_gripper_control:
            realworld_dir = str(Path(self.args.realworld_dir).resolve())
            if realworld_dir not in sys.path:
                sys.path.insert(0, realworld_dir)
            from rtde_gripper.robotiq_gripper import RobotiqGripper

            self.gripper = LatchedGripper(
                RobotiqGripper,
                self.args.robot_ip,
                open_mm=self.args.gripper_open_mm,
                close_mm=self.args.gripper_close_mm,
                speed=self.args.gripper_speed,
                force=self.args.gripper_force,
            )
            self.command_gripper(-1.0)

    def actual_pose(self) -> np.ndarray:
        if not self.execute:
            return self._dry_pose.copy()
        is_connected = getattr(self.rtde_r, "isConnected", None)
        if callable(is_connected) and not is_connected():
            raise RuntimeError("RTDE receive connection is down")
        pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"invalid actual TCP pose: {pose!r}")
        return pose

    def joint_positions(self) -> np.ndarray:
        if not self.execute:
            return np.zeros(6, dtype=np.float32)
        joints = np.asarray(self.rtde_r.getActualQ(), dtype=np.float64)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"invalid actual joint positions: {joints!r}")
        return joints.astype(np.float32)

    def observation(self) -> dict[str, np.ndarray]:
        pose = self.actual_pose()
        return {
            "tcp_pose": pose,
            "eef_pos_base": pose[:3].astype(np.float32),
            "eef_quat_base": quat_wxyz_from_rotvec(pose[3:6]),
            "joint_pos": self.joint_positions(),
        }

    def start_servo(self) -> None:
        if self.servo is not None:
            raise RuntimeError("servo is already armed")
        if not self.execute:
            return
        initial = self.actual_pose()
        self.servo = ServoWorker(
            self.rtde_c,
            initial,
            hz=self.args.servo_hz,
            speed=self.args.servol_speed,
            acceleration=self.args.servol_acc,
            lookahead=self.args.servol_lookahead,
            gain=self.args.servol_gain,
            interpolation_alpha=self.args.servol_interp_alpha,
            max_position_step=self.args.servol_max_pos_step,
            max_rotation_step=self.args.servol_max_rot_step,
        )
        self.servo.start()
        print("[SERVO] armed")

    def queue_target(self, target_pose: Sequence[float]) -> None:
        target = np.asarray(target_pose, dtype=np.float64).reshape(6)
        if not self.execute:
            self._dry_pose = target.copy()
            return
        if self.servo is None:
            raise RuntimeError("servo is not armed")
        self.servo.update(target)

    def stop_servo(self) -> None:
        if self.servo is None:
            return
        worker = self.servo
        self.servo = None
        worker.stop()
        print("[SERVO] disarmed")

    def command_gripper(self, value: float) -> None:
        state = 1.0 if float(value) > float(self.args.gripper_threshold) else -1.0
        if self.gripper is not None:
            self.gripper.command(state)
        self.gripper_state = state

    def move_home(self) -> None:
        self.stop_servo()
        self.command_gripper(-1.0)
        if not self.execute:
            return
        accepted = self.rtde_c.moveJ(
            np.asarray(self.args.home_q, dtype=np.float64).tolist(),
            float(self.args.home_movej_speed),
            float(self.args.home_movej_acc),
        )
        if accepted is False:
            raise RuntimeError("RTDE moveJ home was rejected")
        print("[HOME] complete")

    def close(self) -> None:
        try:
            self.stop_servo()
        finally:
            if self.gripper is not None:
                self.gripper.close()
                self.gripper = None
            if self.rtde_c is not None:
                try:
                    self.rtde_c.stopScript()
                except Exception:
                    pass
