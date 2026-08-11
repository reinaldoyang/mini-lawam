"""Meta Quest stream parsing and side-grip takeover control.

The wire format matches ``QuestUR7eTeleop`` so the existing Quest APK can be
used unchanged::

    px,py,pz,qx,qy,qz,qw,trigger,grip,a,b,x,y\n

``grip`` is the side trigger / clutch.  While it is held, :class:`VRClutch`
owns the robot target.  The front ``trigger`` toggles the gripper only while
the clutch is active.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_MAPPING_MATRIX = np.asarray(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

QUEST_CONTROLS = (
    ("button_a", "start"),
    ("button_b", "home"),
    ("button_x", "discard"),
    ("button_y", "save"),
)


class QuestConnectionLost(ConnectionError):
    pass


class MalformedQuestMessage(ValueError):
    pass


def _binary(value: float, name: str) -> bool:
    if value not in (0.0, 1.0):
        raise MalformedQuestMessage(f"{name} must be 0 or 1")
    return bool(value)


@dataclass(frozen=True)
class QuestPose:
    position: np.ndarray
    quaternion: np.ndarray
    trigger: bool
    grip: bool
    button_a: bool = False
    button_b: bool = False
    button_x: bool = False
    button_y: bool = False

    @classmethod
    def from_csv(cls, line: str) -> "QuestPose":
        fields = line.strip().split(",")
        if len(fields) not in (9, 13):
            raise MalformedQuestMessage(f"expected 9 or 13 CSV fields, received {len(fields)}")
        try:
            values = [float(value) for value in fields]
        except ValueError as exc:
            raise MalformedQuestMessage("all Quest fields must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise MalformedQuestMessage("Quest fields cannot contain NaN or infinity")
        quaternion = np.asarray(values[3:7], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-6 or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=0.05):
            raise MalformedQuestMessage(f"Quest quaternion norm must be near one, got {norm:.5f}")
        buttons = [False] * 4
        if len(values) == 13:
            buttons = [_binary(value, name) for value, name in zip(values[9:13], "ABXY")]
        return cls(
            position=np.asarray(values[:3], dtype=np.float64),
            quaternion=quaternion / norm,
            trigger=_binary(values[7], "trigger"),
            grip=_binary(values[8], "grip"),
            button_a=buttons[0],
            button_b=buttons[1],
            button_x=buttons[2],
            button_y=buttons[3],
        )

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [*self.position, *self.quaternion, float(self.trigger), float(self.grip)],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class StampedQuestPose:
    pose: QuestPose
    received_at: float
    sequence: int


class QuestReceiver:
    """Background newline-delimited TCP receiver with edge preservation."""

    def __init__(self, host: str, port: int, *, socket_timeout: float = 0.1, max_line_bytes: int = 1024):
        self.host = str(host)
        self.port = int(port)
        self.socket_timeout = float(socket_timeout)
        self.max_line_bytes = int(max_line_bytes)
        self._condition = threading.Condition()
        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._buffer = bytearray()
        self._samples: deque[StampedQuestPose] = deque(maxlen=1024)
        self._controls: deque[str] = deque()
        self._latest: Optional[StampedQuestPose] = None
        self._previous_buttons: Optional[dict[str, bool]] = None
        self._error: Optional[BaseException] = None
        self._sequence = 0

    def start(self, connect_timeout: float = 5.0) -> None:
        if self._running:
            raise RuntimeError("Quest receiver is already running")
        print(f"[QUEST] connecting to {self.host}:{self.port}")
        sock = socket.create_connection((self.host, self.port), timeout=float(connect_timeout))
        sock.settimeout(self.socket_timeout)
        self._socket = sock
        self._running = True
        self._thread = threading.Thread(target=self._run, name="quest-receiver", daemon=True)
        self._thread.start()
        print("[QUEST] connected")

    def _run(self) -> None:
        try:
            while self._running:
                assert self._socket is not None
                try:
                    packet = self._socket.recv(4096)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._running:
                        raise QuestConnectionLost(str(exc)) from exc
                    return
                if not packet:
                    raise QuestConnectionLost("Quest closed the TCP connection")
                self._buffer.extend(packet)
                if len(self._buffer) > self.max_line_bytes and b"\n" not in self._buffer:
                    raise MalformedQuestMessage("Quest message exceeds line limit")
                while b"\n" in self._buffer:
                    raw, _, remainder = self._buffer.partition(b"\n")
                    self._buffer = bytearray(remainder)
                    if raw.endswith(b"\r"):
                        raw = raw[:-1]
                    if len(raw) > self.max_line_bytes:
                        raise MalformedQuestMessage("Quest message exceeds line limit")
                    try:
                        pose = QuestPose.from_csv(raw.decode("utf-8", errors="strict"))
                    except UnicodeDecodeError as exc:
                        raise MalformedQuestMessage("Quest message is not valid UTF-8") from exc
                    self._publish(pose)
        except BaseException as exc:
            with self._condition:
                if self._running:
                    self._error = exc
                self._condition.notify_all()

    def _publish(self, pose: QuestPose) -> None:
        now = time.monotonic()
        with self._condition:
            self._sequence += 1
            sample = StampedQuestPose(pose=pose, received_at=now, sequence=self._sequence)
            self._latest = sample
            self._samples.append(sample)
            current = {field: bool(getattr(pose, field)) for field, _ in QUEST_CONTROLS}
            if self._previous_buttons is not None:
                for field, command in QUEST_CONTROLS:
                    if current[field] and not self._previous_buttons[field]:
                        self._controls.append(command)
            self._previous_buttons = current
            self._condition.notify_all()

    def wait_for_first_pose(self, timeout: float = 5.0) -> StampedQuestPose:
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while self._latest is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for the first Quest pose")
                self._condition.wait(remaining)
            self._raise_if_error_locked()
            assert self._latest is not None
            return self._latest

    def latest(self) -> Optional[StampedQuestPose]:
        with self._condition:
            self._raise_if_error_locked()
            return self._latest

    def drain_samples(self) -> list[StampedQuestPose]:
        with self._condition:
            self._raise_if_error_locked()
            samples = list(self._samples)
            self._samples.clear()
            return samples

    def drain_controls(self) -> list[str]:
        with self._condition:
            controls = list(self._controls)
            self._controls.clear()
            return controls

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Quest receiver failed: {self._error}") from self._error

    def close(self) -> None:
        self._running = False
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None


def _wrapped_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _robot_z_yaw(rotvec: Sequence[float], locked_rotvec: Sequence[float]) -> float:
    actual = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64))
    locked = Rotation.from_rotvec(np.asarray(locked_rotvec, dtype=np.float64))
    relative = (actual * locked.inv()).as_matrix()
    return math.atan2(float(relative[1, 0]), float(relative[0, 0]))


def _quest_yaw(anchor_xyzw: Sequence[float], current_xyzw: Sequence[float]) -> float:
    anchor = Rotation.from_quat(np.asarray(anchor_xyzw, dtype=np.float64))
    current = Rotation.from_quat(np.asarray(current_xyzw, dtype=np.float64))
    quat = (current * anchor.inv()).as_quat()
    if quat[3] < 0.0:
        quat = -quat
    twist_norm = math.hypot(float(quat[1]), float(quat[3]))
    if twist_norm < 1e-12:
        return 0.0
    return _wrapped_angle(2.0 * math.atan2(float(quat[1]) / twist_norm, float(quat[3]) / twist_norm))


@dataclass(frozen=True)
class VRControlOutput:
    active: bool
    started: bool
    released: bool
    target_pose: Optional[np.ndarray]
    gripper_state: float
    gripper_toggled: bool


class VRClutch:
    """Anchor-relative Quest motion controlled by the side-grip deadman switch."""

    def __init__(
        self,
        *,
        mapping_matrix: Sequence[float] = DEFAULT_MAPPING_MATRIX,
        position_scale: float = 1.2,
        rz_scale: float = -1.0,
        max_linear_speed: float = 0.2,
        max_angular_speed: float = 0.5,
        locked_rotvec: Sequence[float] = (0.0036, 3.14094, -0.00024),
        enable_rz: bool = False,
    ) -> None:
        matrix = np.asarray(mapping_matrix, dtype=np.float64)
        if matrix.size != 9:
            raise ValueError("mapping_matrix must contain nine values")
        self.mapping_matrix = matrix.reshape(3, 3)
        self.position_scale = float(position_scale)
        self.rz_scale = float(rz_scale)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.locked_rotvec = np.asarray(locked_rotvec, dtype=np.float64).reshape(3)
        self.enable_rz = bool(enable_rz)
        self.reset(require_release=False)

    def reset(self, *, require_release: bool) -> None:
        self.active = False
        self.previous_grip = bool(require_release)
        self.previous_trigger = False
        self.controller_anchor_position: Optional[np.ndarray] = None
        self.controller_anchor_quaternion: Optional[np.ndarray] = None
        self.robot_anchor_pose: Optional[np.ndarray] = None
        self.robot_anchor_yaw = 0.0
        self.last_target_pose: Optional[np.ndarray] = None
        self.last_yaw = 0.0
        self.last_update_time: Optional[float] = None
        self.desired_gripper_state = -1.0
        self._zero_target_pending = False

    def update(
        self,
        pose: QuestPose,
        received_at: float,
        *,
        actual_pose: Sequence[float],
        current_gripper_state: float,
    ) -> VRControlOutput:
        actual = np.asarray(actual_pose, dtype=np.float64)
        if actual.shape != (6,) or not np.all(np.isfinite(actual)):
            raise ValueError("actual_pose must contain six finite values")
        rising = pose.grip and not self.previous_grip
        falling = not pose.grip and self.previous_grip
        trigger_rising = pose.trigger and not self.previous_trigger
        started = released = toggled = False

        if rising:
            self.active = True
            started = True
            self.controller_anchor_position = pose.position.copy()
            self.controller_anchor_quaternion = pose.quaternion.copy()
            self.robot_anchor_pose = actual.copy()
            self.robot_anchor_yaw = _robot_z_yaw(actual[3:6], self.locked_rotvec)
            self.last_target_pose = actual.copy()
            self.last_yaw = self.robot_anchor_yaw
            self.last_update_time = float(received_at)
            self.desired_gripper_state = 1.0 if float(current_gripper_state) > 0.0 else -1.0
            self._zero_target_pending = True

        if falling:
            self.active = False
            released = True
            self.controller_anchor_position = None
            self.controller_anchor_quaternion = None
            self.robot_anchor_pose = None
            self.last_target_pose = None
            self.last_update_time = None
            self._zero_target_pending = False

        if self.active and trigger_rising:
            self.desired_gripper_state *= -1.0
            toggled = True

        target: Optional[np.ndarray] = None
        if self.active:
            assert self.controller_anchor_position is not None
            assert self.controller_anchor_quaternion is not None
            assert self.robot_anchor_pose is not None
            assert self.last_target_pose is not None
            assert self.last_update_time is not None
            target = self.robot_anchor_pose.copy()
            mapped = self.mapping_matrix @ (pose.position - self.controller_anchor_position)
            target[:3] += mapped * self.position_scale
            desired_yaw = self.robot_anchor_yaw
            if self.enable_rz:
                desired_yaw += self.rz_scale * _quest_yaw(
                    self.controller_anchor_quaternion,
                    pose.quaternion,
                )
            elapsed = max(0.0, float(received_at) - self.last_update_time)
            linear_delta = target[:3] - self.last_target_pose[:3]
            linear_distance = float(np.linalg.norm(linear_delta))
            max_distance = self.max_linear_speed * elapsed
            if linear_distance > max_distance and linear_distance > 1e-12:
                target[:3] = self.last_target_pose[:3] + linear_delta * (max_distance / linear_distance)
            yaw_delta = _wrapped_angle(desired_yaw - self.last_yaw)
            max_angle = self.max_angular_speed * elapsed
            command_yaw = self.last_yaw + float(np.clip(yaw_delta, -max_angle, max_angle))
            target[3:6] = (
                Rotation.from_rotvec([0.0, 0.0, command_yaw])
                * Rotation.from_rotvec(self.locked_rotvec)
            ).as_rotvec()
            if self._zero_target_pending:
                target = self.robot_anchor_pose.copy()
                command_yaw = self.robot_anchor_yaw
                self._zero_target_pending = False
            self.last_target_pose = target.copy()
            self.last_yaw = command_yaw
            self.last_update_time = float(received_at)

        self.previous_grip = bool(pose.grip)
        self.previous_trigger = bool(pose.trigger)
        return VRControlOutput(
            active=self.active,
            started=started,
            released=released,
            target_pose=target,
            gripper_state=self.desired_gripper_state,
            gripper_toggled=toggled,
        )

    def commit_target(self, target_pose: Sequence[float]) -> None:
        """Keep the clutch limiter aligned with the safety-clamped target."""
        if not self.active:
            return
        target = np.asarray(target_pose, dtype=np.float64).reshape(6)
        self.last_target_pose = target.copy()
        self.last_yaw = _robot_z_yaw(target[3:6], self.locked_rotvec)
