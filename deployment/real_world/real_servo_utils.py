#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class SharedServoState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    target_tcp: np.ndarray | None = None
    target_seq: int = 0
    running: bool = False
    paused: bool = False


def apply_delta_pose_on_tcp_compose(current_tcp_pose, delta_pose):
    cur = np.asarray(current_tcp_pose, dtype=np.float64).copy()
    delta = np.asarray(delta_pose, dtype=np.float64).reshape(-1)

    if delta.shape[0] < 6:
        raise ValueError(f"delta_pose must have at least 6 dims, got {delta.shape}")

    cur_pos = cur[:3]
    cur_rotvec = cur[3:6]
    dpos = delta[:3]
    drot = delta[3:6]

    cur_R = R.from_rotvec(cur_rotvec)
    d_R = R.from_rotvec(drot)

    tgt_pos = cur_pos + dpos
    tgt_R = d_R * cur_R
    tgt_rotvec = tgt_R.as_rotvec()
    return np.concatenate([tgt_pos, tgt_rotvec], axis=0).astype(np.float64)


def compose_target_tcp_locked_orientation(current_tcp_pose, delta_xyz, locked_rotvec, rz_delta, rz_accum):
    cur = np.asarray(current_tcp_pose, dtype=np.float64).copy()
    delta_xyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    locked_rotvec = np.asarray(locked_rotvec, dtype=np.float64).reshape(3)
    rz_accum = float(rz_accum) + float(rz_delta)

    tgt = cur.copy()
    tgt[:3] = cur[:3] + delta_xyz
    locked_R = R.from_rotvec(locked_rotvec)
    rz_R = R.from_rotvec(np.array([0.0, 0.0, rz_accum], dtype=np.float64))
    tgt[3:6] = (rz_R * locked_R).as_rotvec()
    return tgt.astype(np.float64), rz_accum


def compose_target_tcp_from_action(
    *,
    current_tcp_pose,
    action,
    zero_rotation_action: bool,
    locked_rotvec=None,
    rz_accum: float = 0.0,
):
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] < 6:
        raise ValueError(f"action must have at least 6 dims, got {action.shape}")

    if zero_rotation_action:
        if locked_rotvec is None:
            locked_rotvec = np.asarray(current_tcp_pose, dtype=np.float64)[3:6].copy()
        return compose_target_tcp_locked_orientation(
            current_tcp_pose=current_tcp_pose,
            delta_xyz=action[:3],
            locked_rotvec=locked_rotvec,
            rz_delta=action[5],
            rz_accum=rz_accum,
        )

    target_tcp = apply_delta_pose_on_tcp_compose(current_tcp_pose=current_tcp_pose, delta_pose=action[:6])
    return target_tcp, float(rz_accum)


def update_shared_servo_target(shared_state: SharedServoState, target_tcp: np.ndarray):
    with shared_state.lock:
        shared_state.target_tcp = np.asarray(target_tcp, dtype=np.float64).copy()
        shared_state.target_seq += 1


def _clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= max_norm or n < 1e-12:
        return vec
    return vec * (max_norm / n)


def interpolate_tcp_toward_target(
    current_tcp: np.ndarray,
    target_tcp: np.ndarray,
    alpha: float = 0.25,
    max_pos_step: float = 0.0015,
    max_rot_step: float = 0.02,
) -> np.ndarray:
    cur = np.asarray(current_tcp, dtype=np.float64).copy()
    tgt = np.asarray(target_tcp, dtype=np.float64).copy()
    out = cur.copy()

    dpos = (tgt[:3] - cur[:3]) * float(alpha)
    dpos = _clamp_norm(dpos, max_pos_step)
    out[:3] = cur[:3] + dpos

    drot = (tgt[3:6] - cur[3:6]) * float(alpha)
    drot = _clamp_norm(drot, max_rot_step)
    out[3:6] = cur[3:6] + drot
    return out


def servo_loop_thread(
    shared_state: SharedServoState,
    rtde_c,
    control_hz=500.0,
    speed=0.25,
    acc=0.25,
    lookahead_time=0.08,
    gain=300.0,
    interp_alpha=0.25,
    max_pos_step=0.0015,
    max_rot_step=0.02,
):
    dt = 1.0 / float(control_hz)
    commanded_tcp = None
    latest_target = None

    print(f"[SERVO] started at {control_hz:.1f} Hz (dt={dt:.6f}s)")
    next_t = time.perf_counter()

    while True:
        with shared_state.lock:
            running = shared_state.running
            paused = shared_state.paused
            target = None if shared_state.target_tcp is None else shared_state.target_tcp.copy()

        if not running:
            break

        if paused:
            commanded_tcp = None
            latest_target = None
            next_t += dt
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()
            continue

        if target is not None:
            latest_target = target

        if latest_target is not None:
            if commanded_tcp is None:
                commanded_tcp = latest_target.copy()
            else:
                commanded_tcp = interpolate_tcp_toward_target(
                    current_tcp=commanded_tcp,
                    target_tcp=latest_target,
                    alpha=interp_alpha,
                    max_pos_step=max_pos_step,
                    max_rot_step=max_rot_step,
                )

            try:
                rtde_c.servoL(
                    commanded_tcp.tolist(),
                    speed,
                    acc,
                    dt,
                    lookahead_time,
                    gain,
                )
            except Exception as exc:
                print(f"[SERVO] servoL failed: {exc}")

        next_t += dt
        sleep_t = next_t - time.perf_counter()
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            next_t = time.perf_counter()

    try:
        rtde_c.servoStop()
    except Exception:
        pass

    print("[SERVO] stopped")


def start_servo_pipeline(
    *,
    rtde_r,
    rtde_c,
    control_hz,
    speed,
    acc,
    lookahead_time,
    gain,
    interp_alpha,
    max_pos_step,
    max_rot_step,
):
    servo_state = SharedServoState()
    servo_state.running = True
    current_tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
    servo_state.target_tcp = current_tcp.copy()

    servo_thread = threading.Thread(
        target=servo_loop_thread,
        kwargs=dict(
            shared_state=servo_state,
            rtde_c=rtde_c,
            control_hz=control_hz,
            speed=speed,
            acc=acc,
            lookahead_time=lookahead_time,
            gain=gain,
            interp_alpha=interp_alpha,
            max_pos_step=max_pos_step,
            max_rot_step=max_rot_step,
        ),
        daemon=True,
    )
    servo_thread.start()
    return servo_state, servo_thread


def stop_servo_pipeline(shared_state: SharedServoState | None, servo_thread=None, join_timeout: float = 2.0):
    if shared_state is not None:
        try:
            with shared_state.lock:
                shared_state.running = False
        except Exception:
            pass

    if servo_thread is not None:
        try:
            servo_thread.join(timeout=float(join_timeout))
        except Exception:
            pass


def pause_servo(shared_state: SharedServoState, rtde_c=None):
    with shared_state.lock:
        shared_state.paused = True
    if rtde_c is not None:
        try:
            rtde_c.servoStop()
        except Exception as exc:
            print(f"[SERVO] servoStop failed during pause: {exc}")


def resume_servo_with_current_tcp(shared_state: SharedServoState, rtde_r):
    cur_tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
    with shared_state.lock:
        shared_state.target_tcp = cur_tcp.copy()
        shared_state.target_seq += 1
        shared_state.paused = False


def move_robot_home(
    rtde_c,
    rtde_r=None,
    shared_state: SharedServoState | None = None,
    speed: float = 0.6,
    acc: float = 1.2,
    home_q=None,
):
    if home_q is None:
        home_q = [
            0.0,
            -np.pi / 2,
            -np.pi / 2,
            -np.pi / 2,
            np.pi / 2,
            np.pi / 2,
        ]

    if shared_state is not None:
        pause_servo(shared_state, rtde_c=rtde_c)

    try:
        print(f"[HOME] moveJ -> home_q={np.array2string(np.asarray(home_q), precision=4, suppress_small=True)}")
        rtde_c.moveJ(home_q, float(speed), float(acc))
    finally:
        if shared_state is not None and rtde_r is not None:
            resume_servo_with_current_tcp(shared_state, rtde_r)


def queue_servo_target_from_action(
    *,
    shared_state: SharedServoState,
    rtde_r,
    action,
    zero_rotation_action: bool,
    locked_rotvec=None,
    rz_accum: float = 0.0,
):
    current_tcp_pose = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
    target_tcp, next_rz_accum = compose_target_tcp_from_action(
        current_tcp_pose=current_tcp_pose,
        action=action,
        zero_rotation_action=zero_rotation_action,
        locked_rotvec=locked_rotvec,
        rz_accum=rz_accum,
    )
    update_shared_servo_target(shared_state, target_tcp)
    return target_tcp, next_rz_accum
