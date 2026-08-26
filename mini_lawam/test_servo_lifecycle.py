import numpy as np

from mini_lawam.rollout_ur7e import (
    read_checked_actual_tcp,
    stop_servo_pipeline_for_idle,
)


class FakeThread:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


class FakeReceive:
    def __init__(self, pose, connected=True):
        self.pose = pose
        self.connected = connected

    def isConnected(self):
        return self.connected

    def getActualTCPPose(self):
        return self.pose


def test_idle_transition_stops_and_joins_servo_worker():
    state = object()
    thread = FakeThread()
    calls = []

    def stop_fn(received_state, received_thread):
        calls.append((received_state, received_thread))
        received_thread.alive = False

    next_state, next_thread = stop_servo_pipeline_for_idle(
        state, thread, stop_fn
    )

    assert calls == [(state, thread)]
    assert next_state is None
    assert next_thread is None


def test_home_is_refused_if_servo_worker_does_not_exit():
    state = object()
    thread = FakeThread()

    try:
        stop_servo_pipeline_for_idle(state, thread, lambda *_: None)
    except RuntimeError as exc:
        assert "refusing" in str(exc)
    else:
        raise AssertionError("live servo worker was accepted at the home boundary")


def test_already_stopped_pipeline_is_a_noop():
    def unexpected_stop(*_):
        raise AssertionError("stop function should not be called")

    assert stop_servo_pipeline_for_idle(None, None, unexpected_stop) == (None, None)


def test_actual_tcp_read_fails_closed_on_disconnect_or_invalid_pose():
    valid = np.asarray([0.4, 0.1, 0.3, 0.0, 3.14, 0.0])
    np.testing.assert_array_equal(read_checked_actual_tcp(FakeReceive(valid)), valid)

    invalid_receivers = [
        FakeReceive(valid, connected=False),
        FakeReceive([0.1, 0.2, 0.3]),
        FakeReceive([0.4, 0.1, np.nan, 0.0, 3.14, 0.0]),
    ]
    for receiver in invalid_receivers:
        try:
            read_checked_actual_tcp(receiver)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid RTDE state was accepted")


if __name__ == "__main__":
    test_idle_transition_stops_and_joins_servo_worker()
    test_home_is_refused_if_servo_worker_does_not_exit()
    test_already_stopped_pipeline_is_a_noop()
    test_actual_tcp_read_fails_closed_on_disconnect_or_invalid_pose()
    print("4 focused servo-lifecycle tests passed")
