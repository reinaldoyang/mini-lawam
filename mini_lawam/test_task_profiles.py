from dataclasses import asdict, replace
import tempfile
from pathlib import Path

import numpy as np
import torch

from mini_lawam.model import MiniLaWAMConfig
from mini_lawam.rollout import MiniLaWAMPolicy
from mini_lawam.rollout_ur7e import LatchedGripper, build_task_profiles


class FakeRobotiqGripper:
    def __init__(self):
        self.moves = []
        self.connected = None

    def connect(self, host, port):
        self.connected = (host, port)

    def move(self, position, speed, force):
        self.moves.append((position, speed, force))

    def disconnect(self):
        pass


class FakePolicyModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prior = torch.nn.Linear(2, 2)
        self.action_head = torch.nn.Linear(2, 4)
        self.eval_called = False

    def eval(self):
        self.eval_called = True


def test_task_profiles_bind_keys_to_checkpoints_and_close_widths():
    profiles, switching = build_task_profiles(
        [("1", "pick_egg.pt", "15"), ("2", "pick_bamboo.pt", "23")],
        default_ckpt="ignored.pt",
        default_close_mm=10,
        open_mm=52,
    )
    assert switching is True
    assert list(profiles) == ["1", "2"]
    assert profiles["1"] == {
        "key": "1", "ckpt": "pick_egg.pt", "close_mm": 15.0,
    }
    assert profiles["2"]["close_mm"] == 23.0


def test_no_task_profiles_preserves_existing_single_task_flags():
    profiles, switching = build_task_profiles(None, "original.pt", 18, 52)
    assert switching is False
    assert profiles["1"]["ckpt"] == "original.pt"
    assert profiles["1"]["close_mm"] == 18.0


def test_invalid_task_profiles_are_rejected():
    invalid_cases = [
        ([("0", "bad.pt", "10")], 52),
        ([("1", "a.pt", "10"), ("1", "b.pt", "20")], 52),
        ([("1", "bad.pt", "53")], 52),
        ([("1", "bad.pt", "-1")], 52),
        ([("1", "bad.pt", "nan")], 52),
    ]
    for raw_profiles, open_mm in invalid_cases:
        try:
            build_task_profiles(raw_profiles, "default.pt", 23, open_mm)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid task profile was accepted: {raw_profiles}")


def test_switching_close_width_does_not_move_until_next_command():
    gripper = LatchedGripper(
        FakeRobotiqGripper, "127.0.0.1", open_mm=52, close_mm=23,
        speed=100, force=50,
    )
    gripper.command("close")
    first_raw = gripper._g.moves[-1][0]

    gripper.set_close_mm(15)
    assert len(gripper._g.moves) == 1
    assert gripper.state is None

    gripper.command("close")
    second_raw = gripper._g.moves[-1][0]
    assert len(gripper._g.moves) == 2
    assert second_raw > first_raw
    assert second_raw == int(np.clip(round(255 * (1 - 15 / 52)), 0, 255))


def test_compatible_checkpoint_hot_swap_replaces_task_weights_and_metadata():
    cfg = MiniLaWAMConfig(head_type="attn", use_wrist=True, target_mode="joystick")
    policy = MiniLaWAMPolicy.__new__(MiniLaWAMPolicy)
    policy.cfg = cfg
    policy.model = FakePolicyModel(cfg)

    new_prior = torch.nn.Linear(2, 2)
    new_head = torch.nn.Linear(2, 4)
    with torch.no_grad():
        new_prior.weight.fill_(3.0)
        new_head.weight.fill_(5.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "task.pt"
        torch.save(
            {
                "cfg": asdict(replace(cfg, gripper_target_offset=1)),
                "prior": new_prior.state_dict(),
                "action_head": new_head.state_dict(),
                "action_mean": np.asarray([1, 2, 3, 4], dtype=np.float32),
                "action_std": np.asarray([5, 6, 7, 8], dtype=np.float32),
            },
            path,
        )
        loaded_cfg = policy.reload_checkpoint(str(path))

    torch.testing.assert_close(policy.model.prior.weight, new_prior.weight)
    torch.testing.assert_close(policy.model.action_head.weight, new_head.weight)
    np.testing.assert_array_equal(policy.action_mean, [1, 2, 3, 4])
    assert loaded_cfg.gripper_target_offset == 1
    assert policy.model.eval_called


def test_incompatible_checkpoint_is_rejected_before_weights_change():
    cfg = MiniLaWAMConfig()
    policy = MiniLaWAMPolicy.__new__(MiniLaWAMPolicy)
    policy.cfg = cfg
    policy.model = FakePolicyModel(cfg)
    original = policy.model.prior.weight.detach().clone()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "incompatible.pt"
        torch.save(
            {
                "cfg": asdict(replace(cfg, action_horizon=12)),
                "prior": policy.model.prior.state_dict(),
                "action_head": policy.model.action_head.state_dict(),
                "action_mean": np.zeros(4, dtype=np.float32),
                "action_std": np.ones(4, dtype=np.float32),
            },
            path,
        )
        try:
            policy.reload_checkpoint(str(path))
        except ValueError as exc:
            assert "action_horizon" in str(exc)
        else:
            raise AssertionError("incompatible checkpoint was accepted")

    torch.testing.assert_close(policy.model.prior.weight, original)


if __name__ == "__main__":
    test_task_profiles_bind_keys_to_checkpoints_and_close_widths()
    test_no_task_profiles_preserves_existing_single_task_flags()
    test_invalid_task_profiles_are_rejected()
    test_switching_close_width_does_not_move_until_next_command()
    test_compatible_checkpoint_hot_swap_replaces_task_weights_and_metadata()
    test_incompatible_checkpoint_is_rejected_before_weights_change()
    print("6 focused task-profile tests passed")
