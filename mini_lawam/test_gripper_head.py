import numpy as np
import torch

from mini_lawam.model import (
    AttnActionHead,
    MLPActionHead,
    compute_action_losses,
)
from mini_lawam.rollout import decode_action_prediction


def test_attention_binary_head_has_independent_outputs_and_gradients():
    head = AttnActionHead(
        token_dim=8,
        action_dim=4,
        horizon=4,
        n_views=2,
        hidden=12,
        n_layers=1,
        n_heads=3,
        gripper_head="binary",
    )
    pred = head([torch.randn(2, 5, 8), torch.randn(2, 5, 8)])
    assert pred.shape == (2, 4, 4)
    state_keys = set(head.state_dict())
    assert "xyz_out.weight" in state_keys
    assert "gripper_out.weight" in state_keys
    assert "out.weight" not in state_keys

    actions = torch.randn(2, 4, 4)
    mask = torch.ones_like(actions)
    gripper_targets = torch.tensor(
        [[[0.0], [0.0], [1.0], [1.0]], [[1.0], [0.0], [1.0], [0.0]]]
    )
    loss_act, loss_xyz, loss_gripper, accuracy = compute_action_losses(
        pred, actions, mask, gripper_targets,
        gripper_head="binary", lambda_gripper=1.0,
    )
    torch.testing.assert_close(loss_act, loss_xyz + loss_gripper)
    assert 0.0 <= float(accuracy) <= 1.0
    loss_act.backward()
    assert head.xyz_out.weight.grad is not None
    assert head.gripper_out.weight.grad is not None
    assert float(head.xyz_out.weight.grad.abs().sum()) > 0.0
    assert float(head.gripper_out.weight.grad.abs().sum()) > 0.0


def test_mlp_binary_head_shape():
    head = MLPActionHead(
        in_dim=10, action_dim=4, horizon=3, hidden=16,
        gripper_head="binary",
    )
    assert head(torch.randn(2, 10)).shape == (2, 3, 4)


def test_regression_attention_layout_remains_checkpoint_compatible():
    head = AttnActionHead(
        token_dim=8,
        action_dim=4,
        horizon=4,
        n_views=2,
        hidden=12,
        n_layers=1,
        n_heads=3,
        gripper_head="regression",
    )
    state_keys = set(head.state_dict())
    assert "out.weight" in state_keys
    assert "xyz_out.weight" not in state_keys
    assert "gripper_out.weight" not in state_keys


def test_binary_decode_denormalizes_xyz_and_emits_exact_gripper_states():
    raw_pred = np.asarray(
        [
            [1.0, -2.0, 0.5, -0.01],
            [-1.0, 2.0, -0.5, 0.00],
            [0.0, 0.0, 0.0, 3.00],
        ],
        dtype=np.float32,
    )
    mean = np.asarray([0.1, 0.2, 0.3, -0.157], dtype=np.float32)
    std = np.asarray([2.0, 3.0, 4.0, 0.988], dtype=np.float32)
    chunk = decode_action_prediction(raw_pred, mean, std, gripper_head="binary")

    np.testing.assert_allclose(chunk[:, :3], raw_pred[:, :3] * std[:3] + mean[:3])
    np.testing.assert_array_equal(
        chunk[:, 3], np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
    )


def test_regression_loss_preserves_legacy_four_dimensional_mse():
    pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred)
    loss_act, _, _, accuracy = compute_action_losses(
        pred, target, mask, gripper_targets=None,
        gripper_head="regression", lambda_gripper=99.0,
    )
    torch.testing.assert_close(loss_act, ((pred - target) ** 2).mean())
    assert accuracy is None


if __name__ == "__main__":
    test_attention_binary_head_has_independent_outputs_and_gradients()
    test_mlp_binary_head_shape()
    test_regression_attention_layout_remains_checkpoint_compatible()
    test_binary_decode_denormalizes_xyz_and_emits_exact_gripper_states()
    test_regression_loss_preserves_legacy_four_dimensional_mse()
    print("5 focused binary-gripper-head tests passed")
