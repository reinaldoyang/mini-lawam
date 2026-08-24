from pathlib import Path
import tempfile

import torch
from torch import nn

from latent_action_model.core.checkpoint import load_pretrained_weights


class _TinyLAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 1)
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False


def test_load_pretrained_weights_loads_only_model_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = _TinyLAM()
        with torch.no_grad():
            source.vision_encoder.weight.fill_(2.0)
            source.encoder.weight.fill_(3.0)
        checkpoint_path = Path(tmpdir) / "released.pt"
        torch.save(
            {
                "state_dict": source.state_dict(),
                "optimizer_states": [{"must_not_be_restored": True}],
                "epoch": 99,
                "global_step": 1234,
            },
            checkpoint_path,
        )

        target = _TinyLAM()
        loaded_count = load_pretrained_weights(target, checkpoint_path)

        assert loaded_count == len(source.state_dict())
        assert torch.equal(target.vision_encoder.weight, source.vision_encoder.weight)
        assert torch.equal(target.encoder.weight, source.encoder.weight)
        assert not target.vision_encoder.weight.requires_grad
        assert target.encoder.weight.requires_grad


def test_load_pretrained_weights_is_strict_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "incomplete.pt"
        torch.save({"state_dict": {"encoder.weight": torch.ones(1, 2)}}, checkpoint_path)

        try:
            load_pretrained_weights(_TinyLAM(), checkpoint_path)
        except RuntimeError as exc:
            assert "Missing key" in str(exc)
        else:
            raise AssertionError("incomplete state dictionary loaded without an error")


def test_load_pretrained_weights_accepts_raw_state_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = _TinyLAM()
        checkpoint_path = Path(tmpdir) / "raw.pt"
        torch.save(source.state_dict(), checkpoint_path)

        target = _TinyLAM()
        assert load_pretrained_weights(target, checkpoint_path) == len(source.state_dict())


def test_load_pretrained_weights_rejects_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            load_pretrained_weights(_TinyLAM(), Path(tmpdir) / "missing.pt")
        except FileNotFoundError as exc:
            assert "Pretrained LAM checkpoint not found" in str(exc)
        else:
            raise AssertionError("missing checkpoint did not raise FileNotFoundError")


if __name__ == "__main__":
    test_load_pretrained_weights_loads_only_model_state()
    test_load_pretrained_weights_is_strict_by_default()
    test_load_pretrained_weights_accepts_raw_state_dict()
    test_load_pretrained_weights_rejects_missing_file()
    print("4 focused weights-only checkpoint tests passed")
