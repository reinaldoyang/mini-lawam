import unittest

from mini_lawam.model import MiniLaWAMConfig
from mini_lawam.train import validate_phase2_prior_contract


def _config() -> MiniLaWAMConfig:
    return MiniLaWAMConfig(
        lam_ckpt="/tmp/lawam/stage1.ckpt",
        lam_yaml="/tmp/lawam/stage1.yaml",
        future_horizon=32,
        action_horizon=32,
    )


class Phase2ContractTest(unittest.TestCase):
    def test_accepts_matching_stage1_contract(self):
        cfg = _config()
        checkpoint = {"prior": {}, "cfg": cfg.__dict__.copy()}

        validate_phase2_prior_contract(checkpoint, cfg)

    def test_rejects_mismatched_stage1_contract(self):
        for key, different in (
            ("lam_ckpt", "other.ckpt"),
            ("lam_yaml", "other.yaml"),
            ("future_horizon", 24),
            ("action_horizon", 24),
        ):
            with self.subTest(key=key):
                cfg = _config()
                saved_cfg = cfg.__dict__.copy()
                saved_cfg[key] = different

                with self.assertRaisesRegex(ValueError, key):
                    validate_phase2_prior_contract(
                        {"prior": {}, "cfg": saved_cfg}, cfg,
                    )

    def test_rejects_checkpoint_without_metadata(self):
        with self.assertRaisesRegex(ValueError, "missing `cfg`"):
            validate_phase2_prior_contract({"prior": {}}, _config())


if __name__ == "__main__":
    unittest.main()
