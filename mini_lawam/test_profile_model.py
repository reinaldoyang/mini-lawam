import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from mini_lawam.profile_model import (
    _canonical_device,
    _print_component_latency,
    benchmark_components,
    build_parser,
    render_markdown_report,
)


class _DummyModel:
    def __init__(self, use_wrist: bool):
        self.cfg = SimpleNamespace(use_wrist=use_wrist, use_state=False)
        self.lam = SimpleNamespace(decoder=lambda u_t, _z_hat: u_t + 1)
        self.prior = lambda tokens: tokens + 1
        self.action_head = lambda views, state=None: sum(views)

    def _feat(self, images):
        return images + 1


class ProfileModelTest(unittest.TestCase):
    @staticmethod
    def _stats(mean):
        return {
            "mean": mean,
            "std": 0.1,
            "min": mean - 0.1,
            "p50": mean,
            "p95": mean + 0.1,
            "p99": mean + 0.2,
            "max": mean + 0.2,
            "throughput_hz": 1000.0 / mean,
        }

    def test_bare_cuda_device_resolves_to_logical_device_zero(self):
        self.assertEqual(_canonical_device(torch.device("cuda")), torch.device("cuda:0"))
        self.assertEqual(_canonical_device(torch.device("cuda:2")), torch.device("cuda:2"))
        self.assertEqual(_canonical_device(torch.device("cpu")), torch.device("cpu"))

    def test_component_benchmark_reports_all_high_level_stages(self):
        model = _DummyModel(use_wrist=True)
        image = torch.zeros(1, 1, 4, 3)

        result = benchmark_components(
            model=model,
            o_t=image,
            wrist=image,
            state=None,
            device=torch.device("cpu"),
            warmup=1,
            iterations=2,
            amp="none",
        )

        self.assertEqual(
            set(result), {"dino", "conv_prior", "lawam_decoder", "action_head"}
        )
        for stats in result.values():
            self.assertGreater(stats["mean"], 0.0)
            self.assertIn("p50", stats)
            self.assertIn("p95", stats)

    def test_component_latency_terminal_format(self):
        output = io.StringIO()
        with redirect_stdout(output):
            _print_component_latency("DINO", {"mean": 8.2})

        self.assertEqual(output.getvalue(), "DINO:                 8.200 ms\n")

    def test_markdown_report_contains_readable_result_sections(self):
        results = {
            "generated_at": "2026-08-06T17:00:00+08:00",
            "checkpoint": "checkpoint.pt",
            "device": "cuda:0",
            "torch_version": "2.7.1",
            "amp": "none",
            "warmup": 5,
            "iterations": 10,
            "inputs": {
                "table": "Synthetic RGB (seed=0)",
                "wrist": "Synthetic RGB (seed=1)",
                "table_frame_hw": [480, 640],
                "wrist_frame_hw": [480, 640],
                "train_frame_hw": [168, 224],
            },
            "config": {"use_wrist": True},
            "model_only": {"cuda_event_ms": self._stats(13.1)},
            "component_latency_ms": {
                "dino": self._stats(8.2),
                "conv_prior": self._stats(1.4),
                "lawam_decoder": self._stats(2.7),
                "action_head": self._stats(0.8),
            },
            "raw_frame_policy_act_ms": self._stats(15.0),
            "parameters": {"model": {"total": 1000, "trainable": 100}},
            "cuda_memory": {"peak_allocated_mib": 512.0},
            "profiler_trace_dir": "results/mini_lawam/profile",
        }

        report = render_markdown_report(results)

        self.assertIn("# Mini-LaWAM Profiling Report", report)
        self.assertIn("## Component latency", report)
        self.assertIn("| DINO | 8.200", report)
        self.assertIn("| **Component sum** | **13.100**", report)
        self.assertIn("## Raw-frame policy latency", report)
        self.assertIn("## CUDA memory", report)
        self.assertIn("tensorboard --logdir results/mini_lawam/profile", report)

    def test_output_markdown_flag_has_a_default_path(self):
        args = build_parser().parse_args(["--ckpt", "checkpoint.pt", "--output-markdown"])
        self.assertEqual(
            args.output_markdown, "results/mini_lawam/profile_report.md"
        )


if __name__ == "__main__":
    unittest.main()
