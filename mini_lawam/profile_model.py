"""Profile Mini-LaWAM deployment inference without changing the rollout path.

This utility intentionally separates two questions:

1. Low-overhead timing: how long does ``model.predict()`` take?
2. PyTorch Profiler: which model component/operator consumes that time?

By default the input images are deterministic synthetic RGB frames. Pass real
images when desired; image content does not normally change dense-model runtime,
but using the real preprocessing path is useful for deployment measurements.

Examples:

    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.profile_model \
      --ckpt results/mini_lawam/ckpt.pt

    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.profile_model \
      --ckpt results/mini_lawam/ckpt.pt \
      --table-image table.png --wrist-image wrist.png \
      --include-preprocess --torch-profiler
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)


def latency_summary(values_ms: Sequence[float]) -> Dict[str, float]:
    """Return stable latency statistics for a non-empty sequence."""
    values = np.asarray(values_ms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("latency values must be a non-empty 1D sequence")
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "throughput_hz": float(1000.0 / values.mean()),
    }


def parameter_counts(module: torch.nn.Module) -> Dict[str, int]:
    return {
        "total": int(sum(p.numel() for p in module.parameters())),
        "trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
    }


def _autocast_context(device: torch.device, amp: str):
    if amp == "none":
        return nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    return torch.autocast(device_type=device.type, dtype=dtype)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iterations: int,
    amp: str,
) -> Dict[str, Dict[str, float]]:
    """Benchmark a model callable with synchronized wall and CUDA-event timing."""
    with torch.inference_mode(), _autocast_context(device, amp):
        for _ in range(warmup):
            fn()
        _sync(device)

        wall_ms = []
        cuda_ms = []
        for _ in range(iterations):
            _sync(device)
            start_wall = time.perf_counter_ns()
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            fn()

            if device.type == "cuda":
                end_event.record()
            _sync(device)
            wall_ms.append((time.perf_counter_ns() - start_wall) / 1e6)
            if device.type == "cuda":
                cuda_ms.append(float(start_event.elapsed_time(end_event)))

    result = {"synchronized_wall_ms": latency_summary(wall_ms)}
    if cuda_ms:
        result["cuda_event_ms"] = latency_summary(cuda_ms)
    return result


def benchmark_end_to_end(
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iterations: int,
    amp: str,
) -> Dict[str, float]:
    """Time raw-frame policy calls, including CPU preprocessing and decoding."""
    with torch.inference_mode(), _autocast_context(device, amp):
        for _ in range(warmup):
            fn()
        _sync(device)

        wall_ms = []
        for _ in range(iterations):
            _sync(device)
            start = time.perf_counter_ns()
            fn()
            _sync(device)
            wall_ms.append((time.perf_counter_ns() - start) / 1e6)
    return latency_summary(wall_ms)


def predict_with_component_markers(model, o_t, wrist=None, state=None):
    """Equivalent to ``MiniLaWAM.predict`` with high-level profiler ranges."""
    with record_function("mini_lawam_predict"):
        with record_function("dino_table"):
            u_t = model._feat(o_t)[:, :1]

        with record_function("conv_prior"):
            z_hat = model.prior(u_t[:, 0])

        with record_function("lawam_decoder"):
            u_hat_t = model.lam.decoder(u_t, z_hat)
            if isinstance(u_hat_t, tuple):
                u_hat_t = u_hat_t[0]

        wrist_tok = None
        if model.cfg.use_wrist:
            if wrist is None:
                raise ValueError("checkpoint requires a wrist-camera input")
            with record_function("dino_wrist"):
                wrist_tok = model._feat(wrist)[:, 0]

        with record_function("action_head"):
            st = state if (model.cfg.use_state and state is not None) else None
            views = [u_t[:, 0], u_hat_t[:, 0]]
            if wrist_tok is not None:
                views.append(wrist_tok)
            return model.action_head(views, state=st)


def run_torch_profiler(
    fn: Callable[[], Any],
    device: torch.device,
    amp: str,
    trace_dir: Path,
    top_ops: int,
) -> None:
    """Capture one short trace after an unprofiled warm-up."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), _autocast_context(device, amp):
        for _ in range(5):
            fn()
        _sync(device)

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)
        handler = tensorboard_trace_handler(
            str(trace_dir),
            worker_name=f"mini_lawam_{device.type}",
            use_gzip=True,
        )
        with profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            for _ in range(5):
                fn()
                prof.step()
        _sync(device)

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print("\nPyTorch Profiler top operations")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=top_ops))
    print(f"Trace written to: {trace_dir}")
    print(f"Open with: tensorboard --logdir {trace_dir}")


def load_rgb(path: Optional[str], raw_hw: Sequence[int], seed: int) -> np.ndarray:
    if path:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB")).copy()
    rng = np.random.default_rng(seed)
    h, w = map(int, raw_hw)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _print_latency(name: str, stats: Dict[str, float]) -> None:
    print(
        f"{name:<34} "
        f"mean={stats['mean']:8.3f} ms  "
        f"p50={stats['p50']:8.3f}  "
        f"p95={stats['p95']:8.3f}  "
        f"p99={stats['p99']:8.3f}  "
        f"rate={stats['throughput_hz']:7.2f} Hz"
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark Mini-LaWAM model.predict() and optionally capture a "
            "PyTorch Profiler trace."
        )
    )
    ap.add_argument("--ckpt", required=True, help="Trained Mini-LaWAM checkpoint.")
    ap.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument(
        "--amp",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Autocast precision. 'none' matches the current rollout default.",
    )
    ap.add_argument(
        "--include-preprocess",
        action="store_true",
        help="Also time policy.act() from raw RGB through decoded CPU action chunk.",
    )
    ap.add_argument("--table-image", default=None, help="Optional table RGB image.")
    ap.add_argument("--wrist-image", default=None, help="Optional wrist RGB image.")
    ap.add_argument(
        "--raw-frame-hw",
        nargs=2,
        type=int,
        default=(480, 640),
        metavar=("H", "W"),
        help="Synthetic raw-frame size when --table-image is omitted.",
    )
    ap.add_argument(
        "--train-frame-hw",
        nargs=2,
        type=int,
        default=(168, 224),
        metavar=("H", "W"),
        help="Recorded training resolution used by rollout preprocessing; 0 0 disables.",
    )
    ap.add_argument(
        "--state-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="State used by state-conditioned or delta checkpoints.",
    )
    ap.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Capture a short operator/component trace after latency measurement.",
    )
    ap.add_argument(
        "--trace-dir",
        default="results/mini_lawam/profile",
        help="TensorBoard profiler output directory.",
    )
    ap.add_argument("--top-ops", type=int, default=20)
    ap.add_argument(
        "--output-json",
        default=None,
        help="Optional path for machine-readable benchmark results.",
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.top_ops < 1:
        raise ValueError("--top-ops must be >= 1")
    if any(v < 1 for v in args.raw_frame_hw):
        raise ValueError("--raw-frame-hw values must be >= 1")
    if tuple(args.train_frame_hw) == (0, 0):
        train_frame_hw = None
    elif any(v < 1 for v in args.train_frame_hw):
        raise ValueError("--train-frame-hw must contain positive values or exactly 0 0")
    else:
        train_frame_hw = tuple(args.train_frame_hw)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
    elif args.amp != "none":
        raise ValueError("--amp currently requires a CUDA device")

    # Import after argument validation so `--help` remains fast and does not load LaWAM.
    from mini_lawam.rollout import MiniLaWAMPolicy

    print(f"Loading checkpoint: {ckpt_path}")
    policy = MiniLaWAMPolicy(
        str(ckpt_path),
        device=str(device),
        train_frame_hw=train_frame_hw,
    )
    model = policy.model

    table_rgb = load_rgb(args.table_image, args.raw_frame_hw, args.seed)
    wrist_rgb = None
    if policy.cfg.use_wrist:
        if args.wrist_image:
            wrist_rgb = load_rgb(args.wrist_image, args.raw_frame_hw, args.seed + 1)
        elif args.table_image:
            print("No --wrist-image supplied; reusing the table image for timing.")
            wrist_rgb = table_rgb.copy()
        else:
            wrist_rgb = load_rgb(None, args.raw_frame_hw, args.seed + 1)
    elif args.wrist_image:
        print("Checkpoint does not use wrist input; --wrist-image is ignored.")

    needs_state = bool(policy.cfg.use_state) or policy.cfg.target_mode == "delta"
    state_xyz = None
    if needs_state:
        if args.state_xyz is None:
            state_xyz = policy.action_mean[:3].copy()
            print(
                "No --state-xyz supplied; using the action normalization center "
                "for timing only."
            )
        else:
            state_xyz = np.asarray(args.state_xyz, dtype=np.float32)

    # Preprocess once outside the model-only benchmark.
    o_t = policy.preprocess(table_rgb)
    wrist = policy.preprocess(wrist_rgb) if wrist_rgb is not None else None
    state = None
    if policy.cfg.use_state:
        normalized = (
            np.asarray(state_xyz, dtype=np.float32) - policy.action_mean[:3]
        ) / policy.action_std[:3]
        state = torch.from_numpy(normalized).view(1, 3).to(device)
    _sync(device)

    params = {
        "model": parameter_counts(model),
        "lam_container": parameter_counts(model.lam),
        "prior": parameter_counts(model.prior),
        "action_head": parameter_counts(model.action_head),
    }
    if hasattr(model.action_head, "xyz_out"):
        params["xyz_output"] = parameter_counts(model.action_head.xyz_out)
    if hasattr(model.action_head, "gripper_out"):
        params["gripper_output"] = parameter_counts(model.action_head.gripper_out)

    print(
        f"Device={device} | head={policy.cfg.head_type} | "
        f"gripper_head={policy.cfg.gripper_head} | wrist={policy.cfg.use_wrist} | "
        f"state={policy.cfg.use_state} | amp={args.amp}"
    )
    for name, counts in params.items():
        print(
            f"params {name:<14} total={counts['total']:,} "
            f"trainable={counts['trainable']:,}"
        )

    if device.type == "cuda":
        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_allocated = 0

    model_fn = lambda: model.predict(o_t, wrist=wrist, state=state)
    results: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "torch_version": torch.__version__,
        "amp": args.amp,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "config": asdict(policy.cfg),
        "parameters": params,
    }

    print("\nModel-only latency")
    model_latency = benchmark_model(
        model_fn, device, args.warmup, args.iterations, args.amp
    )
    results["model_only"] = model_latency
    for name, stats in model_latency.items():
        _print_latency(name, stats)

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        memory = {
            "baseline_allocated_mib": baseline_allocated / (1024**2),
            "peak_allocated_mib": peak_allocated / (1024**2),
            "incremental_peak_mib": max(0, peak_allocated - baseline_allocated) / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        }
        results["cuda_memory"] = memory
        print(
            "CUDA memory                       "
            f"baseline={memory['baseline_allocated_mib']:.1f} MiB  "
            f"peak={memory['peak_allocated_mib']:.1f} MiB  "
            f"incremental={memory['incremental_peak_mib']:.1f} MiB"
        )

    if args.include_preprocess:
        print("\nRaw-frame policy latency (preprocess + predict + decode)")
        act_fn = lambda: policy.act(
            table_rgb,
            wrist_hwc_uint8=wrist_rgb,
            state_xyz=state_xyz,
        )
        end_to_end = benchmark_end_to_end(
            act_fn, device, args.warmup, args.iterations, args.amp
        )
        results["raw_frame_policy_act_ms"] = end_to_end
        _print_latency("policy.act wall", end_to_end)

    if args.torch_profiler:
        print("\nCapturing PyTorch Profiler trace...")
        component_fn = lambda: predict_with_component_markers(
            model, o_t, wrist=wrist, state=state
        )
        run_torch_profiler(
            component_fn,
            device,
            args.amp,
            Path(args.trace_dir),
            args.top_ops,
        )
        results["profiler_trace_dir"] = args.trace_dir

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"JSON results written to: {output_path}")


if __name__ == "__main__":
    main()
