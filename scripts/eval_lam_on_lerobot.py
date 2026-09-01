#!/usr/bin/env python3
"""Evaluate pretrained LaWM/LAM rollout quality on a LeRobot v3 dataset.

The metric definitions match the pre-existing local HDF5 evaluator
``scripts/eval_lam_on_dataset.py``. This script changes only dataset access,
sampling, aggregation, and result persistence.

Default gap seconds reproduce the HDF5 evaluator's physical-time sweep:
0.8, 1.2, 1.6, and 2.4 seconds. On a 30 FPS dataset these become 24, 36,
48, and 72 frames. The released LAM was trained with a 1.6-second interval.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.video_aug import LAM_IMAGE_HW, gpu_two_view_video_aug
from mini_lawam.lerobot_video import LeRobotEpisodeSource

DEFAULT_LAM_CKPT = (
    "latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
)
DEFAULT_LAM_YAML = (
    "latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"
)
DEFAULT_CAMERA = "observation.images.cam_high"
DEFAULT_GAP_SECONDS = (0.8, 1.2, 1.6, 2.4)


@dataclass(frozen=True)
class FramePair:
    episode_index: int
    t: int
    gap: int


def _read_lam_frame_dt_sec(config_path: str | os.PathLike[str]) -> float:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"LAM config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config: Any = yaml.safe_load(handle)
    try:
        value = float(config["data"]["frame_dt_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing positive data.frame_dt_sec in LAM config: {path}") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"data.frame_dt_sec must be positive in {path}, got {value}")
    return value


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(requested)


def _resolve_gaps(
    *,
    fps: float,
    gaps: Iterable[int] | None,
    gap_seconds: Iterable[float] | None,
) -> list[int]:
    if gaps is not None:
        resolved = [int(value) for value in gaps]
    else:
        seconds = DEFAULT_GAP_SECONDS if gap_seconds is None else gap_seconds
        resolved = [max(1, round(float(value) * fps)) for value in seconds]
    if not resolved or any(value < 1 for value in resolved):
        raise ValueError("All frame gaps must be positive")
    return list(dict.fromkeys(resolved))


def _plan_samples(
    sources: dict[int, LeRobotEpisodeSource],
    gap: int,
    count: int,
    rng: np.random.Generator,
    sampling: str,
) -> list[FramePair]:
    eligible = [source for source in sources.values() if source.length > gap]
    if not eligible:
        return []

    plan: list[FramePair] = []
    if sampling == "episode":
        for _ in range(count):
            source = eligible[int(rng.integers(len(eligible)))]
            t = int(rng.integers(0, source.length - gap))
            plan.append(FramePair(source.episode_index, t, gap))
        return plan

    if sampling != "pair":
        raise ValueError(f"Unknown sampling policy: {sampling}")
    pair_counts = np.asarray([source.length - gap for source in eligible], dtype=np.int64)
    cumulative = np.cumsum(pair_counts)
    total_pairs = int(cumulative[-1])
    for flat_index in rng.integers(0, total_pairs, size=count):
        episode_position = int(np.searchsorted(cumulative, flat_index, side="right"))
        previous = 0 if episode_position == 0 else int(cumulative[episode_position - 1])
        source = eligible[episode_position]
        plan.append(
            FramePair(
                episode_index=source.episode_index,
                t=int(flat_index) - previous,
                gap=gap,
            )
        )
    return plan


def _load_pair_batch(
    samples: list[FramePair],
    sources: dict[int, LeRobotEpisodeSource],
    camera: str,
    resize: v2.Resize,
) -> torch.Tensor:
    pairs: list[np.ndarray | None] = [None] * len(samples)
    positions_by_episode: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        positions_by_episode.setdefault(sample.episode_index, []).append(position)

    for episode_index, positions in positions_by_episode.items():
        indices: list[int] = []
        for position in positions:
            sample = samples[position]
            indices.extend([sample.t, sample.t + sample.gap])
        decoded = sources[episode_index].frames(camera, indices)
        for local_position, output_position in enumerate(positions):
            start = local_position * 2
            pairs[output_position] = decoded[start : start + 2]

    clips: list[torch.Tensor] = []
    for pair in pairs:
        if pair is None:
            raise RuntimeError("Internal error: a planned frame pair was not decoded")
        tensor = torch.from_numpy(np.ascontiguousarray(pair)).permute(0, 3, 1, 2)
        clips.append(resize(tensor))
    return torch.stack(clips).to(torch.uint8)


def _per_sample_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim == 4:
        a = a[:, 0]
    if b.ndim == 4:
        b = b[:, 0]
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean(dim=-1)


def _per_sample_region_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    cosine = F.cosine_similarity(
        prediction[:, 0].float(), target[:, 0].float(), dim=-1
    )
    return cosine.gather(1, indices).mean(dim=-1)


@torch.no_grad()
def _evaluate_gap(
    *,
    lam: torch.nn.Module,
    plan: list[FramePair],
    sources: dict[int, LeRobotEpisodeSource],
    camera: str,
    batch_size: int,
    topk: int,
    device: torch.device,
    resize: v2.Resize,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(plan), batch_size):
        samples = plan[start : start + batch_size]
        clips_u8 = _load_pair_batch(samples, sources, camera, resize)
        videos, _ = gpu_two_view_video_aug(clips_u8.to(device), training=False)
        output = lam.get_latent_action(
            videos=videos,
            states=None,
            dec_videos=videos,
            predict_future_frame=True,
        )
        u_t = output["dec_in"]
        u_T = output["tgt"]
        u_hat = output["recon"]
        latent = output["quantized"]

        token_count = int(u_t.shape[-2])
        if topk > token_count:
            raise ValueError(f"--topk {topk} exceeds model token count {token_count}")
        motion_order = F.cosine_similarity(
            u_t[:, 0].float(), u_T[:, 0].float(), dim=-1
        ).argsort(dim=1)
        motion_indices = motion_order[:, :topk]

        rollout = _per_sample_cosine(u_hat, u_T)
        initial = _per_sample_cosine(u_t, u_T)
        rollout_region = _per_sample_region_cosine(u_hat, u_T, motion_indices)
        initial_region = _per_sample_region_cosine(u_t, u_T, motion_indices)

        shuffled = torch.full_like(rollout, torch.nan)
        shuffled_region = torch.full_like(rollout_region, torch.nan)
        if len(samples) > 1:
            # A one-position rotation is a deterministic derangement: every
            # sample receives another sample's latent action, with no fixed points.
            permutation = torch.roll(
                torch.arange(len(samples), device=latent.device), shifts=1
            )
            shuffled_hat = lam.decoder(u_t, latent[permutation])
            if isinstance(shuffled_hat, tuple):
                shuffled_hat = shuffled_hat[0]
            shuffled = _per_sample_cosine(shuffled_hat, u_T)
            shuffled_region = _per_sample_region_cosine(
                shuffled_hat, u_T, motion_indices
            )

        arrays = {
            "rollout_vs_gt": rollout.cpu().numpy(),
            "init_vs_gt": initial.cpu().numpy(),
            "shuffled_vs_gt": shuffled.cpu().numpy(),
            "rollout_region": rollout_region.cpu().numpy(),
            "init_region": initial_region.cpu().numpy(),
            "shuffled_region": shuffled_region.cpu().numpy(),
        }
        for position, sample in enumerate(samples):
            row: dict[str, Any] = {
                "episode_index": sample.episode_index,
                "t": sample.t,
                "gap_frames": sample.gap,
            }
            row.update({name: float(values[position]) for name, values in arrays.items()})
            rows.append(row)
        print(f"  evaluated {min(start + batch_size, len(plan))}/{len(plan)} pairs")
    return rows


def _finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _summarize(rows: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    gap = int(rows[0]["gap_frames"])
    result = {
        "gap_frames": gap,
        "gap_seconds": gap / fps,
        "num_pairs": len(rows),
        "num_shuffled_pairs": int(
            sum(np.isfinite(row["shuffled_vs_gt"]) for row in rows)
        ),
    }
    for key in (
        "rollout_vs_gt",
        "init_vs_gt",
        "shuffled_vs_gt",
        "rollout_region",
        "init_region",
        "shuffled_region",
    ):
        result[key] = _finite_mean(rows, key)
    result["roll_minus_init"] = result["rollout_vs_gt"] - result["init_vs_gt"]
    result["roll_minus_shuffled"] = (
        result["rollout_vs_gt"] - result["shuffled_vs_gt"]
    )
    result["region_roll_minus_init"] = (
        result["rollout_region"] - result["init_region"]
    )
    result["region_roll_minus_shuffled"] = (
        result["rollout_region"] - result["shuffled_region"]
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(results: list[dict[str, Any]], topk: int) -> None:
    header = (
        f"{'gap':>5} {'~sec':>6} {'rollout_vs_gt':>14} {'init_vs_gt':>11} "
        f"{'shuffled_vs_gt':>15} {'roll-init':>10}"
    )
    print("\n=== WHOLE FRAME ===")
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['gap_frames']:>5} {row['gap_seconds']:>6.2f} "
            f"{row['rollout_vs_gt']:>14.4f} {row['init_vs_gt']:>11.4f} "
            f"{row['shuffled_vs_gt']:>15.4f} {row['roll_minus_init']:>10.4f}"
        )

    print(f"\n=== MOTION REGION (lowest-cosine {topk} patches) ===")
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['gap_frames']:>5} {row['gap_seconds']:>6.2f} "
            f"{row['rollout_region']:>14.4f} {row['init_region']:>11.4f} "
            f"{row['shuffled_region']:>15.4f} "
            f"{row['region_roll_minus_init']:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot", required=True, help="local LeRobot v3 dataset")
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="video feature used for LAM evaluation")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="episode_index values; default = all episodes")
    gap_group = parser.add_mutually_exclusive_group()
    gap_group.add_argument("--gaps", type=int, nargs="+", default=None,
                           help="frame gaps to evaluate")
    gap_group.add_argument("--gap-seconds", type=float, nargs="+", default=None,
                           help="physical-time gaps converted with dataset FPS")
    parser.add_argument("--num-pairs", type=int, default=256,
                        help="sample count per gap (sampling with replacement)")
    parser.add_argument("--sampling", choices=("episode", "pair"), default="episode",
                        help="uniform episodes (HDF5-compatible) or uniform frame pairs")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--topk", type=int, default=24,
                        help="number of most-changing tokens in motion-region metrics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", default=DEFAULT_LAM_CKPT)
    parser.add_argument("--yaml", default=DEFAULT_LAM_YAML)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--out-dir", default="results/lam_check/lerobot_metrics")
    args = parser.parse_args()

    if args.num_pairs < 2:
        raise ValueError("--num-pairs must be >= 2 to compute the shuffled baseline")
    if args.batch < 2:
        raise ValueError("--batch must be >= 2 to compute the shuffled baseline")
    if args.topk < 1:
        raise ValueError("--topk must be >= 1")

    sources = LeRobotEpisodeSource.open_all(
        args.lerobot,
        camera_keys=[args.camera],
        episode_indices=args.episodes,
    )
    fps_values = {source.fps for source in sources.values()}
    if len(fps_values) != 1:
        raise ValueError(f"Selected episodes report inconsistent FPS values: {fps_values}")
    fps = fps_values.pop()
    gaps = _resolve_gaps(fps=fps, gaps=args.gaps, gap_seconds=args.gap_seconds)
    trained_frame_dt_sec = _read_lam_frame_dt_sec(args.yaml)
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"dataset={next(iter(sources.values())).root} | episodes={len(sources)} | "
        f"camera={args.camera} | fps={fps:g}"
    )
    print(
        f"gaps={gaps} frames | pretrained interval={trained_frame_dt_sec:g}s "
        f"(~{round(trained_frame_dt_sec * fps)} frames)"
    )
    print(f"sampling={args.sampling} | pairs/gap={args.num_pairs} | seed={args.seed}")
    print(f"Loading LAM on {device} ...")
    lam = load_latent_action_model(args.ckpt, args.yaml).to(device).eval()
    resize = v2.Resize(LAM_IMAGE_HW, antialias=True)

    all_rows: list[dict[str, Any]] = []
    plans: dict[str, list[dict[str, int]]] = {}
    summaries: list[dict[str, Any]] = []
    for gap in gaps:
        plan = _plan_samples(sources, gap, args.num_pairs, rng, args.sampling)
        if not plan:
            print(f"skip gap={gap}: no selected episode is long enough")
            continue
        plans[str(gap)] = [
            {"episode_index": sample.episode_index, "t": sample.t, "gap": sample.gap}
            for sample in plan
        ]
        print(f"\ngap={gap} frames ({gap / fps:.3f}s)")
        rows = _evaluate_gap(
            lam=lam,
            plan=plan,
            sources=sources,
            camera=args.camera,
            batch_size=args.batch,
            topk=args.topk,
            device=device,
            resize=resize,
        )
        all_rows.extend(rows)
        summaries.append(_summarize(rows, fps))

    if not summaries:
        raise RuntimeError("No gap produced any evaluation samples")
    _print_summary(summaries, args.topk)

    output_dir = Path(args.out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(next(iter(sources.values())).root),
        "codebase_version": next(iter(sources.values())).info.get("codebase_version"),
        "camera": args.camera,
        "fps": fps,
        "episode_indices": sorted(sources),
        "checkpoint": str(Path(args.ckpt).expanduser()),
        "yaml": str(Path(args.yaml).expanduser()),
        "trained_frame_dt_sec": trained_frame_dt_sec,
        "gaps": gaps,
        "num_pairs_per_gap": args.num_pairs,
        "sampling": args.sampling,
        "batch_size": args.batch,
        "topk": args.topk,
        "seed": args.seed,
        "metric_definitions": {
            "rollout_vs_gt": "mean token cosine(u_hat_T, u_T)",
            "init_vs_gt": "mean token cosine(u_t, u_T)",
            "shuffled_vs_gt": "mean token cosine(decoder(u_t, z_other), u_T)",
            "motion_region": f"the {args.topk} tokens with lowest cosine(u_t, u_T)",
        },
        "results": summaries,
    }
    metrics_json = output_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=True)
    with (output_dir / "sample_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plans, handle, indent=2)
    _write_csv(output_dir / "metrics.csv", summaries)
    _write_csv(output_dir / "per_sample.csv", all_rows)
    print(f"\nsaved aggregate metrics: {metrics_json}")
    print(f"saved CSV and deterministic sample plan under: {output_dir}")


if __name__ == "__main__":
    main()
