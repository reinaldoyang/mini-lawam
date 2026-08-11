#!/usr/bin/env python3
"""Train a Stage 2 gate that decides when to apply the Stage 1 correction."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .stage1_dataset import (
    Stage1CorrectionDataset,
    Stage1SampleRef,
    build_sample_index,
    collate_batch,
    split_refs,
)
from .stage1_model import Stage1CorrectionPolicy, Stage1ModelConfig, load_stage1_checkpoint
from .stage2_model import Stage2GateConfig, Stage2GatedCorrectionPolicy, freeze_except_gate
from .train_stage1 import inspect_data


DEFAULT_STAGE1_CHECKPOINT = (
    "results/hil/mini_lawam_stage1_xyz_rz_grip/residual_stage1.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", default="dataset/hil_mini_lawam_vr")
    parser.add_argument(
        "--stage1-checkpoint",
        "--stage1_checkpoint",
        default=DEFAULT_STAGE1_CHECKPOINT,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="results/hil/mini_lawam_stage2_gate",
    )
    parser.add_argument("--epochs", "--gate-epochs", "--gate_epochs", type=int, default=50)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--val-fraction", "--val_fraction", type=float, default=0.1)
    parser.add_argument("--split-unit", "--split_unit", choices=("demo", "frame"), default="demo")

    model = parser.add_argument_group("gate model")
    model.add_argument("--gate-hidden-dim", "--gate_hidden_dim", type=int, default=128)
    model.add_argument("--gate-hidden-depth", "--gate_hidden_depth", type=int, default=3)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--weight-decay", "--weight_decay", type=float, default=0.0)
    optimization.add_argument("--lr-warmup-steps", "--lr_warmup_steps", type=int, default=1000)
    optimization.add_argument("--lr-cosine-steps", "--lr_cosine_steps", type=int, default=100000)
    optimization.add_argument("--lr-cosine-min", "--lr_cosine_min", type=float, default=1e-6)
    optimization.add_argument("--grad-clip", "--grad_clip", type=float, default=1.0)
    optimization.add_argument(
        "--gate-eval-threshold",
        "--gate_eval_threshold",
        type=float,
        default=0.3,
    )
    optimization.add_argument(
        "--selection-metric",
        "--gate-selection-metric",
        "--gate_selection_metric",
        dest="selection_metric",
        choices=("f1", "loss"),
        default="f1",
    )

    resume = parser.add_argument_group("resume")
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-checkpoint", "--resume_checkpoint", default=None)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "gate_hidden_dim",
        "gate_hidden_depth",
        "lr",
        "grad_clip",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0,1)")
    if not 0.0 <= args.gate_eval_threshold <= 1.0:
        parser.error("--gate-eval-threshold must be in [0,1]")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        parser.error("--weight-decay must be finite and non-negative")


def load_torch_checkpoint(path: str | Path, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def gate_training_semantics(args: argparse.Namespace) -> dict[str, object]:
    return {
        "gate_label_schema": "intervene_mask_binary_v1",
        "negative_sampling": "keep_all_positive_batch_rate_negative_v1",
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "split_unit": str(args.split_unit),
        "gate_eval_threshold": float(args.gate_eval_threshold),
        "selection_metric": str(args.selection_metric),
        "gate_hidden_dim": int(args.gate_hidden_dim),
        "gate_hidden_depth": int(args.gate_hidden_depth),
    }


def assert_matching_stage1_split(args: argparse.Namespace, checkpoint: dict, path: Path) -> None:
    stage1_args = checkpoint.get("args", {})
    expected = {
        "seed": int(stage1_args.get("seed", args.seed)),
        "val_fraction": float(stage1_args.get("val_fraction", args.val_fraction)),
        "split_unit": str(stage1_args.get("split_unit", args.split_unit)),
    }
    current = {
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "split_unit": str(args.split_unit),
    }
    mismatches = [name for name in expected if expected[name] != current[name]]
    if mismatches:
        detail = ", ".join(f"{name}={current[name]!r} (Stage 1: {expected[name]!r})" for name in mismatches)
        raise RuntimeError(
            f"Stage 2 must reuse the Stage 1 train/validation split from {path}: {detail}"
        )


def validate_correction_checkpoint(checkpoint: dict, path: Path) -> Stage1ModelConfig:
    config = Stage1ModelConfig.from_dict(checkpoint.get("model_config", {}))
    if config.arm_action_dim != 4 or config.gripper_classes != 2:
        raise RuntimeError(
            f"{path} is incompatible: expected 4D XYZ/RZ residual and binary gripper, "
            f"got arm_action_dim={config.arm_action_dim}, gripper_classes={config.gripper_classes}"
        )
    mean = np.asarray(checkpoint.get("low_dim_mean"), dtype=np.float32)
    std = np.asarray(checkpoint.get("low_dim_std"), dtype=np.float32)
    clip = np.asarray(checkpoint.get("action_clip"), dtype=np.float32)
    if mean.shape != (config.low_dim_dim,) or std.shape != mean.shape:
        raise RuntimeError(f"{path} contains incompatible low-dimensional normalization statistics")
    if clip.shape != (4,):
        raise RuntimeError(f"{path} action_clip must have shape (4,), got {clip.shape}")
    return config


def validate_data_identity(data_summary: dict[str, object], checkpoint: dict, path: Path) -> None:
    previous = checkpoint.get("data_summary", {})
    keys = ("schema", "base_policy_checkpoint", "base_policy_target_mode")
    mismatches = [key for key in keys if previous.get(key) and previous.get(key) != data_summary.get(key)]
    if mismatches:
        detail = ", ".join(
            f"{key}={data_summary.get(key)!r} (Stage 1: {previous.get(key)!r})"
            for key in mismatches
        )
        raise RuntimeError(f"Stage 2 data does not match {path}: {detail}")


def count_positive_refs(
    refs: Sequence[Stage1SampleRef],
    positive_keys: set[tuple[str, str, int]],
) -> int:
    return sum((ref.path, ref.demo, ref.step) in positive_keys for ref in refs)


def make_loaders(
    args: argparse.Namespace,
    checkpoint: dict,
) -> tuple[DataLoader, DataLoader, int, int, int, int]:
    all_refs = build_sample_index(args.data_dir, intervention_only=False)
    if not all_refs:
        raise RuntimeError(f"no Stage 2 samples found in {args.data_dir}")
    train_refs, val_refs = split_refs(
        all_refs,
        val_fraction=args.val_fraction,
        seed=args.seed,
        split_unit=args.split_unit,
    )
    positive_keys = {
        (ref.path, ref.demo, ref.step)
        for ref in build_sample_index(args.data_dir, intervention_only=True)
    }
    train_positive = count_positive_refs(train_refs, positive_keys)
    val_positive = count_positive_refs(val_refs, positive_keys)
    train_has_both = 0 < train_positive < len(train_refs)
    val_has_both = 0 < val_positive < len(val_refs)
    if not train_refs or not val_refs or not train_has_both or not val_has_both:
        raise RuntimeError(
            "Stage 2 requires non-empty train/validation splits containing both "
            "intervention labels; collect corrections in more episodes or use --split-unit frame"
        )

    config = Stage1ModelConfig.from_dict(checkpoint["model_config"])
    common_dataset = {
        "low_dim_mode": config.low_dim_mode,
        "low_dim_mean": np.asarray(checkpoint["low_dim_mean"], dtype=np.float32),
        "low_dim_std": np.asarray(checkpoint["low_dim_std"], dtype=np.float32),
        "temporal_context": config.temporal_context,
    }
    train_dataset = Stage1CorrectionDataset(args.data_dir, refs=train_refs, **common_dataset)
    val_dataset = Stage1CorrectionDataset(args.data_dir, refs=val_refs, **common_dataset)
    common_loader = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_batch,
        "pin_memory": str(args.device).startswith("cuda") and torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common_loader),
        DataLoader(val_dataset, shuffle=False, **common_loader),
        len(train_refs),
        len(val_refs),
        train_positive,
        val_positive,
    )


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}


def learning_rate_factor(step: int, args: argparse.Namespace) -> float:
    if step < args.lr_warmup_steps:
        return max(
            float(step + 1) / max(float(args.lr_warmup_steps), 1.0),
            float(args.lr_cosine_min) / float(args.lr),
        )
    progress = min(
        max((step - args.lr_warmup_steps) / max(float(args.lr_cosine_steps), 1.0), 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    minimum = float(args.lr_cosine_min) / float(args.lr)
    return minimum + (1.0 - minimum) * cosine


@torch.no_grad()
def evaluate_gate(
    model: Stage2GatedCorrectionPolicy,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    total = correct = positive = true_positive = false_positive = 0
    for batch in loader:
        batch = to_device(batch, device)
        target = batch["intervene_mask"].long()
        logits = model.gate_logits(batch)
        loss = F.cross_entropy(logits, target)
        prediction = F.softmax(logits, dim=-1)[:, 1] >= float(threshold)
        target_bool = target.bool()
        batch_size = int(target.shape[0])
        loss_sum += float(loss) * batch_size
        total += batch_size
        correct += int((prediction == target_bool).sum())
        positive += int(target_bool.sum())
        true_positive += int((prediction & target_bool).sum())
        false_positive += int((prediction & ~target_bool).sum())
    recall = true_positive / max(positive, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "positive_rate": positive / max(total, 1),
        "threshold": float(threshold),
    }


def save_checkpoint(
    path: Path,
    *,
    model: Stage2GatedCorrectionPolicy,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: dict,
    source_path: Path,
    args: argparse.Namespace,
    data_summary: dict[str, object],
    epoch: int,
    global_step: int,
    best_score: float,
    history: Sequence[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "residual_gate_stage2",
            "gate_label_meaning": "0=use base policy, 1=apply Stage 1 correction",
            "stage1_checkpoint": str(source_path.resolve()),
            "model_config": model.correction_policy.config.to_dict(),
            "gate_config": model.gate_config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "low_dim_mean": np.asarray(source_checkpoint["low_dim_mean"], dtype=np.float32),
            "low_dim_std": np.asarray(source_checkpoint["low_dim_std"], dtype=np.float32),
            "action_clip": np.asarray(source_checkpoint["action_clip"], dtype=np.float32),
            "arm_correction_indices": source_checkpoint.get("arm_correction_indices"),
            "arm_correction_meaning": source_checkpoint.get("arm_correction_meaning"),
            "gripper_correction_meaning": source_checkpoint.get("gripper_correction_meaning"),
            "gate_training_semantics": gate_training_semantics(args),
            "args": vars(args),
            "data_summary": data_summary,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_score": float(best_score),
            "history": list(history),
        },
        path,
    )


def load_stage2_checkpoint(path: Path, device: torch.device):
    checkpoint = load_torch_checkpoint(path, map_location=device)
    if checkpoint.get("stage") != "residual_gate_stage2":
        raise RuntimeError(f"{path} is not a Stage 2 gate checkpoint")
    config = validate_correction_checkpoint(checkpoint, path)
    correction = Stage1CorrectionPolicy(config, initialize_pretrained=False)
    gate_config = Stage2GateConfig.from_dict(checkpoint.get("gate_config", {}))
    model = Stage2GatedCorrectionPolicy(correction, gate_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "residual_gate_stage2.pt"
    last_path = output_dir / "residual_gate_stage2_last.pt"
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else (last_path if args.resume else None)

    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Stage 2 resume checkpoint does not exist: {resume_path}")
        model, resume_checkpoint = load_stage2_checkpoint(resume_path, device)
        previous = resume_checkpoint.get("gate_training_semantics", {})
        current = gate_training_semantics(args)
        mismatch = [key for key in current if previous.get(key) != current[key]]
        if mismatch:
            raise RuntimeError(f"cannot resume with changed Stage 2 semantics: {mismatch}")
        source_checkpoint = resume_checkpoint
        source_path = Path(resume_checkpoint.get("stage1_checkpoint", args.stage1_checkpoint))
    else:
        source_path = Path(args.stage1_checkpoint).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"Stage 1 checkpoint does not exist: {source_path}")
        correction, source_checkpoint = load_stage1_checkpoint(source_path, device=device)
        validate_correction_checkpoint(source_checkpoint, source_path)
        assert_matching_stage1_split(args, source_checkpoint, source_path)
        gate_config = Stage2GateConfig(
            hidden_dim=args.gate_hidden_dim,
            hidden_depth=args.gate_hidden_depth,
        )
        model = Stage2GatedCorrectionPolicy(correction, gate_config).to(device)
        resume_checkpoint = None

    freeze_except_gate(model)
    data_summary = inspect_data(args.data_dir)
    validate_data_identity(data_summary, source_checkpoint, source_path)
    train_loader, val_loader, train_count, val_count, train_positive, val_positive = make_loaders(
        args,
        source_checkpoint,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, object]] = []
    start_epoch = 1
    global_step = 0
    best_score = float("inf")
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        history = list(resume_checkpoint.get("history", []))
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        global_step = int(resume_checkpoint["global_step"])
        best_score = float(resume_checkpoint["best_score"])
        print(f"[RESUME] {resume_path} at epoch={start_epoch} step={global_step}")

    print(
        f"[DATA] train={train_count} positive={train_positive} "
        f"val={val_count} positive={val_positive}"
    )
    print(
        f"[STAGE2] device={device} label=intervene_mask low_dim={model.correction_policy.config.low_dim_mode} "
        f"temporal={model.correction_policy.config.temporal_context}; only gate_head is trainable"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * learning_rate_factor(global_step, args)
            loss, _ = model.gate_loss(
                batch,
                batch["intervene_mask"],
                downsample_negatives=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            optimizer.step()
            batch_size = int(batch["intervene_mask"].shape[0])
            running_loss += float(loss.detach()) * batch_size
            seen += batch_size
            global_step += 1

        validation = evaluate_gate(
            model,
            val_loader,
            device,
            args.gate_eval_threshold,
        )
        train_loss = running_loss / max(seen, 1)
        item: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation": validation,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(item)
        print(
            f"[GATE {epoch:04d}] train={train_loss:.5f} val={validation['loss']:.5f} "
            f"acc={validation['accuracy']:.3f} recall={validation['recall']:.3f} "
            f"precision={validation['precision']:.3f} f1={validation['f1']:.3f}"
        )
        score = validation["loss"] if args.selection_metric == "loss" else -validation["f1"]
        if not math.isfinite(score):
            score = train_loss
        if score < best_score or not best_path.exists():
            best_score = float(score)
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                source_checkpoint=source_checkpoint,
                source_path=source_path,
                args=args,
                data_summary=data_summary,
                epoch=epoch,
                global_step=global_step,
                best_score=best_score,
                history=history,
            )
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            source_checkpoint=source_checkpoint,
            source_path=source_path,
            args=args,
            data_summary=data_summary,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
            history=history,
        )

    history_path = output_dir / f"stage2_history_{int(time.time())}.json"
    history_path.write_text(
        json.dumps({"args": vars(args), "data": data_summary, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] best={best_path} last={last_path} history={history_path}")


if __name__ == "__main__":
    main()
