#!/usr/bin/env python3
"""Train Stage 1 XYZ/RZ arm residual and gripper correction heads."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import GRIPPER_LABEL_MEANING
from .stage1_dataset import (
    LOW_DIM_MODES,
    Stage1CorrectionDataset,
    build_sample_index,
    collate_batch,
    compute_low_dim_stats,
    find_hdf5_files,
    split_refs,
)
from .stage1_model import (
    ARM_CORRECTION_INDICES,
    ARM_CORRECTION_MEANING,
    Stage1CorrectionPolicy,
    Stage1ModelConfig,
    denormalize_arm_residual,
    make_action_clip,
    normalize_arm_residual,
    select_arm_correction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", default="dataset/hil_mini_lawam_vr")
    parser.add_argument("--output-dir", "--output_dir", default="results/hil/mini_lawam_stage1")
    parser.add_argument("--epochs", "--residual-epochs", "--residual_epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--val-fraction", "--val_fraction", type=float, default=0.1)
    parser.add_argument("--split-unit", "--split_unit", choices=("demo", "frame"), default="demo")
    parser.add_argument(
        "--residual-samples",
        "--residual_samples",
        choices=("intervention_only", "all_weighted"),
        default="intervention_only",
        help="Use correction frames only, or all frames with intervention up-weighting.",
    )
    parser.add_argument(
        "--residual-correction-weight",
        "--residual_correction_weight",
        type=float,
        default=5.0,
    )

    model = parser.add_argument_group("correction model")
    model.add_argument(
        "--low-dim-mode",
        "--low_dim_mode",
        choices=LOW_DIM_MODES,
        default="image_bc_xyz_rz_grip",
        help="Default is [base dx,dy,dz,dRz,current gripper] plus both camera views.",
    )
    model.add_argument(
        "--image-encoder",
        "--image_encoder",
        choices=("small_cnn", "resnet18_spatial"),
        default="small_cnn",
    )
    model.add_argument("--image-pretrained", "--image_pretrained", action="store_true")
    model.add_argument("--freeze-image-backbone", "--freeze_image_backbone", action="store_true")
    model.add_argument("--spatial-keypoints", "--spatial_keypoints", type=int, default=32)
    model.add_argument("--temporal-context", "--temporal_context", type=int, default=1)
    model.add_argument(
        "--action-head-type",
        "--action_head_type",
        choices=("deterministic", "gmm"),
        default="deterministic",
    )
    model.add_argument("--num-gmm-modes", "--num_gmm_modes", type=int, default=5)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--weight-decay", "--weight_decay", type=float, default=0.0)
    optimization.add_argument("--lr-warmup-steps", "--lr_warmup_steps", type=int, default=1000)
    optimization.add_argument("--lr-cosine-steps", "--lr_cosine_steps", type=int, default=100000)
    optimization.add_argument("--lr-cosine-min", "--lr_cosine_min", type=float, default=1e-6)
    optimization.add_argument("--grad-clip", "--grad_clip", type=float, default=1.0)
    optimization.add_argument("--gripper-loss-weight", "--gripper_loss_weight", type=float, default=0.01)
    optimization.add_argument(
        "--selection-metric",
        choices=("arm_physical_mae", "loss"),
        default="arm_physical_mae",
    )
    optimization.add_argument(
        "--max-xyz-residual-per-step",
        "--max_xyz_residual_per_step",
        type=float,
        default=0.05,
    )
    optimization.add_argument(
        "--max-rz-residual-per-step",
        "--max_rot_residual_per_step",
        dest="max_rz_residual_per_step",
        type=float,
        default=0.05,
    )

    resume = parser.add_argument_group("resume")
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-checkpoint", "--resume_checkpoint", default=None)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "lr",
        "grad_clip",
        "max_xyz_residual_per_step",
        "max_rz_residual_per_step",
        "residual_correction_weight",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0,1)")
    if args.temporal_context < 1 or args.spatial_keypoints < 1 or args.num_gmm_modes < 1:
        parser.error("temporal context, spatial keypoints, and GMM modes must be positive")
    if not math.isfinite(args.gripper_loss_weight) or args.gripper_loss_weight < 0.0:
        parser.error("--gripper-loss-weight must be finite and non-negative")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        parser.error("--weight-decay must be finite and non-negative")


def load_torch_checkpoint(path: str | Path, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def inspect_data(data_path: str | Path) -> dict[str, object]:
    files = find_hdf5_files(data_path)
    if not files:
        raise RuntimeError(f"no HDF5 correction files found in {data_path}")
    identities: set[tuple[str, str, str]] = set()
    samples = interventions = 0
    labels = np.zeros(2, dtype=np.int64)
    intervention_labels = np.zeros(2, dtype=np.int64)
    max_ignored_rotation = 0.0
    for path in files:
        with h5py.File(path, "r") as file:
            meta = file.get("meta")
            if meta is None:
                raise RuntimeError(f"{path} has no meta group; expected the HIL collector schema")
            identity = (
                str(meta.attrs.get("schema", "")),
                str(meta.attrs.get("base_policy_checkpoint", "")),
                str(meta.attrs.get("base_policy_target_mode", "")),
            )
            identities.add(identity)
            for demo in file.get("data", {}):
                group = file["data"][demo]
                if "residual_targets" not in group:
                    continue
                if "executed_actions" not in group:
                    raise RuntimeError(
                        f"{path}:{demo} has no executed_actions; binary gripper targets "
                        "cannot be recovered"
                    )
                residual = np.asarray(group["residual_targets"], dtype=np.float32)
                mask = np.asarray(group["intervene_mask"], dtype=bool)
                executed = np.asarray(group["executed_actions"], dtype=np.float32)
                if executed.ndim != 2 or executed.shape[1] != 7:
                    raise RuntimeError(
                        f"{path}:{demo} executed_actions has shape {executed.shape}, expected (N,7)"
                    )
                grip = (executed[:, 6] > 0.0).astype(np.int64)
                samples += int(residual.shape[0])
                interventions += int(mask.sum())
                for label in range(2):
                    labels[label] += int((grip == label).sum())
                    intervention_labels[label] += int((grip[mask] == label).sum())
                if residual.size:
                    max_ignored_rotation = max(
                        max_ignored_rotation,
                        float(np.max(np.abs(residual[:, [3, 4]]))),
                    )
    if len(identities) != 1:
        raise RuntimeError(
            "Stage 1 data mixes incompatible action schemas or base policies: "
            f"{sorted(identities)}"
        )
    identity = next(iter(identities))
    if "forward_command_delta" not in identity[0]:
        raise RuntimeError(f"unsupported HIL action schema {identity[0]!r}")
    if interventions == 0:
        raise RuntimeError("dataset contains no VR intervention frames")
    return {
        "files": files,
        "schema": identity[0],
        "base_policy_checkpoint": identity[1],
        "base_policy_target_mode": identity[2],
        "samples": samples,
        "interventions": interventions,
        "gripper_label_counts": labels.tolist(),
        "intervention_gripper_label_counts": intervention_labels.tolist(),
        "max_ignored_rx_ry": max_ignored_rotation,
    }


def make_loaders(
    args: argparse.Namespace,
    *,
    low_dim_mean: Optional[np.ndarray] = None,
    low_dim_std: Optional[np.ndarray] = None,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, int, int]:
    all_refs = build_sample_index(args.data_dir, intervention_only=False)
    if not all_refs:
        raise RuntimeError(f"no Stage 1 samples found in {args.data_dir}")
    train_refs, val_refs = split_refs(
        all_refs,
        val_fraction=args.val_fraction,
        seed=args.seed,
        split_unit=args.split_unit,
    )
    if low_dim_mean is None or low_dim_std is None:
        low_dim_mean, low_dim_std = compute_low_dim_stats(
            args.data_dir,
            train_refs,
            low_dim_mode=args.low_dim_mode,
        )
    if args.residual_samples == "intervention_only":
        correction_keys = {
            (ref.path, ref.demo, ref.step)
            for ref in build_sample_index(args.data_dir, intervention_only=True)
        }
        train_refs = [ref for ref in train_refs if (ref.path, ref.demo, ref.step) in correction_keys]
        val_refs = [ref for ref in val_refs if (ref.path, ref.demo, ref.step) in correction_keys]
    if not train_refs or not val_refs:
        raise RuntimeError(
            "Stage 1 produced an empty train or validation correction split. "
            "Collect interventions in more episodes or use --split-unit frame."
        )
    train_dataset = Stage1CorrectionDataset(
        args.data_dir,
        refs=train_refs,
        low_dim_mode=args.low_dim_mode,
        low_dim_mean=low_dim_mean,
        low_dim_std=low_dim_std,
        temporal_context=args.temporal_context,
    )
    val_dataset = Stage1CorrectionDataset(
        args.data_dir,
        refs=val_refs,
        low_dim_mode=args.low_dim_mode,
        low_dim_mean=low_dim_mean,
        low_dim_std=low_dim_std,
        temporal_context=args.temporal_context,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_batch,
        "pin_memory": str(args.device).startswith("cuda") and torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        low_dim_mean,
        low_dim_std,
        len(train_refs),
        len(val_refs),
    )


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, non_blocking=True) for key, value in batch.items()}


def sample_weight(batch: dict[str, torch.Tensor], args: argparse.Namespace) -> Optional[torch.Tensor]:
    if args.residual_samples == "intervention_only":
        return None
    mask = batch["intervene_mask"].float()
    return 1.0 + mask * (float(args.residual_correction_weight) - 1.0)


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


def compute_loss(
    model: Stage1CorrectionPolicy,
    batch: dict[str, torch.Tensor],
    action_clip: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    arm_target = select_arm_correction(batch["residual_target"])
    normalized = normalize_arm_residual(arm_target, action_clip)
    return model.loss(
        batch,
        normalized,
        batch["gripper_label"],
        sample_weight=sample_weight(batch, args),
        gripper_loss_weight=args.gripper_loss_weight,
    )


@torch.no_grad()
def evaluate(
    model: Stage1CorrectionPolicy,
    loader: DataLoader,
    device: torch.device,
    action_clip: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "arm_loss": 0.0,
        "gripper_ce": 0.0,
        "arm_abs": 0.0,
        "zero_abs": 0.0,
        "physical_abs": 0.0,
        "physical_zero_abs": 0.0,
        "gripper_correct": 0.0,
    }
    sample_count = element_count = grip_count = 0
    cosines: list[float] = []
    for batch in loader:
        batch = to_device(batch, device)
        loss, logs = compute_loss(model, batch, action_clip, args)
        output = model(batch)
        target_physical = select_arm_correction(batch["residual_target"])
        target_norm = normalize_arm_residual(target_physical, action_clip)
        prediction_norm = output["arm_residual_norm"]
        prediction_physical = denormalize_arm_residual(prediction_norm, action_clip)
        batch_size = int(target_norm.shape[0])
        elements = int(target_norm.numel())
        totals["loss"] += float(loss) * batch_size
        totals["arm_loss"] += float(logs["arm_loss"]) * batch_size
        totals["gripper_ce"] += float(logs["gripper_ce"]) * batch_size
        totals["arm_abs"] += float(torch.abs(prediction_norm - target_norm).sum())
        totals["zero_abs"] += float(torch.abs(target_norm).sum())
        totals["physical_abs"] += float(torch.abs(prediction_physical - target_physical).sum())
        totals["physical_zero_abs"] += float(torch.abs(target_physical).sum())
        predicted_grip = output["gripper_logits"].argmax(dim=-1)
        totals["gripper_correct"] += float((predicted_grip == batch["gripper_label"]).sum())
        nonzero = torch.linalg.vector_norm(target_norm, dim=-1) > 1e-8
        if nonzero.any():
            cosines.extend(F.cosine_similarity(prediction_norm[nonzero], target_norm[nonzero]).cpu().tolist())
        sample_count += batch_size
        element_count += elements
        grip_count += batch_size
    physical_mae = totals["physical_abs"] / max(element_count, 1)
    zero_physical_mae = totals["physical_zero_abs"] / max(element_count, 1)
    return {
        "loss": totals["loss"] / max(sample_count, 1),
        "arm_loss": totals["arm_loss"] / max(sample_count, 1),
        "gripper_ce": totals["gripper_ce"] / max(sample_count, 1),
        "arm_norm_mae": totals["arm_abs"] / max(element_count, 1),
        "zero_arm_norm_mae": totals["zero_abs"] / max(element_count, 1),
        "arm_physical_mae": physical_mae,
        "zero_arm_physical_mae": zero_physical_mae,
        "arm_mae_improvement_vs_zero": 1.0 - physical_mae / max(zero_physical_mae, 1e-8),
        "arm_cosine_similarity": float(np.mean(cosines)) if cosines else float("nan"),
        "gripper_accuracy": totals["gripper_correct"] / max(grip_count, 1),
    }


def checkpoint_semantics(args: argparse.Namespace) -> dict[str, object]:
    keys = (
        "low_dim_mode",
        "image_encoder",
        "image_pretrained",
        "freeze_image_backbone",
        "spatial_keypoints",
        "temporal_context",
        "action_head_type",
        "num_gmm_modes",
        "residual_samples",
        "split_unit",
        "gripper_loss_weight",
        "max_xyz_residual_per_step",
        "max_rz_residual_per_step",
    )
    semantics = {key: getattr(args, key) for key in keys}
    semantics["gripper_label_schema"] = "binary_executed_state_v1"
    return semantics


def assert_resume_compatible(args: argparse.Namespace, checkpoint: dict, path: Path) -> None:
    previous = checkpoint.get("training_semantics", {})
    current = checkpoint_semantics(args)
    mismatches = [key for key, value in current.items() if previous.get(key) != value]
    if mismatches:
        detail = ", ".join(f"{key}: {previous.get(key)!r} != {current[key]!r}" for key in mismatches)
        raise RuntimeError(f"cannot resume {path} with changed Stage 1 semantics: {detail}")


def save_checkpoint(
    path: Path,
    *,
    model: Stage1CorrectionPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    low_dim_mean: np.ndarray,
    low_dim_std: np.ndarray,
    action_clip: torch.Tensor,
    data_summary: dict[str, object],
    epoch: int,
    global_step: int,
    best_score: float,
    history: Sequence[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "residual_stage1",
            "arm_correction_indices": ARM_CORRECTION_INDICES,
            "arm_correction_meaning": ARM_CORRECTION_MEANING,
            "gripper_correction_meaning": (
                f"{GRIPPER_LABEL_MEANING}; absolute executed gripper state"
            ),
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "low_dim_mean": low_dim_mean.astype(np.float32),
            "low_dim_std": low_dim_std.astype(np.float32),
            "action_clip": action_clip.detach().cpu().numpy().astype(np.float32),
            "training_semantics": checkpoint_semantics(args),
            "args": vars(args),
            "data_summary": data_summary,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_score": float(best_score),
            "history": list(history),
        },
        path,
    )


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
    best_path = output_dir / "residual_stage1.pt"
    last_path = output_dir / "residual_stage1_last.pt"
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else (last_path if args.resume else None)
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Stage 1 resume checkpoint does not exist: {resume_path}")
        resume_checkpoint = load_torch_checkpoint(resume_path, map_location=device)
        assert_resume_compatible(args, resume_checkpoint, resume_path)

    data_summary = inspect_data(args.data_dir)
    print(
        f"[DATA] files={len(data_summary['files'])} samples={data_summary['samples']} "
        f"interventions={data_summary['interventions']} "
        f"grip_labels(all)={data_summary['gripper_label_counts']} "
        f"grip_labels(intervention)={data_summary['intervention_gripper_label_counts']}"
    )
    print(
        f"[TARGET] arm={ARM_CORRECTION_MEANING} canonical_indices={ARM_CORRECTION_INDICES}; "
        f"gripper={GRIPPER_LABEL_MEANING} (absolute executed state)"
    )
    if float(data_summary["max_ignored_rx_ry"]) > 1e-4:
        print(
            f"[WARN] ignored Rx/Ry residual reaches {data_summary['max_ignored_rx_ry']:.6f}; "
            "Stage 1 intentionally learns only XYZ/RZ"
        )

    low_dim_mean = None
    low_dim_std = None
    if resume_checkpoint is not None:
        low_dim_mean = np.asarray(resume_checkpoint["low_dim_mean"], dtype=np.float32)
        low_dim_std = np.asarray(resume_checkpoint["low_dim_std"], dtype=np.float32)
    train_loader, val_loader, low_dim_mean, low_dim_std, train_count, val_count = make_loaders(
        args,
        low_dim_mean=low_dim_mean,
        low_dim_std=low_dim_std,
    )
    if resume_checkpoint is None:
        config = Stage1ModelConfig(
            low_dim_dim=int(low_dim_mean.shape[0]),
            low_dim_mode=args.low_dim_mode,
            arm_action_dim=4,
            image_encoder=args.image_encoder,
            image_pretrained=args.image_pretrained,
            freeze_image_backbone=args.freeze_image_backbone,
            spatial_keypoints=args.spatial_keypoints,
            temporal_context=args.temporal_context,
            action_head_type=args.action_head_type,
            num_gmm_modes=args.num_gmm_modes,
        )
    else:
        config = Stage1ModelConfig.from_dict(resume_checkpoint["model_config"])
    model = Stage1CorrectionPolicy(config, initialize_pretrained=resume_checkpoint is None).to(device)
    action_clip = make_action_clip(
        args.max_xyz_residual_per_step,
        args.max_rz_residual_per_step,
        device=device,
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        action_clip = torch.as_tensor(resume_checkpoint["action_clip"], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        f"[TRAIN] device={device} train={train_count} val={val_count} "
        f"low_dim={args.low_dim_mode}:{config.low_dim_dim} temporal={config.temporal_context}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * learning_rate_factor(global_step, args)
            loss, _ = compute_loss(model, batch, action_clip, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = int(batch["gripper_label"].shape[0])
            running_loss += float(loss.detach()) * batch_size
            seen += batch_size
            global_step += 1

        validation = evaluate(model, val_loader, device, action_clip, args)
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
            f"[STAGE1 {epoch:04d}] train={train_loss:.5f} val={validation['loss']:.5f} "
            f"arm={validation['arm_physical_mae']:.6f}m/rad "
            f"gain0={validation['arm_mae_improvement_vs_zero']:.1%} "
            f"cos={validation['arm_cosine_similarity']:.3f} "
            f"grip_acc={validation['gripper_accuracy']:.3f}"
        )
        score = validation[args.selection_metric]
        if not math.isfinite(score):
            score = train_loss
        if score < best_score or not best_path.exists():
            best_score = float(score)
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                args=args,
                low_dim_mean=low_dim_mean,
                low_dim_std=low_dim_std,
                action_clip=action_clip,
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
            args=args,
            low_dim_mean=low_dim_mean,
            low_dim_std=low_dim_std,
            action_clip=action_clip,
            data_summary=data_summary,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
            history=history,
        )

    history_path = output_dir / f"stage1_history_{int(time.time())}.json"
    history_path.write_text(
        json.dumps({"args": vars(args), "data": data_summary, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] best={best_path} last={last_path} history={history_path}")


if __name__ == "__main__":
    main()
