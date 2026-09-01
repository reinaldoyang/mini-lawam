"""Visualize subgoals reconstructed by the released pretrained LAM.

Unlike ``mini_lawam.viz_subgoal``, this evaluator does not require a trained
ConvPrior checkpoint. It gives the pretrained inverse dynamics model the pair
``(o_t, o_T)``, then visualizes the forward-model rollout
``decoder(u_t, IDM(u_t, u_T))`` against the true future features ``u_T``.

Only the requested LeRobot frames are decoded from the shared video shards.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.viz_pretrained_subgoal \
        --lerobot dataset/0827_cardboard_box_50 \
        --episode 0 --camera observation.images.cam_high --t 40 80 120
"""

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.transforms import v2

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.video_aug import LAM_IMAGE_HW, gpu_two_view_video_aug
from mini_lawam.lerobot_video import LeRobotEpisodeSource
from mini_lawam.viz_subgoal import joint_pca_rgb, to_grid

DEFAULT_LAM_CKPT = (
    "latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
)
DEFAULT_LAM_YAML = (
    "latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"
)
DEFAULT_CAMERA = "observation.images.cam_high"


def _load_frame_dt_sec(config_path: str | os.PathLike[str]) -> float:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"LAM config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config: Any = yaml.safe_load(handle)
    try:
        frame_dt_sec = float(config["data"]["frame_dt_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing positive data.frame_dt_sec in LAM config: {path}") from exc
    if not np.isfinite(frame_dt_sec) or frame_dt_sec <= 0:
        raise ValueError(
            f"data.frame_dt_sec must be positive in {path}, got {frame_dt_sec}"
        )
    return frame_dt_sec


def _prepare_frame(image: np.ndarray, resize: v2.Resize) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return resize(tensor).to(torch.uint8)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot", required=True, help="local LeRobot v3 dataset")
    parser.add_argument("--episode", type=int, default=None,
                        help="episode_index; default = first episode")
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="LeRobot video feature used as the main view")
    parser.add_argument("--t", type=int, nargs="+", default=[0],
                        help="one or more episode-relative start frames")
    parser.add_argument("--ckpt", default=DEFAULT_LAM_CKPT,
                        help="released pretrained LAM checkpoint")
    parser.add_argument("--yaml", default=DEFAULT_LAM_YAML,
                        help="LAM model/data configuration")
    parser.add_argument("--gap", type=int, default=None,
                        help="future-frame gap; default = round(frame_dt_sec * dataset FPS)")
    parser.add_argument("--out-dir", default="results/lam_check/pretrained_subgoal_viz")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.dpi < 1:
        raise ValueError("--dpi must be >= 1")
    if args.gap is not None and args.gap < 1:
        raise ValueError("--gap must be >= 1")

    source = LeRobotEpisodeSource.open(
        args.lerobot,
        episode_index=args.episode,
        camera_keys=[args.camera],
    )
    frame_dt_sec = _load_frame_dt_sec(args.yaml)
    gap = args.gap if args.gap is not None else max(1, round(frame_dt_sec * source.fps))
    effective_dt_sec = gap / source.fps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pretrained LAM on {device} ...")
    lam = load_latent_action_model(args.ckpt, args.yaml).to(device).eval()
    resize = v2.Resize(LAM_IMAGE_HW, antialias=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"dataset: {source.root} | episode={source.episode_index} | "
        f"camera={args.camera} | fps={source.fps:g}"
    )
    print(
        f"pretrained interval={frame_dt_sec:g}s | gap/horizon={gap} frames "
        f"({effective_dt_sec:g}s)"
    )

    sample_name = f"episode_{source.episode_index:06d}"
    for t in args.t:
        if t < 0:
            print(f"skip t={t}: frame index must be non-negative")
            continue
        if t + gap >= source.length:
            print(f"skip t={t}: t+{gap} beyond episode length {source.length}")
            continue

        raw_t, raw_T = source.frames(args.camera, [t, t + gap])
        image_t = _prepare_frame(raw_t, resize)
        image_T = _prepare_frame(raw_T, resize)
        frames = torch.stack([image_t, image_T])[None].to(device)
        videos, _ = gpu_two_view_video_aug(frames, training=False)

        output = lam.get_latent_action(
            videos=videos,
            states=None,
            dec_videos=videos,
            predict_future_frame=True,
        )
        u_t = output["dec_in"][0, 0]
        u_T = output["tgt"][0, 0]
        u_hat = output["recon"][0, 0]

        pca_t, pca_hat, pca_T = joint_pca_rgb([u_t, u_hat, u_T])
        change_hat = to_grid((u_hat - u_t).norm(dim=-1))
        change_true = to_grid((u_T - u_t).norm(dim=-1))
        agreement = to_grid(F.cosine_similarity(u_hat, u_T, dim=-1))
        vmax = max(float(change_hat.max()), float(change_true.max()))
        cos_full = float(F.cosine_similarity(u_hat.flatten(), u_T.flatten(), dim=0))

        rgb_t = np.asarray(image_t.permute(1, 2, 0))
        rgb_T = np.asarray(image_T.permute(1, 2, 0))
        panels = [
            (rgb_t, None, f"o_t ({sample_name}, t={t})"),
            (rgb_T, None, f"true o_T (t={t + gap})"),
            (change_hat, "change", "pretrained LAM rollout change"),
            (change_true, "change", "true feature change"),
            (pca_t, None, "PCA(u_t)"),
            (pca_hat, None, "PCA(u_hat_T) — reconstructed subgoal"),
            (pca_T, None, "PCA(u_T) — true future"),
            (agreement, "cos", f"agreement (full cos={cos_full:.3f})"),
        ]

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for axis, (image, kind, title) in zip(axes.flat, panels):
            if kind is None:
                axis.imshow(image)
            elif kind == "cos":
                plot = axis.imshow(image, cmap="viridis", vmin=0.0, vmax=1.0)
                fig.colorbar(plot, ax=axis, fraction=0.046)
            else:
                axis.imshow(rgb_t, extent=(0, 16, 16, 0))
                plot = axis.imshow(
                    image,
                    cmap="jet",
                    alpha=0.55,
                    vmin=0.0,
                    vmax=vmax,
                    extent=(0, 16, 16, 0),
                )
                fig.colorbar(plot, ax=axis, fraction=0.046)
            axis.set_title(title, fontsize=10)
            axis.axis("off")

        fig.suptitle(
            f"Pretrained LAM subgoal reconstruction — {sample_name}, "
            f"t={t}, gap={gap} ({effective_dt_sec:g}s)",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.4, w_pad=1.2)
        output_path = os.path.join(
            args.out_dir, f"{sample_name}_t{t}_gap{gap}_pretrained.png"
        )
        fig.savefig(output_path, dpi=args.dpi)
        plt.close(fig)
        print(f"saved {output_path} (full-frame cos={cos_full:.3f})")


if __name__ == "__main__":
    main()
