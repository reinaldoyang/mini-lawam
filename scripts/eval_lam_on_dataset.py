"""Stage-1 sanity check: is the released LaWM/LAM good enough on my dataset?

Reads an IsaacLab/RoboMimic-style HDF5 directly, runs the frozen LAM on
(o_t, o_{t+gap}) frame pairs, and reports three diagnostics per frame gap:

  1. rollout_vs_gt = cos(u_hat, u_T)   -- forward-prediction accuracy   (want HIGH)
  2. init_vs_gt    = cos(u_t,   u_T)   -- non-triviality baseline        (want < rollout, and < ~0.95)
  3. shuffled_vs_gt= cos(dec(u_t, z_shuffled), u_T) -- is z actually used? (want << rollout)

Reading of the numbers:
  - rollout_vs_gt clearly beats init_vs_gt      -> LaWM predicts real dynamics, not a copy.
  - rollout_vs_gt clearly beats shuffled_vs_gt  -> the latent action z carries the transition.
  - init_vs_gt well below 1.0                   -> the chosen gap spans real motion.
If all three hold at some gap, the released Stage-1 is good enough; go to Stage 2.

Preprocessing reuses the repo's own gpu_two_view_video_aug (eval mode), so it
matches LAM training/inference exactly.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from latent_action_model.core.lam_model import load_latent_action_model
from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug, LAM_IMAGE_HW

# Reuse the repo's Fig-8 visualization primitives.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "eval_utils"))
from similarity_video import (  # noqa: E402
    extract_anchor_feature,
    render_similarity_overlay,
    render_pca_image,
    fit_token_rgb_projection,
    save_video,
)

GRID_HW = (LAM_IMAGE_HW[0] // 16, LAM_IMAGE_HW[1] // 16)  # (16, 16)


def _cos_tokens(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-token cosine over feature dim. a,b: [B,1,K,D] or [B,K,D]."""
    if a.dim() == 4:
        a = a[:, 0]
    if b.dim() == 4:
        b = b[:, 0]
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


def _load_pair_batch(f, samples, resize):
    """samples: list of (demo_key, t, gap). Returns uint8 [N,2,3,256,256]."""
    clips = []
    for demo, t, gap in samples:
        cam = f["data"][demo]["obs"]["table_cam"]  # [T,H,W,3] uint8
        pair = cam[[t, t + gap]]  # [2,H,W,3]
        x = torch.from_numpy(np.ascontiguousarray(pair)).permute(0, 3, 1, 2)  # [2,3,H,W]
        x = resize(x)  # [2,3,256,256], uint8
        clips.append(x)
    return torch.stack(clips, dim=0).to(torch.uint8)  # [N,2,3,256,256]


def _plan_samples(f, gap, n, rng):
    """Sample (demo, t, gap) triples where t and t+gap are both valid."""
    demos = list(f["data"].keys())
    plan = []
    tries = 0
    while len(plan) < n and tries < n * 50:
        tries += 1
        demo = demos[rng.integers(len(demos))]
        T = int(f["data"][demo]["obs"]["table_cam"].shape[0])
        if T <= gap + 1:
            continue
        t = int(rng.integers(0, T - gap))
        plan.append((demo, t, gap))
    return plan


def _region_cos(a, u_T, idx):
    """Mean cosine(a, u_T) over the per-sample motion tokens in idx. a,u_T:[B,1,K,D]."""
    ca = F.cosine_similarity(a[:, 0].float(), u_T[:, 0].float(), dim=-1)  # [B,K]
    return ca.gather(1, idx).mean().item()


@torch.no_grad()
def eval_gap(lam, f, gap, n, batch, device, resize, rng, topk=24):
    plan = _plan_samples(f, gap, n, rng)
    if not plan:
        return None
    roll, init, shuf = [], [], []
    roll_r, init_r, shuf_r = [], [], []
    for i in range(0, len(plan), batch):
        chunk = plan[i:i + batch]
        clips_u8 = _load_pair_batch(f, chunk, resize)  # [B,2,3,256,256]
        # Exact training-time preprocessing (eval mode: resize already done -> normalize only).
        vids, _ = gpu_two_view_video_aug(clips_u8.to(device), training=False)
        out = lam.get_latent_action(
            videos=vids, states=None, dec_videos=vids, predict_future_frame=True,
        )
        u_t = out["dec_in"]      # [B,1,256,768]
        u_T = out["tgt"]         # [B,1,256,768]
        u_hat = out["recon"]     # [B,1,256,768]
        z = out["quantized"]     # [B,1,32]

        roll.append(_cos_tokens(u_hat, u_T))
        init.append(_cos_tokens(u_t, u_T))

        # Motion region = the topk patches that change most between now and future
        # (lowest cosine(u_t, u_T)). This is where the arm/objects actually move.
        c_gt_tok = F.cosine_similarity(u_t[:, 0].float(), u_T[:, 0].float(), dim=-1)  # [B,K]
        idx = c_gt_tok.argsort(dim=1)[:, :topk]  # [B,topk]
        roll_r.append(_region_cos(u_hat, u_T, idx))
        init_r.append(_region_cos(u_t, u_T, idx))

        # shuffled-z control: decode current feature with someone else's latent action.
        if z.shape[0] > 1:
            perm = torch.randperm(z.shape[0], device=z.device)
            u_hat_shuf = lam.decoder(u_t, z[perm])
            if isinstance(u_hat_shuf, tuple):
                u_hat_shuf = u_hat_shuf[0]
            shuf.append(_cos_tokens(u_hat_shuf, u_T))
            shuf_r.append(_region_cos(u_hat_shuf, u_T, idx))
    return {
        "n": len(plan),
        "rollout_vs_gt": float(np.mean(roll)),
        "init_vs_gt": float(np.mean(init)),
        "shuffled_vs_gt": float(np.mean(shuf)) if shuf else float("nan"),
        "rollout_region": float(np.mean(roll_r)),
        "init_region": float(np.mean(init_r)),
        "shuffled_region": float(np.mean(shuf_r)) if shuf_r else float("nan"),
    }


@torch.no_grad()
def dump_heatmaps(lam, f, gap, num, device, resize, rng, out_dir,
                  vmin=0.3, vmax=0.9, alpha=0.55, cmap="jet"):
    """Fig-8-style panels: current frame | (arm-anchor -> subgoal) similarity | subgoal PCA."""
    import imageio.v2 as imageio

    os.makedirs(out_dir, exist_ok=True)
    plan = _plan_samples(f, gap, num, rng)[:num]
    clips_u8 = _load_pair_batch(f, plan, resize)          # [N,2,3,256,256] uint8
    cur_imgs = clips_u8[:, 0].permute(0, 2, 3, 1).numpy()  # [N,256,256,3] uint8 (o_t)
    vids, _ = gpu_two_view_video_aug(clips_u8.to(device), training=False)
    out = lam.get_latent_action(
        videos=vids, states=None, dec_videos=vids, predict_future_frame=True,
    )
    u_t = out["dec_in"][:, 0]     # [N,256,768]  current features
    u_T = out["tgt"][:, 0]        # [N,256,768]  true future features
    u_hat = out["recon"][:, 0]    # [N,256,768]  predicted subgoal features

    for i in range(u_t.shape[0]):
        toks_t = u_t[i].float().cpu()
        toks_T = u_T[i].float().cpu()
        toks_hat = u_hat[i].float().cpu()
        # Auto-pick the arm patch = the token that changes most (lowest cos to future).
        c = F.cosine_similarity(toks_t, toks_T, dim=-1)  # [256]
        flat = int(c.argmin().item())
        r, col = divmod(flat, GRID_HW[1])

        anchor = extract_anchor_feature(toks_t, r, col, GRID_HW)  # [768]
        # Similarity of that arm patch to every patch of the PREDICTED SUBGOAL.
        overlay = render_similarity_overlay(
            toks_hat, anchor, GRID_HW, cur_imgs[i],
            vmin=vmin, vmax=vmax, alpha=alpha, cmap=cmap,
        )
        proj = fit_token_rgb_projection([toks_hat])
        pca = render_pca_image(toks_hat, proj, GRID_HW, target_hw=(256, 256))
        panel = np.concatenate([cur_imgs[i], overlay, pca], axis=1)  # [256,768,3]
        path = os.path.join(out_dir, f"heatmap_gap{gap}_{i:02d}_patch{r}-{col}.png")
        imageio.imwrite(path, panel)
    print(f"[heatmaps] wrote {u_t.shape[0]} panels to {out_dir}/ "
          f"(left=o_t | mid=arm-anchor->subgoal similarity | right=subgoal PCA)")


def _draw_patch_box(img, r, c, grid_hw, color=(0, 255, 0), thick=2):
    """Draw a box on uint8 [H,W,3] at grid patch (r=row, c=col)."""
    ph = img.shape[0] // grid_hw[0]
    pw = img.shape[1] // grid_hw[1]
    y0, x0, y1, x1 = r * ph, c * pw, r * ph + ph, c * pw + pw
    img[y0:y0 + thick, x0:x1] = color
    img[y1 - thick:y1, x0:x1] = color
    img[y0:y1, x0:x0 + thick] = color
    img[y0:y1, x1 - thick:x1] = color
    return img


@torch.no_grad()
def dump_sequence(lam, f, demo, gap, stride, device, resize, out_dir,
                  anchor_mode="auto", manual_anchor=None,
                  vmin=0.3, vmax=0.9, alpha=0.55, cmap="jet", fps=2):
    """Fig-8-style sequence over ONE episode: chunk-by-chunk arm->subgoal panels.

    anchor_mode: 'auto' picks the arm patch fresh each chunk (follows the arm);
                 'first' fixes the arm feature from chunk 0 and tracks where it
                 migrates in every later subgoal (paper-style).
    """
    import imageio.v2 as imageio

    os.makedirs(out_dir, exist_ok=True)
    demos = list(f["data"].keys())
    if isinstance(demo, int) or str(demo).isdigit():
        demo_key = demos[int(demo)] if int(demo) < len(demos) else f"demo_{demo}"
    else:
        demo_key = demo
    if demo_key not in f["data"]:
        raise KeyError(f"demo '{demo_key}' not found; have e.g. {demos[:3]} ...")

    cam = f["data"][demo_key]["obs"]["table_cam"]
    T = int(cam.shape[0])
    starts = list(range(0, T - gap, stride))
    if not starts:
        print(f"[sequence] {demo_key} too short (T={T}) for gap={gap}")
        return

    samples = [(demo_key, t, gap) for t in starts]
    clips_u8 = _load_pair_batch(f, samples, resize)          # [C,2,3,256,256]
    cur_imgs = clips_u8[:, 0].permute(0, 2, 3, 1).numpy()     # [C,256,256,3]
    vids, _ = gpu_two_view_video_aug(clips_u8.to(device), training=False)
    out = lam.get_latent_action(
        videos=vids, states=None, dec_videos=vids, predict_future_frame=True,
    )
    u_t = out["dec_in"][:, 0].float().cpu()   # [C,256,768]
    u_T = out["tgt"][:, 0].float().cpu()
    u_hat = out["recon"][:, 0].float().cpu()

    def _auto_rc(i):
        c = F.cosine_similarity(u_t[i], u_T[i], dim=-1)
        return divmod(int(c.argmin().item()), GRID_HW[1])

    fixed_feat, fixed_rc = None, None
    if anchor_mode == "first" or manual_anchor is not None:
        r0, c0 = manual_anchor if manual_anchor is not None else _auto_rc(0)
        fixed_rc = (r0, c0)
        fixed_feat = extract_anchor_feature(u_t[0], r0, c0, GRID_HW)

    # Shared PCA projection across the whole sequence -> consistent colors.
    proj = fit_token_rgb_projection([u_hat[i] for i in range(u_hat.shape[0])])

    frames = []
    for i in range(u_t.shape[0]):
        if fixed_feat is not None:
            anchor, (r, c) = fixed_feat, fixed_rc
        else:
            r, c = _auto_rc(i)
            anchor = extract_anchor_feature(u_t[i], r, c, GRID_HW)
        img_box = _draw_patch_box(cur_imgs[i].copy(), r, c, GRID_HW)
        overlay = render_similarity_overlay(
            u_hat[i], anchor, GRID_HW, cur_imgs[i],
            vmin=vmin, vmax=vmax, alpha=alpha, cmap=cmap,
        )
        pca = render_pca_image(u_hat[i], proj, GRID_HW, target_hw=(256, 256))
        frames.append(np.concatenate([img_box, overlay, pca], axis=1))

    tag = f"{demo_key}_gap{gap}_{anchor_mode}"
    png = os.path.join(out_dir, f"sequence_{tag}.png")
    imageio.imwrite(png, np.concatenate(frames, axis=0))  # vertical filmstrip
    anim = None
    gif = os.path.join(out_dir, f"sequence_{tag}.gif")
    try:  # GIF via pillow -> no video codec dependency
        imageio.mimsave(gif, frames, duration=1.0 / max(fps, 1), loop=0)
        anim = gif
    except Exception as e:  # noqa: BLE001
        print(f"[sequence] animated output skipped ({type(e).__name__}: {e})")
    print(f"[sequence] {demo_key}: {len(frames)} chunks (anchor={anchor_mode}) "
          f"-> {png}" + (f" and {anim}" if anim else " (PNG only)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="dataset/demo_dataset_100.hdf5")
    ap.add_argument("--ckpt", default="latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt")
    ap.add_argument("--yaml", default="latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml")
    ap.add_argument("--gaps", type=int, nargs="+", default=[16, 24, 32, 48],
                    help="Frame gaps to sweep. 20Hz control => tau=1.6s ~ 32 frames.")
    ap.add_argument("--num-pairs", type=int, default=256, help="Pairs sampled per gap.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--topk", type=int, default=24, help="# motion patches for region metric.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-heatmaps", type=int, default=0,
                    help="If >0, write this many Fig-8-style heatmap panels.")
    ap.add_argument("--heatmap-gap", type=int, default=32, help="Frame gap for heatmaps.")
    ap.add_argument("--out-dir", default="results/lam_check/heatmaps")
    ap.add_argument("--sequence", default=None,
                    help="Demo id (index or 'demo_N'): dump Fig-8 sequence over one episode.")
    ap.add_argument("--seq-stride", type=int, default=0,
                    help="Chunk stride in frames (default 0 -> use heatmap-gap = non-overlapping).")
    ap.add_argument("--seq-anchor", choices=["auto", "first"], default="auto",
                    help="'auto': re-pick arm each chunk; 'first': track chunk-0 arm feature.")
    ap.add_argument("--anchor", type=int, nargs=2, default=None, metavar=("ROW", "COL"),
                    help="Manual anchor patch (row col) on the 16x16 grid; overrides auto.")
    ap.add_argument("--anchor-file", default=None,
                    help="JSON from scripts/pick_anchor.py with {row, col}; sets --anchor.")
    args = ap.parse_args()

    if args.anchor is None and args.anchor_file:
        import json
        with open(args.anchor_file) as _f:
            _a = json.load(_f)
        args.anchor = [int(_a["row"]), int(_a["col"])]
        print(f"[anchor] loaded row={args.anchor[0]} col={args.anchor[1]} from {args.anchor_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    resize = v2.Resize(LAM_IMAGE_HW, antialias=True)

    print(f"Loading LAM on {device} ...")
    lam = load_latent_action_model(args.ckpt, args.yaml).to(device).eval()

    f = h5py.File(args.hdf5, "r")
    print(f"\nHDF5: {args.hdf5}  |  {len(f['data'].keys())} demos")
    print("view=table_cam  resize=256x256  (20Hz: tau=1.6s ~= 32 frames)\n")
    print("=== WHOLE-FRAME (mean over all 256 tokens) ===")
    header = f"{'gap':>5} {'~sec':>6} {'rollout_vs_gt':>14} {'init_vs_gt':>11} {'shuffled_vs_gt':>15} {'roll-init':>10}"
    print(header)
    print("-" * len(header))
    results = {}
    for gap in args.gaps:
        r = eval_gap(lam, f, gap, args.num_pairs, args.batch, device, resize, rng, topk=args.topk)
        results[gap] = r
        if r is None:
            print(f"{gap:>5}  (no valid pairs)")
            continue
        delta = r["rollout_vs_gt"] - r["init_vs_gt"]
        print(f"{gap:>5} {gap/20.0:>6.2f} {r['rollout_vs_gt']:>14.4f} "
              f"{r['init_vs_gt']:>11.4f} {r['shuffled_vs_gt']:>15.4f} {delta:>10.4f}")

    print(f"\n=== MOTION-REGION (mean over top-{args.topk} moving patches only) ===")
    print(header)
    print("-" * len(header))
    for gap in args.gaps:
        r = results[gap]
        if r is None:
            continue
        delta = r["rollout_region"] - r["init_region"]
        print(f"{gap:>5} {gap/20.0:>6.2f} {r['rollout_region']:>14.4f} "
              f"{r['init_region']:>11.4f} {r['shuffled_region']:>15.4f} {delta:>10.4f}")

    if args.dump_heatmaps > 0:
        print()
        dump_heatmaps(lam, f, args.heatmap_gap, args.dump_heatmaps,
                      device, resize, rng, args.out_dir)
    if args.sequence is not None:
        print()
        stride = args.seq_stride if args.seq_stride > 0 else args.heatmap_gap
        dump_sequence(lam, f, args.sequence, args.heatmap_gap, stride,
                      device, resize, args.out_dir,
                      anchor_mode=args.seq_anchor,
                      manual_anchor=tuple(args.anchor) if args.anchor else None)
    f.close()

    print("\nHow to read:")
    print("  Use the MOTION-REGION table -- whole-frame is diluted by static background.")
    print("  rollout > init  (roll-init large)  -> LaWM predicts the actual motion")
    print("  rollout >> shuffled               -> latent action z drives the prediction")
    print("  If both hold at some gap -> Stage 1 is good enough; proceed to Stage 2.")


if __name__ == "__main__":
    main()
