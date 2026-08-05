"""Live-camera vs dataset check: is the table_cam view still in-distribution?

Answers, in ~10 seconds, the question that decides whether a rollout is worth
starting: does the LIVE table_cam image still look (to the policy's own DINO
features) like the recorded dataset, and does the policy's step-0 prediction at
the home pose land near home?

Three signals:
  1. blend overlay   : live frame blended 50/50 with a recorded start frame ->
                       physically re-aim the camera until edges line up.
  2. feature cosine  : cos(DINO(live), DINO(dataset start frames)), judged
                       against the dataset's own start-vs-start spread
                       (in-distribution baseline).
  3. step-0 gate     : policy.act(live)[0] vs the demos' home position.
                       (>~5 cm at the home pose = do NOT start a rollout.)

GUI mode (default; needs display + camera):
    python -m mini_lawam.check_camera --table-cam-serial 244422300964
        keys: Q quit | S save snapshot PNG | R cycle reference demo

One-shot mode (no GUI; prints verdict, saves a PNG report):
    python -m mini_lawam.check_camera --table-cam-serial 244422300964 --once
Self-test without a camera (fake the live frame from the dataset):
    python -m mini_lawam.check_camera --fake-live demo_10:0 --once
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from mini_lawam.rollout import MiniLaWAMPolicy


def token_feats(policy: MiniLaWAMPolicy, frame_hwc_u8: np.ndarray) -> torch.Tensor:
    """Frame -> DINO tokens [K, D] via the policy's own preprocessing."""
    o = policy.preprocess(frame_hwc_u8)
    with torch.no_grad():
        return policy.model._feat(o)[0, 0]  # [K, D]


def fcos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten()[None], b.flatten()[None]).item())


def load_refs(hdf5_path: str, n: int):
    """n demo start frames spread over the dataset + the shared home position."""
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        sel = demos[:: max(1, len(demos) // n)][:n]
        frames = [f["data"][d]["obs"]["table_cam"][0][:] for d in sel]
        home = f["data"][sel[0]]["obs"]["eef_pos_base"][0].astype(np.float64)
    return sel, frames, home


def load_wrist_refs(hdf5_path: str, ref_names) -> list:
    """wrist_cam start frames for the SAME demos used as table refs (or None)."""
    with h5py.File(hdf5_path, "r") as f:
        if "wrist_cam" not in f["data"][ref_names[0]]["obs"]:
            return None
        return [f["data"][d]["obs"]["wrist_cam"][0][:] for d in ref_names]


def feature_cos_vs_refs(policy, live, ref_feats):
    """max cosine of DINO(live) against a list of reference token maps -> (cos, idx)."""
    lf = token_feats(policy, live)
    sims = [fcos(lf, rf) for rf in ref_feats]
    return max(sims), int(np.argmax(sims))


def resize_hw(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def evaluate(policy, live, ref_feats, baseline_min, home_xyz, wrist_live=None,
             action_scale=0.3):
    lf = token_feats(policy, live)
    sims = [fcos(lf, rf) for rf in ref_feats]
    live_cos, best_ref = max(sims), int(np.argmax(sims))
    # act() requires the wrist frame when the checkpoint uses the wrist view, and
    # current eef xyz for use_state/delta checkpoints (robot sits at HOME here).
    need_xyz = (getattr(policy.cfg, "use_state", False)
                or getattr(policy.cfg, "target_mode", "abs") == "delta")
    chunk = policy.act(live, wrist_live,
                       state_xyz=home_xyz if need_xyz else None)
    if getattr(policy.cfg, "target_mode", "abs") == "joystick":
        # policy.act() returns the unscaled raw joystick command. Show/check the
        # absolute step-0 target that rollout would execute from the home pose.
        pred0 = home_xyz + float(action_scale) * chunk[0, :3]
    else:
        pred0 = chunk[0, :3]
    gap_cm = float(np.linalg.norm(pred0 - home_xyz)) * 100.0
    feat_ok = live_cos >= baseline_min
    pred_ok = gap_cm <= 5.0
    return {
        "live_cos": live_cos, "best_ref": best_ref, "sims": sims,
        "pred0": pred0, "grip0": float(chunk[0, -1]), "gap_cm": gap_cm,
        "feat_ok": feat_ok, "pred_ok": pred_ok, "ok": feat_ok and pred_ok,
    }


def report_text(r, baseline_min, baseline_mean, home_xyz, ref_names):
    v = "IN-DISTRIBUTION  -> rollout OK" if r["ok"] else "OUT-OF-DISTRIBUTION -> fix camera/scene first"
    return [
        f"feature cos(live, dataset) = {r['live_cos']:.4f}   "
        f"(baseline: dataset starts vs each other min={baseline_min:.4f} mean={baseline_mean:.4f})"
        f"   {'OK' if r['feat_ok'] else 'LOW'}",
        f"step-0 gate: pred={np.round(r['pred0'], 3)} grip={r['grip0']:+.2f} vs home={np.round(home_xyz, 3)}"
        f"  gap={r['gap_cm']:.1f} cm   {'OK (<=5cm)' if r['pred_ok'] else 'FAR (>5cm)'}",
        f"closest reference: {ref_names[r['best_ref']]}",
        f"VERDICT: {v}",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/mini_lawam/ckpt_phase2.pt")
    ap.add_argument("--hdf5", default="dataset/multi_egg_114ep.hdf5")
    ap.add_argument("--table-cam-serial", default="", help="RealSense serial (as in rollout_ur7e)")
    ap.add_argument("--wrist-cam-serial", default="",
                    help="Also score the wrist feed vs dataset wrist start-frames. "
                         "Answers: is the wrist view more in-distribution than the table "
                         "view? (feature cos only; the action head mixes both views.)")
    ap.add_argument("--n-refs", type=int, default=8, help="dataset start frames to compare against")
    ap.add_argument("--fake-live", default=None, metavar="DEMO:T",
                    help="use a dataset frame as the 'live' frame (self-test, no camera)")
    ap.add_argument("--once", action="store_true", help="single check, print + save PNG, no GUI")
    ap.add_argument("--action-scale", type=float, default=0.3,
                    help="Joystick checkpoint only: scale used to convert the raw "
                         "step-0 command into the rollout target (default: 0.3).")
    ap.add_argument("--out-dir", default="results/mini_lawam/camera_check")
    ap.add_argument("--train-frame-hw", type=int, nargs=2, default=[168, 224],
                    metavar=("H", "W"),
                    help="Match rollout_ur7e: recorded training resolution. Use 0 0 "
                         "for native-256 checkpoints (multi_egg_30_moved_256).")
    # per-camera exposure/gain (0.1 ms units), matching rollout_ur7e. None = auto.
    ap.add_argument("--table-exposure", type=float, default=None)
    ap.add_argument("--table-gain", type=float, default=None)
    ap.add_argument("--wrist-exposure", type=float, default=None)
    ap.add_argument("--wrist-gain", type=float, default=None)
    args = ap.parse_args()
    if not np.isfinite(args.action_scale) or args.action_scale < 0.0:
        raise ValueError("--action-scale must be a finite value >= 0")

    train_hw = None if args.train_frame_hw[0] <= 0 else tuple(args.train_frame_hw)
    policy = MiniLaWAMPolicy(args.ckpt, train_frame_hw=train_hw)
    ref_names, ref_frames, home_xyz = load_refs(args.hdf5, args.n_refs)
    ref_feats = [token_feats(policy, fr) for fr in ref_frames]

    # In-distribution baseline: how similar are the dataset's OWN start frames?
    pair = [fcos(ref_feats[i], ref_feats[j])
            for i in range(len(ref_feats)) for j in range(i + 1, len(ref_feats))]
    baseline_min, baseline_mean = float(np.min(pair)), float(np.mean(pair))
    print(f"refs: {len(ref_frames)} start frames | baseline start-vs-start cos: "
          f"min={baseline_min:.4f} mean={baseline_mean:.4f} | home={np.round(home_xyz, 3)}")

    # ---- optional wrist reference baseline ----
    wrist_feats = wrist_baseline_min = None
    if args.wrist_cam_serial:
        wrist_frames = load_wrist_refs(args.hdf5, ref_names)
        if wrist_frames is None:
            print("[wrist] dataset has no wrist_cam obs -> skipping wrist check")
        else:
            wrist_feats = [token_feats(policy, fr) for fr in wrist_frames]
            wpair = [fcos(wrist_feats[i], wrist_feats[j])
                     for i in range(len(wrist_feats)) for j in range(i + 1, len(wrist_feats))]
            wrist_baseline_min, wrist_baseline_mean = float(np.min(wpair)), float(np.mean(wpair))
            print(f"wrist refs: {len(wrist_feats)} | baseline wrist start-vs-start cos: "
                  f"min={wrist_baseline_min:.4f} mean={wrist_baseline_mean:.4f}")

    # ---- live source ----
    reader = None
    if args.fake_live:
        demo, t = args.fake_live.split(":")
        with h5py.File(args.hdf5, "r") as f:
            fake = f["data"][demo]["obs"]["table_cam"][int(t)][:]
        get_live = lambda: fake  # noqa: E731
        print(f"[fake-live] using {demo} t={t} as the live frame")
    else:
        from mini_lawam.rollout_ur7e import RealSenseTableReader
        reader = RealSenseTableReader(args.table_cam_serial, name="table_cam",
                                      exposure=args.table_exposure, gain=args.table_gain)
        reader.start()
        get_live = reader.read_rgb

    os.makedirs(args.out_dir, exist_ok=True)

    def save_report(live, r, tag=""):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Show EXACTLY what the policy ingests: the 256x256 model input (after the
        # --train-frame-hw downscale), for both live and the dataset ref -- upscaled
        # only for display. This is the real like-for-like the features compare.
        mlive = resize_hw(policy.model_input_u8(live), 640, 480)
        mref = resize_hw(policy.model_input_u8(ref_frames[r["best_ref"]]), 640, 480)
        blend = (0.5 * mlive.astype(np.float32) + 0.5 * mref.astype(np.float32)).astype(np.uint8)
        diff = np.abs(mlive.astype(np.int16) - mref.astype(np.int16)).mean(-1)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
        for ax, im, ttl in zip(
                axes, [mlive, mref, blend, diff],
                ["LIVE model input (256)",
                 f"dataset ref model input ({ref_names[r['best_ref']]} t=0)",
                 "50/50 blend (edges should align)", "abs diff"]):
            ax.imshow(im, cmap="magma" if im.ndim == 2 else None)
            ax.set_title(ttl, fontsize=11)
            ax.axis("off")
        fig.suptitle(" | ".join(report_text(r, baseline_min, baseline_mean,
                                            home_xyz, ref_names)[:2]), fontsize=9)
        fp = os.path.join(args.out_dir, f"check_{time.strftime('%Y%m%d%H%M%S')}{tag}.png")
        fig.tight_layout()
        fig.savefig(fp, dpi=100)
        plt.close(fig)
        return fp

    # ---- optional live wrist source ----
    wrist_reader = None
    get_wrist = None
    if wrist_feats is not None:
        if args.fake_live:
            demo, t = args.fake_live.split(":")
            with h5py.File(args.hdf5, "r") as f:
                fake_w = f["data"][demo]["obs"]["wrist_cam"][int(t)][:]
            get_wrist = lambda: fake_w  # noqa: E731
        else:
            from mini_lawam.rollout_ur7e import RealSenseTableReader
            wrist_reader = RealSenseTableReader(args.wrist_cam_serial, name="wrist_cam",
                                                exposure=args.wrist_exposure,
                                                gain=args.wrist_gain)
            wrist_reader.start()
            get_wrist = wrist_reader.read_rgb

    def wrist_line():
        cos, idx = feature_cos_vs_refs(policy, get_wrist(), wrist_feats)
        ok = cos >= wrist_baseline_min
        return (f"WRIST feature cos(live, dataset) = {cos:.4f}   "
                f"(baseline wrist min={wrist_baseline_min:.4f})   "
                f"{'OK (>=baseline)' if ok else 'LOW (wrist also OOD)'}"), ok

    if args.once:
        live = get_live()
        wrist_live = get_wrist() if get_wrist is not None else None
        r = evaluate(policy, live, ref_feats, baseline_min, home_xyz,
                     wrist_live=wrist_live, action_scale=args.action_scale)
        for line in report_text(r, baseline_min, baseline_mean, home_xyz, ref_names):
            print(line)
        if wrist_live is not None:
            cos, _ = feature_cos_vs_refs(policy, wrist_live, wrist_feats)
            ok = cos >= wrist_baseline_min
            print(f"WRIST feature cos(live, dataset) = {cos:.4f}   "
                  f"(baseline wrist min={wrist_baseline_min:.4f})   "
                  f"{'OK (>=baseline)' if ok else 'LOW (wrist also OOD)'}")
        print(f"report -> {save_report(live, r)}")
        return

    # ---- GUI loop ----
    import pygame
    pygame.init()
    W, H = 480, 360  # per-panel display size
    screen = pygame.display.set_mode((W * 3, H + 110))
    pygame.display.set_caption("mini_lawam camera check  [Q quit | S snapshot | R cycle ref]")
    font = pygame.font.SysFont("monospace", 15)
    ref_idx = 0
    print("GUI: aim the camera until the BLEND panel is sharp (no ghosting), "
          "then wait for VERDICT: IN-DISTRIBUTION.")
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    running = False
                elif ev.key == pygame.K_r:
                    ref_idx = (ref_idx + 1) % len(ref_frames)
                elif ev.key == pygame.K_s:
                    print(f"snapshot -> {save_report(live, r, tag='_snap')}")
        live = get_live()
        wrist_live = get_wrist() if get_wrist is not None else None
        r = evaluate(policy, live, ref_feats, baseline_min, home_xyz,
                     wrist_live=wrist_live, action_scale=args.action_scale)
        # resize BOTH to the panel size first (live is 640x480, dataset refs are smaller)
        live_d = resize_hw(live, W, H)
        ref_d = resize_hw(ref_frames[ref_idx], W, H)
        blend = (0.5 * live_d.astype(np.float32) + 0.5 * ref_d.astype(np.float32)).astype(np.uint8)
        for i, im in enumerate([live_d, ref_d, blend]):
            surf = pygame.surfarray.make_surface(
                np.ascontiguousarray(im.swapaxes(0, 1)))
            screen.blit(surf, (i * W, 0))
        screen.fill((20, 20, 20), rect=(0, H, W * 3, 110))
        lines = report_text(r, baseline_min, baseline_mean, home_xyz, ref_names)
        lines[2] += f"   (shown ref: {ref_names[ref_idx]} — press R to cycle)"
        for i, line in enumerate(lines):
            ok = not line.startswith("VERDICT") or r["ok"]
            screen.blit(font.render(line, True, (120, 255, 120) if ok else (255, 120, 120)),
                        (8, H + 6 + i * 25))
        pygame.display.flip()
        time.sleep(0.15)
    pygame.quit()


if __name__ == "__main__":
    main()
