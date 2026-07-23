# mini_lawam — Handoff / Context Document

A minimal, **LaWAM-inspired vision-only behavior-cloning policy** built inside the
LaWAM repo, deployed on a real **UR7e** (egg pick-and-place). Read this to
understand the whole system before touching code. Last updated: 2026-07-22.

## 1. What it is (one paragraph)

`mini_lawam/` is a simplified re-implementation of **LaWAM Stage 2** (paper:
"LaWAM: Latent World Action Models"). Real LaWAM uses a VLM (Qwen3-VL) to predict
a latent action, decodes it through a frozen Latent World Model (LaWM) into a
latent visual **subgoal**, and conditions a flow-matching action expert on it.
`mini_lawam` keeps the frozen LaWM + teacher-student distillation but replaces the
VLM with a small **ConvPrior** (CNN over DINO tokens) and the flow expert with
either an **MLP head** (v0, pooled features) or a **cross-attention head**
(`--head attn`, token-level — the current best; fixed the grasp-precision problems
the pooled MLP had). Single-task, language-free.

## 2. Architecture / data flow

```
o_t(table) ─DINO(frozen)─► u_t ─ConvPrior─► ẑ ─LaWM decoder(frozen)─► subgoal û_T
                            │                                            │
                            ▼                                            ▼
        action head:  MLP:  [pool(u_t) ‖ pool(û_T) ‖ pool(wrist)] ──► chunk [H,4]
                      ATTN: 24 learned queries cross-attend the RAW tokens of
                            {u_t, û_T, wrist} (+optional state token) ──► chunk
teacher (train only): LAM inverse-dynamics(u_t, u_T) ─► z_teacher ⇒ distill ẑ ← z
```

- Losses: `loss_act` (masked MSE on chunk) + `λ_d·loss_distill` + `λ_wm·loss_wm`.
- Frozen: DINOv3, LAM IDM teacher, LaWM decoder. Trainable: ConvPrior + head.
- **Horizon H = 24 frames = 1.2 s @ 20 Hz** for BOTH the LaWM pair gap and the
  action chunk (`--horizon`, default 24).
- Target (`--target`, stored in ckpt as `target_mode`): `abs` = absolute
  `[eef_pos_base(3), gripper]` (v0); `delta` = `pos[t+i]-pos[t]` relative to the
  current frame — deployment composes `current_TCP + Δ` each replan (servo-like,
  immune to systematic absolute-position bias; added 2026-07-23 to counter the
  constant ~8-10 cm live grasp offset — mirrors lapa-barry's delta action space).
  All z-scored (stats in ckpt). `--use-state` + delta is disallowed (stats clash).
- `use_wrist`: wrist_cam feeds the **action head only** (never prior/LaWM — §C.2).
- `use_state` (optional, off in current ckpts): current eef xyz as extra head input.
- Inference `predict()`: current frame(s) only → chunk. No future frame.

### Heads
- `head_type="mlp"` (~1.1–1.5M params): 3-layer MLP on mean-pooled features.
  Weakness (measured): pooling destroys fine spatial signal → coarse z, early
  grasp commits.
- `head_type="attn"` (~7.4M): `AttnActionHead` — per-timestep queries, 3
  cross-attn blocks (hidden 384, 6 heads) over all patch tokens. **Use this.**

## 3. Two-phase training (current workflow)

Phase 1 trains ConvPrior only (distillation); phase 2 loads that prior (frozen)
and trains the head. The prior is head- and wrist-independent → **train phase 1
once per dataset, reuse for all phase-2 variants**.

```bash
# Phase 1 (prior only; ~3M params; best ckpt on val loss_distill)
python -m mini_lawam.train --hdf5 <data.hdf5> --phase 1 --steps 10000 \
    --out results/mini_lawam/phase1_<name>.pt

# Phase 2 (attention head + wrist; lr 1e-4 for the transformer head)
python -m mini_lawam.train --hdf5 <data.hdf5> --phase 2 --head attn --use-wrist \
    --prior-ckpt results/mini_lawam/phase1_<name>.pt \
    --steps 10000 --batch 32 --lr 1e-4 \
    --out results/mini_lawam/ckpt_<name>_attn.pt --csv-log results/mini_lawam/log_<name>.csv
```
`--phase joint` = original single-phase. Phase-2 ckpt is self-contained
(prior + head + cfg + action stats) → deployment needs only that one file.
`head_type`/`use_wrist`/`use_state` are stored in the ckpt and auto-detected
everywhere downstream.

## 4. Datasets (all robomimic HDF5, 20 Hz, UR7e)

| file | demos | stored img | recorded at | start joint pose |
|---|---|---|---|---|
| `multi_egg_114ep.hdf5` (old scene) | 114 | 168×224 | 224 | old home `[0,-π/2,-π/2,-π/2,π/2,π/2]` |
| `multi_egg_30_moved_256.hdf5` | 30 | 256×256 | **native 256** | new home `[0.4076,-1.4255,-1.7052,-1.5821,1.5703,1.9768]` |
| `new_100ep_multi_egg_exp_plate(_256).hdf5` | 101 | 224 (/256 converted) | **224** | new home (same as above) |

- `convert_hdf5_to_256.py` (repo root) pre-resizes a 224 file to stored-256 —
  **training-identical** to on-the-fly resize; it does NOT change the recording
  resolution (see deployment rule below).
- Collection exposure for the 100ep set: **table 180/gain 16, wrist 100/gain 16**.
- Layout: `data/demo_k/obs/{table_cam,wrist_cam,eef_pos_base,eef_quat_base,joint_pos}`,
  `actions (T,7)` with gripper = col 6 (−1 open / +1 close). Grasp z ≈ 0.18,
  place z ≈ 0.25 in the 100ep set.

## 5. Evaluation ladder (run in this order, cheap → expensive)

1. **Phase-1 prior**: `python -m mini_lawam.eval_prior --ckpt <phase1.pt> --hdf5 <data>`
   → R² vs mean/shuffle baselines + subgoal floor(copy)/ceiling(oracle) bracket.
   Good: R² ≥ ~0.85, margin recovery ≥ 70%. (100ep prior: R²≈0.93, 99%.)
2. **Offline action error**: `python -m mini_lawam.rollout --mode eval --ckpt <p2.pt> --hdf5 <data>`
   → step-0/horizon L2 (cm), per-dim MAE, gripper acc, train vs val.
   Current best (100ep attn): **1.6 cm step-0, z-MAE 0.62 cm, 99.2% gripper** — model is NOT the bottleneck.
3. **Subgoal viz**: `python -m mini_lawam.viz_subgoal --ckpt <any ckpt with prior> --hdf5 <data> --demo demo_0 --t 40 80`
   → PCA maps + pred-vs-true change heatmaps of the LaWM subgoal.
4. **Pre-flight camera check** (robot at HOME, before every rollout session):
   `python -m mini_lawam.check_camera --ckpt <p2.pt> --hdf5 <data> --table-cam-serial ... --wrist-cam-serial ... --train-frame-hw 168 224 --table-exposure 180 --table-gain 16 --wrist-exposure 100 --wrist-gain 16 --once`
   → live-vs-dataset DINO cos (table + wrist, each vs own baseline) + step-0 gate
   (pred at home must be ≤5 cm from home). GUI mode (no `--once`): blend panel to
   physically re-align scene/camera; `R` cycles reference demos.
   **Caveat**: the cos baseline is session-inflated (refs share white balance /
   lighting); treat cos as a relative gauge, the step-0 gate as the harder signal.

## 6. Deployment (`rollout_ur7e.py`)

servoL stack: background 500 Hz thread interpolates toward a shared target TCP;
policy loop at 20 Hz. Orientation locked to `DEMO_LOCKED_ROTVEC` (tool down);
gripper close at 23 mm. Keys: S start / E end / H home / Q quit.

**Current best-practice command (100ep attn checkpoint):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_new_100ep_multi_egg_exp_plate_attn_256.pt \
  --table-cam-serial 244422300964 --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.125 --execute --use-gripper-control \
  --train-frame-hw 168 224 \
  --temporal-ensemble --te-m 0.1 --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces --save-frames 8 --show-camera
```

**The deployment-matching rules (each one was a debugged failure):**
- `--train-frame-hw H W` = the dataset's **recording** resolution (NOT stored
  size): `168 224` for 224-recorded sets (incl. their _256 conversions!); `0 0`
  only for native-256 recordings (30_moved_256). Wrong flag ⇒ sharpness OOD
  (~40 mm prediction drift measured).
- `--table/wrist-exposure/-gain` must match collection values (auto-exposure
  drifts with room light). White balance is still auto — a known residual.
- `--home-q` must equal the dataset's `joint_pos[0]` (default in code = the new
  home; old-scene ckpts need the old π/2 home). 23° mismatch ⇒ start-frame OOD.
- `--temporal-ensemble` (ACT-style): replans every step (15 ms ≪ 50 ms budget),
  executes an exp-weighted average of all overlapping chunks → smooth motion.
  Gripper deliberately taken from the **newest chunk[0] only** (averaging a
  ±1 switch fires it early → grasps 4 cm high; measured + fixed).
  Old receding-horizon mode (exec 8/replan) remains the non-TE fallback.
- `--save-frames 8` dumps the exact policy-input frames per rollout → offline
  forensics (`policy.act` on saved frames, nearest-neighbor vs dataset, etc.).
- GUI shows the **256×256 model inputs** (table + wrist) — what the policy sees.

## 7. Debug history — what broke and what fixed it (chronological)

1. **Scene-change OOD** (old 114ep scene dismantled): predictions collapsed to
   dataset-mean region (~50 cm off). Diagnosed via `check_camera` (cos below
   dataset-internal baseline; offline predictions fine). Fix: fresh demos.
2. **Resolution mismatch**: live 640→256 direct vs training 224-recorded.
   Fix: `--train-frame-hw` pre-resize in `MiniLaWAMPolicy.preprocess`.
3. **Home-pose mismatch**: rollout homed to old π/2 pose, new data starts 23°
   away. Fix: `--home-q` + new default.
4. **Speed throttling**: `--max-reach 0.005` capped the arm at ~30 mm/s; demos
   move ~70 mm/s → trajectory never completed. Fix: 0.02 + TE smoothing.
5. **Jitter**: receding-horizon chunk switching (target reversed direction ~50%
   of steps). Fix: temporal ensembling.
6. **Grasped 4 cm high**: TE-averaged gripper fired early (far-horizon "+1"
   leaking into the average). Fix: newest-chunk gripper.
7. **MLP pooled head plateau**: 2.2–3.2 cm offline, train≈val (fit limit, not
   data limit) → replaced pooling with cross-attention head → clear improvement
   (user-confirmed on robot; offline 1.6 cm on 100ep).
8. **OPEN ISSUE (as of 2026-07-22)**: with everything matched, the 100ep attn
   policy approaches but **overshoots the egg by ~8–10 cm in y and hovers at
   z≈0.24 over the BOWL (behind the plate), gripper dithering**. Intent
   reconstruction from saved frames: model plans a full grasp at [0.07, 0.53]
   (grip_end +1, z 0.19) — it believes the egg is where the bowl is. Camera
   shift ruled out (0.6 px). Leading hypothesis: the bowl's yellow, egg-like
   contents hijack egg localization. Pending experiments:
   **(A) empty-bowl test** — remove bowl contents, rerun;
   **(B) measure true egg TCP** (freedrive over egg,
   `rtde_receive.getActualTCPPose()`) → exact miss vector.

## 8. Environment

- Repo: `/home/iclu200/reinaldoyang/LaWAM` (branch `mini-lawam`, GitLab origin).
  Related: `~/reinaldoyang/ur7e_ramen_il/scripts/real_world/` (recorder,
  `real_servo_utils.py` — servo thread + `move_robot_home(home_q=...)`).
- Conda env `lawam` (`~/miniconda3/envs/lawam`). RTX 5090 needs torch 2.7.1/cu128.
- **DINOv3**: official gated HF weights downloaded into
  `weights/dinov3-vitb16-pretrain-lvd1689m/` (YAML `vision_model_id` points at
  that local dir). Bit-exact with the earlier local .pth conversion — see
  `summary.md` for history/revert.
- Always run as modules from the repo root. Training data ≠ this machine
  sometimes (user trains on a second box) — checkpoints/datasets get copied over.
- The user runs all real training/rollout commands themselves; agents should
  hand over commands, not execute them (short smoke tests OK).

## 9. Gotchas

- `gpu_two_view_video_aug(..., training=False)` does the ImageNet norm on GPU —
  never double-normalize; `data.py` returns uint8 only.
- LAM teacher runs under `inference_mode` → teacher outputs are
  `.detach().clone()`d in `_teacher()`.
- Phase-1 ckpts have NO action head/stats → `rollout.py` can't load them
  (`eval_prior`/`viz_subgoal` can).
- `check_camera`/`rollout_ur7e` share `RealSenseTableReader` (name, exposure,
  gain per camera). Quit one before starting the other (camera is exclusive).
- Trace JSONs store `pred_xyz`/`tgt_xyz`/`grip` per control step — the analysis
  scripts in the debug history all read these.
- h5py truncated-file EOF error after copying a dataset = incomplete transfer;
  check file size.
