# mini_lawam — Handoff / Context Document

A minimal, **LaWAM-inspired vision-only behavior-cloning policy** built inside the
LaWAM repo, deployed on a real **UR7e** (egg pick-and-place). Read this to
understand the whole system before touching code. Last updated: 2026-07-30.

## 1. What it is (one paragraph)

`mini_lawam/` is a simplified re-implementation of **LaWAM Stage 2** (paper:
"LaWAM: Latent World Action Models"). Real LaWAM uses a VLM (Qwen3-VL) to predict
a latent action, decodes it through a frozen Latent World Model (LaWM) into a
latent visual **subgoal**, and conditions a flow-matching action expert on it.
`mini_lawam` keeps the frozen LaWM + teacher-student distillation but replaces the
VLM with a small **ConvPrior** (CNN over DINO tokens) and the flow expert with a
**cross-attention head** (`--head attn`, token-level). The legacy pooled MLP head
has been removed. Single-task, language-free.

## 2. Architecture / data flow

```
o_t(table) ─DINO(frozen)─► u_t ─ConvPrior─► ẑ ─LaWM decoder(frozen)─► subgoal û_T
                            │                                            │
                            ▼                                            ▼
        action head:  24 learned queries cross-attend the RAW tokens of
                      {u_t, û_T, wrist} (+optional state token)
                        ├─ XYZ regression projection ─► [H,3]
                        └─ optional binary grip projection ─► [H,1] logits
teacher (train only): LAM inverse-dynamics(u_t, u_T) ─► z_teacher ⇒ distill ẑ ← z
```

- Legacy `--gripper-head regression`: `loss_act` is masked MSE on all four
  outputs. New `--gripper-head binary`: `loss_act = loss_xyz + λ_g·loss_gripper`,
  where XYZ uses masked MSE and gripper uses binary cross-entropy on the raw
  class (`0=open`, `1=close`). Binary inference emits exact `−1/+1`.
- Frozen: DINOv3, LAM IDM teacher, LaWM decoder. Trainable: ConvPrior + head.
- **Horizon H = 24 frames = 1.2 s @ 20 Hz** for BOTH the LaWM pair gap and the
  action chunk (`--horizon`, default 24).
- Target (`--target`, stored in ckpt as `target_mode`): `abs` = absolute
  `[eef_pos_base(3), gripper]` (v0); `delta` = cumulative EEF displacement from
  the current frame; `joystick` = recorded `[actions[0:3], actions[6]]`. See the
  exact contracts below.
- Gripper timing is independently checkpointed as `gripper_target_offset`.
  `--gripper-target-offset 1` means action-head row `i` uses the gripper command
  from `t+i+1` even when joystick XYZ remains at `t+i`.
- `--include-tail-actions` keeps the final placement/release frames as phase-2
  action-head anchors. Short terminal chunks use the existing right-padding
  mask, and the LaWM future image is clamped to the terminal frame.
- `use_wrist`: wrist_cam feeds the **action head only** (never prior/LaWM — §C.2).
- `use_state` (optional, off in current ckpts): current eef xyz as extra head input.
- Inference `predict()`: current frame(s) only → chunk. No future frame.

### Heads
- `head_type="attn"` (~7.4M) is the only supported action head:
  `AttnActionHead` uses per-timestep queries and 3 cross-attention blocks
  (hidden 384, 6 heads) over all patch tokens.
- `gripper_head="regression"` preserves old checkpoint state dictionaries.
  `gripper_head="binary"` shares the attention trunk but splits the final XYZ
  and gripper projections; use it for new training.

### Delta action contract (current deployment)

The deployed `ckpt_100ep_attn_delta.pt` uses `target_mode="delta"`,
`action_horizon=24`, `head_type="attn"`, `use_wrist=True`, and
`use_state=False`. For an anchor frame `t`, training row `i` is:

```
chunk[i] = [pos[t+i+1] - pos[t], gripper[t+i+1]],  i = 0..23
```

- Each XYZ row is a **cumulative displacement from the same `pos[t]` anchor**,
  in meters in the UR base frame. It is NOT an incremental
  `pos[t+i+1]-pos[t+i]` command, and rows must never be cumulatively summed.
- Gripper remains the raw action channel (approximately −1 open / +1 close).
  Rotation is not predicted; rollout uses the fixed `DEMO_LOCKED_ROTVEC`.
- Delta targets are z-scored during training. This checkpoint stores
  `mean=[-0.01014, 0.01642, -0.01199, -0.11442]` and
  `std=[0.03221, 0.04150, 0.03942, 0.99418]`.
- During deployment, rollout samples the measured current TCP once and passes it
  to `MiniLaWAMPolicy.act()`. The policy unnormalizes the chunk and converts
  every row to an absolute target:
  `target_xyz[i] = current_TCP_xyz + predicted_delta[i]`.
- `use_state=False` means TCP XYZ is **not a neural-network input**. The live TCP
  is still required outside the network as the delta-composition anchor.
- With temporal ensembling, rollout replans at 20 Hz, re-anchors every new chunk
  at the newly measured TCP, averages overlapping **absolute XYZ targets**, and
  executes one ensembled waypoint. The 500 Hz servo thread then interpolates
  toward that target.
- `target_mode` is checkpoint-controlled; there is no deployment CLI switch.
  Old checkpoints without the field fall back to `abs`.

### Joystick action contract

For `--target joystick`, XYZ for input observation `t` is aligned to the
recorded command at the same index:

```
chunk[i] = [actions[t+i, 0:3], actions[t+i, 6]],  i = 0..23
```

- The formula above is the legacy same-index checkpoint contract
  (`gripper_target_offset=0`). The recommended binary-gripper run uses
  `gripper_target_offset=1`, changing only the fourth channel to
  `actions[t+i+1, 6]`. This teaches `chunk[0]` the next-row release command
  without shifting joystick XYZ.
- Rotation columns `3:6` are omitted; the rollout keeps its locked orientation.
- Training uses the original joystick values and target-specific mean/std. It
  does not apply the deployment gain.
- `MiniLaWAMPolicy.act()` denormalizes back to the original joystick scale.
- `rollout_ur7e --action-scale 0.3` multiplies only XYZ. Each scaled command is
  added to the live TCP when that row is executed. Gripper is never scaled and
  is thresholded at zero.
- Switch back without code changes by selecting a checkpoint trained with
  `--target delta`.

## 3. Two-phase training (current workflow)

Phase 1 trains ConvPrior only (distillation); phase 2 loads that prior (frozen)
and trains the head. The prior is head- and wrist-independent → **train phase 1
once per dataset, reuse for all phase-2 variants**.

```bash
# Phase 1 (prior only; ~3M params; best ckpt on val loss_distill)
python -m mini_lawam.train --hdf5 <data.hdf5> --phase 1 --target delta --steps 10000 \
    --out results/mini_lawam/phase1_<name>.pt

# Phase 2 (attention head + wrist + delta target; lr 1e-4)
python -m mini_lawam.train --hdf5 <data.hdf5> --phase 2 --head attn --use-wrist \
    --target delta \
    --prior-ckpt results/mini_lawam/phase1_<name>.pt \
    --steps 10000 --batch 32 --lr 1e-4 \
    --out results/mini_lawam/ckpt_<name>_attn_delta.pt \
    --csv-log results/mini_lawam/log_<name>_attn_delta.csv

# Alternative phase 2: same attention head/prior, raw joystick target
python -m mini_lawam.train --hdf5 <data.hdf5> --phase 2 --head attn --use-wrist \
    --target joystick --gripper-head binary --gripper-target-offset 1 \
    --include-tail-actions --lambda-gripper 1.0 \
    --prior-ckpt results/mini_lawam/phase1_<name>.pt \
    --steps 10000 --batch 32 --lr 1e-4 \
    --out results/mini_lawam/ckpt_<name>_attn_joystick_binary_grip_t1.pt \
    --csv-log results/mini_lawam/log_<name>_attn_joystick_binary_grip_t1.csv
```
`--phase joint` = original single-phase. Phase-2 ckpt is self-contained
(prior + head + cfg + action stats) → deployment needs only that one file.
`head_type`/`gripper_head`/`use_wrist`/`use_state`/`target_mode` are stored in
the ckpt and auto-detected everywhere downstream. MLP-head checkpoints are no
longer supported. Old attention checkpoints without `gripper_head` default to
legacy regression. Phase 1 itself is target-independent
(distillation only), so the same prior can be reused for delta and joystick
phase 2 runs; phase 2 must pass the intended `--target`. Do not combine
`--use-state` with `--target delta` or `--target joystick`: their target
statistics cannot normalize an absolute TCP state.

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
   → step-0/horizon L2, per-dim MAE, gripper acc, train vs val.
   The evaluator reads `target_mode` and reconstructs absolute positions for
   delta checkpoints using each dataset frame's current EEF position; joystick
   checkpoints are compared directly against the raw command rows. The
   100ep absolute-attn reference was **1.6 cm step-0, z-MAE 0.62 cm,
   99.2% gripper**.
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

**Current best-practice command (100ep attention + delta checkpoint):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_100ep_attn_delta.pt \
  --table-cam-serial 244422300964 --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.125 --execute --use-gripper-control \
  --train-frame-hw 168 224 \
  --temporal-ensemble --te-m 0.1 --delta-scale 1.0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces \
  --show-camera --show-subgoal --subgoal-update-steps 8
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
  Gripper is never ensemble-averaged. Closing uses the newest immediate row;
  once already closed, `--gripper-open-lead-steps 1` may use the newest
  chunk's next row to trigger release 50 ms early, then latches open for the
  remainder of the rollout. This avoids both the old early-grasp failure and
  the observed “close now, open next step” release deadlock.
  Old receding-horizon mode (exec 8/replan) remains the non-TE fallback.
- Delta composition happens before temporal ensembling: each new chunk is
  anchored at the actual TCP measured for that replan. `--max-reach` then limits
  how far the resulting absolute target may be from the actual TCP;
  `--servol-max-pos-step` limits the 500 Hz interpolated command step.
- `--delta-scale` is a deployment-only XYZ gain for delta checkpoints:
  `target_xyz = anchor_xyz + delta_scale * predicted_delta`. It is applied before
  temporal ensembling, target smoothing, and safety clamps, and never changes
  the gripper channel. `1.0` exactly preserves existing behavior. Increase
  gradually (`1.25`, then `1.5`; test `2.0` only after confirming TCP tracking).
  Non-default values are rejected for absolute-target checkpoints.
- `--action-scale` is the corresponding deployment-only gain for joystick
  checkpoints (default `0.3`). It scales only the denormalized XYZ command;
  gripper is unchanged. The scaled command is composed from the live TCP at each
  execution tick, before target smoothing and safety clamps.
- `--save-frames 8` dumps the exact policy-input frames per rollout → offline
  forensics (`policy.act` on saved frames, nearest-neighbor vs dataset, etc.).
- `--trace-dir` always saves the first table frame plus a compact summary JSON
  (`started_at`, `finished_at`, S-to-H duration, steps, and frame path).
  It scans existing `trial_<N>_...` files and resumes at `max(N)+1` across
  sessions. Basenames use `trial_10_20260724T133010_{table_cam_first.png,summary.json}`
  with no zero padding or `_async_` marker. Both files share one trial number;
  duplicate files are never counted as separate trials.
- GUI shows the **256×256 model inputs** (table + wrist) — what the policy sees.
- `--show-subgoal` adds a live heatmap over the table input using the predicted
  DINO feature change `||u_hat_T-u_t||`. Red/yellow patches indicate where the
  LaWM subgoal predicts the largest visual-feature change. It reuses tokens from
  the action inference pass rather than running DINO twice; `--subgoal-alpha`
  controls overlay opacity. The GUI refreshes the heatmap every
  `--subgoal-update-steps` policy inferences (default 8, approximately 2.5 Hz
  with temporal ensembling) and holds the previous overlay between updates.
  This display throttle does not freeze or otherwise change the internal
  subgoal used by the action policy.

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
7. **Legacy MLP pooled head plateau**: 2.2–3.2 cm offline, train≈val (fit limit, not
   data limit) → replaced pooling with cross-attention head → clear improvement
   (user-confirmed on robot; offline 1.6 cm on 100ep).
8. **Absolute-target live offset**: the 100ep absolute-attn policy overshot the
   egg by ~8–10 cm in y despite good offline error. Delta targets were added on
   2026-07-23 to remove dependence on regressing a globally biased absolute
   position: each replan now predicts motion relative to the measured TCP.
   Current deployment checkpoint: `ckpt_100ep_attn_delta.pt`. This addresses the
   action-coordinate failure mode; it does not solve visual misidentification
   (for example, confusing bowl contents with the egg).
9. **Release procrastination with receding-horizon gripper prediction**: the
   joystick binary-head policy picked up the egg correctly and reached the bowl,
   but hovered while holding the egg. Trial 59 showed that the model did predict
   release, with the first open row counting down from chunk index 23 → 15 → 8
   → 2 → 1. Once it reached `[close now, open next step, ...]`, rollout kept
   executing only the newest `chunk[0]`. Because the gripper remained closed,
   the camera observation barely changed; every replan repeated the same
   `[close now, open next step]` forecast. The release stayed perpetually one
   step away until visual drift finally changed `chunk[0]` roughly 9.2 seconds
   later. This is a discrete-action receding-horizon deadlock, not a missing
   release class or a slow 20 Hz policy loop.

   Fix: `--gripper-open-lead-steps 1` (default) lets only the release event use
   the newest chunk's next row once the commanded gripper is already closed.
   Closing/grasping still uses the immediate row. When release triggers, it is
   latched open for the rest of that single-pick rollout so the next unchanged
   pre-release image cannot command close again. Set the option to `0` to
   disable lookahead. A trigger prints:
   `[GRIPPER] release triggered from chunk[1] (open lookahead=1 step)`.

   Why this is needed: action chunks are forecasts, not commitments. Continuous
   XYZ commands make partial progress and change the next observation; a
   discrete `+1=close` command makes no progress toward `-1=open`. The
   memoryless head cannot remember that its previous chunk promised to open on
   the next frame. Legacy joystick checkpoints use gripper row `t`, whereas
   delta checkpoints use `t+1`; new training should set the now-explicit
   `--gripper-target-offset 1` so the gripper clock does not depend on XYZ mode.
   In addition, legacy `build_index` required a full 24-row future horizon and
   therefore excluded the final 24 anchor frames of every demo. In `demo_0`,
   the frame immediately before release was not an action-head training anchor:
   release appeared only in later query rows of earlier chunks. Use
   `--include-tail-actions` in phase 2 so query 0 is directly supervised on the
   terminal placement and release states.

   Prevention:
   - Define and test camera/action/state timestamps and label offsets explicitly;
     gripper alignment must not change implicitly with the XYZ target mode.
   - Evaluate closed-loop `predict → execute row 0 → observe → replan`, not only
     open-loop chunk accuracy. Inspect the first predicted open index across
     replans; a countdown that stalls at index 1 reveals procrastination.
   - Score grasp/release transition error in frames. Overall gripper accuracy is
     dominated by long constant-state intervals and can hide a late release.
   - Treat discrete events separately from continuous temporal ensembling:
     use event latching, hysteresis/confirmation, or explicit scheduling rather
     than averaging binary commands.
   - Record actual gripper position/status in future datasets. The current HDF5
     stores the requested latched command only, so actuator latency and command
     completion cannot be learned or measured.
   - For the next clean training ablation, use the explicit
     `--gripper-target-offset 1 --include-tail-actions` and compare joystick-binary with
     `delta + binary gripper`. Training step 0 against `gripper[t+1]` should
     reduce inference-time lookahead, while a configurable deployment lead can
     remain for physical actuator latency.

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
- Delta rows are all relative to one chunk anchor; do not integrate them across
  the horizon. Deployment must provide current TCP XYZ to `policy.act()` even
  though the checkpoint has `use_state=False`.
- `--use-state` with `--target delta` or `--target joystick` is intentionally
  unsupported. Never reinterpret a trained head as another target mode: rollout
  must use the checkpoint's stored `target_mode`.
- `check_camera`/`rollout_ur7e` share `RealSenseTableReader` (name, exposure,
  gain per camera). Quit one before starting the other (camera is exclusive).
- Rollout summary JSONs are compact and no longer contain per-step
  `pred_xyz`/`tgt_xyz`/`grip`; use `--save-frames N` when frame-level forensics
  are needed.
- h5py truncated-file EOF error after copying a dataset = incomplete transfer;
  check file size.
