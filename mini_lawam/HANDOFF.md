# mini_lawam — Handoff / Context Document

A minimal, **LaWAM-inspired vision-only behavior-cloning policy** built inside the
LaWAM repo. Read this to understand the whole thing before touching the code.

## 1. What it is (one paragraph)

`mini_lawam/` is a from-scratch, simplified re-implementation of **LaWAM Stage 2**
(the paper: "LaWAM: Latent World Action Models", arXiv 2606.15768). LaWAM's real
Stage 2 uses a Vision-Language-Model (Qwen3-VL) to predict a latent action, feeds
it through a frozen Latent World Model (LaWM) to produce a latent visual
**subgoal**, and conditions a flow-matching action expert on that subgoal.
`mini_lawam` keeps the frozen LaWM + teacher-student distillation, but **replaces
the VLM with a small ConvNet ("ResNet role") and the flow expert with an MLP+MSE
head** — i.e. a lightweight, single-task, language-free variant for a specific
real-world robot dataset.

## 2. Architecture / data flow

```
o_t ──DINO(frozen)──► u_t ──ConvPrior──► ẑ ──LaWM decoder(frozen)──► subgoal û_T
                       │                                                  │
                       ├──────── pool(u_t) ───────────────┐              │
                       │                                   ▼              ▼
                       │                          [pool(u_t) ‖ pool(û_T)] ──MLP──► action chunk
teacher (train only):  └─► LAM inverse-dynamics(u_t, u_T) ──► z    ⇒   distill ẑ ← z
```

Losses (weighted sum):
- `loss_act`     = masked MSE(pred_chunk, target_chunk)      ← behavior cloning
- `loss_distill` = MSE(ẑ, z_teacher)                         ← ConvPrior learns to predict the latent action
- `loss_wm`      = MSE(û_T, u_T)                             ← light subgoal supervision (weight 0.1)

**Frozen:** DINOv3 encoder, LAM inverse-dynamics teacher, LaWM decoder.
**Trainable:** ConvPrior + MLPActionHead only (~a few M params).

At inference (`predict()`), only `o_t` is used — no future frame — so it's
deployable: `o_t → u_t → ẑ → û_T → action chunk`.

## 3. Files

| File | Contents |
|---|---|
| `mini_lawam/model.py` | `MiniLaWAMConfig`, `ConvPrior` (u_t→ẑ), `MLPActionHead`, `MiniLaWAM` (frozen LAM + prior + head, `forward()` train, `predict()` infer) |
| `mini_lawam/data.py` | `MiniLaWAMDataset` (HDF5 pair loader), `build_index`, `compute_action_stats`, `split_o_t_o_T` |
| `mini_lawam/train.py` | single-GPU AdamW loop (trains prior+head only), cosine LR, val + best-checkpoint saving |
| `mini_lawam/__init__.py` | exports `MiniLaWAM`, `MiniLaWAMConfig` |

## 4. Data (the target dataset)

Real-world **UR7e** teleop, collected by `~/reinaldoyang/ur7e_ramen_il/scripts/real_world/record_real.py`.
File: `dataset/multi_egg.hdf5` (currently **~83 demos**; pick-and-place location
varies across demos). robomimic/IsaacLab HDF5 layout:

```
data/demo_k/obs/table_cam      (T,168,224,3) uint8   # fixed external RealSense  <-- USED (primary view)
data/demo_k/obs/wrist_cam      (T,168,224,3) uint8   # arm-mounted (moves) -- NOT used for LaWM
data/demo_k/obs/eef_pos_base   (T,3) float32         # TCP position, base frame, meters  <-- target
data/demo_k/obs/eef_quat_base  (T,4) float32
data/demo_k/obs/joint_pos      (T,6) float32
data/demo_k/actions            (T,7) float32         # [dx,dy,dz,dRx,dRy,dRz,gripper]
```
Notes:
- **20 Hz control** (from `record_real.py --control_hz 20`). So τ=1.6 s ⇒ **gap = 32 frames**.
- **No `eef_pos` / `gripper_pos` obs** here — that's why position comes from
  `eef_pos_base` and gripper comes from `actions[:,6]` (`{-1 open, +1 close}`).
- Rotation dims (`actions[:,3:6]`) are ~0 (no rotation used) → excluded.
- `ROBOT_ACTION_SCALE=0.3` in collection scales the *recorded delta actions* by 0.3
  before execution — a reason we clone absolute `eef_pos_base` (immune to this),
  not the delta `actions`.

### BC target
`[eef_pos_base(3), actions_gripper(1)]` = 4 dims, absolute, **z-scored** (mean/std
saved in the checkpoint). Target chunk = frames `t+1 … t+H` (H=32).

### Dataloader contract
`MiniLaWAMDataset` returns uint8 `frames_u8 [2,3,256,256]` (o_t, o_{t+gap}), resized
only. The **train loop normalizes on GPU** via
`latent_action_model.data_loader.video_aug.gpu_two_view_video_aug(..., training=False)`
(ImageNet norm), which matches exactly what the frozen LAM expects. Do not
double-normalize.

## 5. Key design decisions (and why)

- **Vision-only, VLM replaced by ConvPrior** — single-task real-world data; no
  language instructions available; lightweight. Downside: `ẑ` is predicted from the
  current frame alone, so the prior is harder (this is what the distill loss tests).
- **MLP+MSE action head (v0), not the flow expert** — only ~83 demos; target is
  absolute EEF pose from a single teleoperator, so likely near-unimodal → MSE is
  workable and far more data-efficient/debuggable than the DiT flow head. Upgrade
  to a flow head (v1) only if rollouts look averaged/hesitant.
- **Absolute `eef_pos_base` target (not delta actions)** — cleaner (delta field is
  quantized + 0.3-scaled), and "go to this position" generalizes to the varying
  pick/place locations. At deploy: `delta = pred_eef_pos_base − current_TCP` → UR `servoL`.
- **`table_cam` only for the world model** — LaWM assumes a stable camera; `wrist_cam`
  moves with the arm and would corrupt the latent action (paper §5 limitation).
  `wrist_cam` may later be added as an *auxiliary* input to the action head only.
- **gap = horizon = 32** — matches the LaWM's native τ=1.6 s at 20 Hz.
- **Everything frozen except prior + head** — reuse the validated LaWM; small
  trainable surface for a small dataset.

## 6. Environment & how to run

- Python: `/home/iclu200/miniconda3/envs/lawam/bin/python` (conda env `lawam`).
- **GPU: RTX 5090 (sm_120)** → requires **torch 2.7.1 + cu128** (2.6/cu124 has no
  sm_120 kernels; upgraded already). torchvision 0.22.1.
- Always run **as a module from the repo root** (imports `latent_action_model`):
  ```bash
  cd /home/iclu200/reinaldoyang/LaWAM
  CUDA_VISIBLE_DEVICES=0 /home/iclu200/miniconda3/envs/lawam/bin/python -m mini_lawam.train \
      --hdf5 dataset/multi_egg.hdf5 --steps 300 --eval-every 100 --log-every 20
  ```
- Model shape smoke test: `python -m mini_lawam.model`
- Checkpoint saved to `results/mini_lawam/ckpt.pt` = `{prior, action_head, cfg,
  action_mean, action_std, step}`.

### Frozen-LAM dependency (must be present)
- LAM checkpoint + yaml: `latent_action_model/logs/dino_large_vae/lam_release/`
  (`checkpoints/pytorch_model.pt`, `dino_large_vae.yaml`).
- The yaml's `model.vision_model_id` must point at a **HF-format DINOv3 ViT-B/16**.
  The gated HF repo (`facebook/dinov3-vitb16-pretrain-lvd1689m`) was **converted
  locally** from the `.pth` into `weights/dinov3-vitb16-pretrain-lvd1689m/` via
  `scripts/dinov3_convert/convert_local.py` (validated against HF reference outputs).
  See the repo-root `summary.md` for how to regenerate or revert to official weights.

## 7. Stage-1 validation status (already done)

Before building Stage 2 we validated the **pretrained LaWM** on this data with
`scripts/eval_lam_on_dataset.py` (metrics + Fig-8-style heatmaps). On `multi_egg`
(UR7e), motion-region at gap 32: `rollout≈0.74 > init≈0.58` (`roll−init≈+0.16`) and
`rollout ≫ shuffled≈0.66` → the LaWM **transfers to the UR7e domain**, but the
subgoal is **"soft but usable"** (weaker than the paper's Franka/LIBERO regime;
blob localizes the arm but doesn't strongly lead it). Conclusion: good enough to
condition a BC policy; don't expect a huge dynamics-conditioning win. Lever to
sharpen it if needed: crop `table_cam` to the workspace so the arm/objects fill
more of the frame.

## 8. Current status & next steps

- **Status:** scaffold complete, **not yet trained** (only the shape smoke test path
  exists). Committed to git (`aa64811`).
- **Next:** run the 300-step training smoke, confirm `loss_act`/`loss_distill` fall,
  then a full run.
- **Deployment (later):** write a UR7e rollout adapter — call `model.predict(o_t)`,
  un-normalize (`pred*std + mean`), convert to `servoL` delta, threshold gripper at 0.

## 9. Known upgrade paths (if v0 underperforms)

- ConvPrior → **attention-query prior** (QFormer-style, mirrors repo's
  `VLMToLAMQFormer`) if `loss_distill` plateaus high — global token relations suit
  the varying-location task.
- MLP head → **flow-matching head** (adapt repo's `ConditionalFlowMatchingHead`,
  drop VLM/CFG/physical-time) if rollouts show multimodal averaging.
- Random-frame val split → **held-out-demo split** to measure location generalization.
- **Cache DINO features to disk** to speed up training (DINO currently runs every step).
