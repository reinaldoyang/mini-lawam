REI LAWAM Documentation

## Installation
Follow the official README for installation and setup

If you don't have DINOv3 access, you can apply in the facebook webpage, and then use this converter to convert to huggingface format, note that this doesn't guarantee that it will become the same as the original DINOv3 from huggingface
```bash
cd /home/iclu200/reinaldoyang/LaWAM
CUDA_VISIBLE_DEVICES="" /home/iclu200/miniconda3/envs/lawam/bin/python \
  scripts/dinov3_convert/convert_local.py \
  --pth weights/dinov3-vitb16-pretrain-lvd1689m/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --save-dir weights/dinov3-vitb16-pretrain-lvd1689m
```

### View data
```bash
python3 scripts/view_hdf5_gui.py --input /home/iclu200/reinaldoyang/LaWAM/dataset/multi_egg_114ep.hdf5
```

to merge two hdf5 files into one, use the following command
```bash
python3 scripts/merge_hdf5.py \
  --inputs \
    dataset/hil_mini_lawam_vr_active/hil_corrections_active.hdf5 \
    dataset/hil_mini_lawam_vr_active/hil_corrections_17ep.hdf5 \
  --output dataset/hil_mini_lawam_vr_active/hil_corrections_34ep.hdf5
```

### Convert data image observation size to 256
ovx server path 
```bash
CUDA_VISIBLE_DEVICES=0 /home/ovxuser02@itriovx.local/miniconda3/envs/lawam/bin/python     convert_hdf5_to_256.py     --in  dataset/new_vr_teleop_egg_rz_75ep.hdf5  --out dataset/new_vr_teleop_egg_rz_75ep_256.hdf5 --overwrite
```

local pc path
```bash
CUDA_VISIBLE_DEVICES=0 python convert_hdf5_to_256.py     --in  dataset/new_vr_teleop_egg_rz_103ep_combined.hdf5 --out dataset/new_vr_teleop_egg_rz_103ep_256.hdf5 --overwrite
```

```bash
CUDA_VISIBLE_DEVICES=0 python convert_hdf5_to_256.py     --in  dataset/new_vr_teleop_egg_locked_rz_101ep.hdf5  --out dataset/new_vr_teleop_egg_locked_rz_101ep_256.hdf5 --overwrite
```

## Evaluate pretrained LaWM

```bash
CUDA_VISIBLE_DEVICES=0 /home/iclu200/miniconda3/envs/lawam/bin/python   scripts/eval_lam_on_dataset.py 2>&1 | grep -avE "Materializing|it/s\]|Loading weights"
```
- **rollout_vs_gt**: LaWM's predicted subgoal ûT vs true future uT (higher = better)
- **init_vs_gt**: copying the current frame uT vs uT — the "do-nothing" baseline (how hard is the prediction / does the scene barely change?)
- **shuffled_vs_gt**: decode uT with someone else's latent action (z from a different, unrelated clip)
- **roll-init**: how much the world model beats plain copy (rollout_vs_gt − init_vs_gt)

### Evaluate on our own data
```bash
cd /home/iclu200/reinaldoyang/LaWAM
CUDA_VISIBLE_DEVICES=0 /home/iclu200/miniconda3/envs/lawam/bin/python \
  scripts/eval_lam_on_dataset.py --dump-heatmaps 6 --heatmap-gap 32 \
  2>&1 | grep -avE "Materializing|it/s\]|Loading weights"
```

To output heatmap
```bash
cd /home/iclu200/reinaldoyang/LaWAM
CUDA_VISIBLE_DEVICES=0 /home/iclu200/miniconda3/envs/lawam/bin/python \
  scripts/eval_lam_on_dataset.py --hdf5 dataset/multi_egg.hdf5 \
  --dump-heatmaps 6 --sequence 0 --heatmap-gap 32 \
  2>&1 | grep -avE "Materializing|it/s\]|Loading weights"
```

Manually pick anchor patch (better result than automatically selecting from patch that change the most)
```bash
cd /home/iclu200/reinaldoyang/LaWAM
/home/iclu200/miniconda3/envs/lawam/bin/python -m scripts.pick_anchor \
  --hdf5 dataset/multi_egg.hdf5 --demo 0 --frame 0
```

## Training

Train 2 Phase: ConvPrior and Action expert, to better understand the model, we divide the training into two phase

The action model uses the token-level attention head. The legacy pooled MLP
action head is no longer supported.
### Phase 1: distill the ConvPrior (run once; reused by both phase-2 variants)
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train --hdf5 dataset/new_100ep_multi_egg_exp_plate_256.hdf5 \
    --phase 1 --steps 10000 --out results/mini_lawam/phase1_new_100ep_multi_egg_exp_plate_256.pt
```

### To train attention head, with binary gripper head, and changed gripper timing (fixed)
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train \
  --hdf5 dataset/new_100ep_multi_egg_exp_plate_256.hdf5 \
  --phase 2 --head attn --gripper-head binary \
  --use-wrist --target joystick \
  --gripper-target-offset 1 \
  --include-tail-actions \
  --prior-ckpt results/mini_lawam/phase1_new_100ep_multi_egg_exp_plate_256.pt \
  --lambda-gripper 1.0 \
  --steps 10000 --batch 32 --lr 1e-4 \
  --out results/mini_lawam/ckpt_100ep_attn_joystick_binary_grip_t1.pt \
  --csv-log results/mini_lawam/log_100ep_attn_joystick_binary_grip_t1.csv
```

To train from VR demonstrations with RZ enabled, add `--include-rz` to the
joystick command above and use a new checkpoint name. The resulting action is
`[X, Y, Z, RZ, gripper]`; commands without this flag remain 4D. Training fails
fast if action column 5 is constant, since that dataset cannot teach RZ.

First create the matching visual prior if it does not already exist:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train \
  --hdf5 dataset/new_vr_teleop_egg_30ep_256.hdf5 \
  --phase 1 --steps 10000 --batch 32 \
  --out results/mini_lawam/phase1_new_vr_teleop_egg_exp_30ep.pt \
  --csv-log results/mini_lawam/log_phase1_vr_teleop_egg_exp_30ep.csv \
  --wandb --wandb-project mini_lawam \
  --run-name phase1-new-vr-teleop-egg-30ep
```

Then train the 5D action head:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train \
  --hdf5 dataset/new_vr_teleop_egg_30ep_256.hdf5 \
  --phase 2 --head attn --gripper-head binary \
  --use-wrist --target joystick --include-rz \
  --gripper-target-offset 1 --include-tail-actions \
  --prior-ckpt results/mini_lawam/phase1_new_vr_teleop_egg_exp_30ep.pt \
  --lambda-gripper 1.0 \
  --steps 10000 --batch 32 --lr 1e-4 \
  --out results/mini_lawam/ckpt_new_vr_teleop_egg_30ep_attn_rz_binary_grip_t1.pt \
  --csv-log results/mini_lawam/log_new_vr_teleop_egg_30ep_attn_rz_binary_grip_t1.csv \
  --wandb --wandb-project mini_lawam \
  --run-name phase2-new-vr-teleop-egg-30ep-attn-rz-binary-grip-t1
```

## Real Robot Rollout
### Check camera serial number
```bash
/home/iclu200/miniconda3/envs/lawam/bin/python -c "
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))"
```

### Run camera only dry run
```bash
cd /home/iclu200/reinaldoyang/LaWAM
CUDA_VISIBLE_DEVICES=0 /home/iclu200/miniconda3/envs/lawam/bin/python -m mini_lawam.rollout_ur7e \
  --table-cam-serial 244422300964
```

### use wrist cam + table cam, with interpolation for smoother movement
```bash
CUDA_VISIBLE_DEVICES=0 /home/iclu200/miniconda3/envs/lawam/bin/python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_mult_egg_30_moved_256.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --robot-ip 140.96.93.125 --execute --use-gripper-control \
  --max-reach 0.005 --target-deadband 0.004 --target-ema 0.3 \
  --servol-max-pos-step 0.001 \
  --trace-dir results/mini_lawam/traces \
  --show-camera
```

### use wrist cam + table cam + 256 image size + temporal ensemble for action chunking
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_new_100ep_multi_egg_exp_plate_attn_256.pt \
  --table-cam-serial 244422300964 --wrist-cam-serial 252122300792 \
  --robot-ip 140.96.93.125 --execute --use-gripper-control \
  --train-frame-hw 0 0 \
  --temporal-ensemble --te-m 0.1 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces --show-camera
```

### Add exposure and also use the resolution of the observation state during data collection 
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_100ep_attn_delta.pt \
  --table-cam-serial 244422300964 --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.125 --execute --use-gripper-control \
  --train-frame-hw 168 224 \
  --temporal-ensemble --te-m 0.3 --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces --show-camera
```

### use delta scale
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/checkpoint/delta_controller/ckpt_100ep_attn_joystick_binary_grip_t1.pt \
  --table-cam-serial 244422300964 --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.7 --execute --use-gripper-control \
  --train-frame-hw 168 224 \
  --temporal-ensemble --te-m 0.3 --delta-scale 1.0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.05 --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces --show-camera
```

### show subgoal at gui (most stable version but slower)
```bash
CUDA_VISIBLE_DEVICES=1 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_100ep_attn_delta.pt \
  --table-cam-serial 244422300964 --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.7 --execute --use-gripper-control \
  --train-frame-hw 168 224 \
  --temporal-ensemble --te-m 0.3 --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces --show-camera --show-subgoal --subgoal-alpha 0.55 \
  --subgoal-update-steps 8 --video-scale 2.0
```

### Rollout of joystick target (gripper state on the bowl is stuck)
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_100ep_attn_joystick.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.7 --execute --use-gripper-control \
  --train-frame-hw 168 224 \
  --temporal-ensemble --te-m 0.3 \
  --action-scale 0.305 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 --servol-max-pos-step 0.002 \
  --show-camera --show-subgoal --subgoal-alpha 0.55 \
  --subgoal-update-steps 8 --video-scale 2.0
```

### Rollout of joystick target with binary gripper head

```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/checkpoint/joystick_checkpoint/ckpt_100ep_attn_joystick_binary_grip_t1.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 \
  --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.7 \
  --execute --use-gripper-control \
  --train-frame-hw 168 224 \
  --temporal-ensemble --te-m 0.1 \
  --action-scale 0.28 \
  --gripper-threshold 0.0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.05 \
  --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces \
  --show-camera --show-subgoal \
  --subgoal-update-steps 8 \
  --gripper-open-lead-steps 0
```

### Bamboo checkpoint with temporal ensembling

```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_50ep_multi_egg_bamboo_attn_joystick_binary_grip.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 \
  --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.7 \
  --execute --use-gripper-control \
  --train-frame-hw 168 224 \
  --temporal-ensemble --te-m 0.1 \
  --action-scale 0.28 \
  --gripper-threshold 0.0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 \
  --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces \
  --show-camera --show-subgoal \
  --subgoal-update-steps 8 \
  --gripper-open-lead-steps 0
```

### Rollout with VR command-relative control

Use `mini_lawam.rollout_ur7e_vr` for VR datasets that store forward deltas
between consecutive commanded poses. Its persistent XYZ target starts at the
measured TCP, accumulates predicted deltas, and stays within `--max-reach` of
the current measured TCP. Start with the native `--action-scale 1.0` and a
small `--max-reach 0.015` target lead.

For a checkpoint trained with `--include-rz`, additionally pass `--enable-rz`.
This applies predicted RZ while roll/pitch remain locked. The flag is rejected
for 4D checkpoints such as the locked-RZ checkpoint below. The `_t1` checkpoint
stores `gripper_target_offset=1`; use `--gripper-open-lead-steps 0` during
rollout to avoid adding a second runtime lookahead step.

```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e_vr \
  --ckpt results/mini_lawam/checkpoint/vr_controller/ckpt_new_vr_teleop_egg_rz_103ep_256_attn_rz_binary_grip_t1.pt\
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 \
  --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.7 \
  --execute --use-gripper-control \
  --train-frame-hw 240 320 \
  --exec-steps 8 \
  --gripper-open-lead-steps 0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.015 \
  --ws-min -0.165 -0.164 0.158 \
  --ws-max 0.54 0.63 0.518 \
  --servol-max-pos-step 0.002 \
  --servol-max-rot-step 0.005 \
  --show-camera --show-subgoal \
  --subgoal-update-steps 8 \
  --trace-dir results/mini_lawam/traces \
  --action-scale 1.0 \
  --enable-rz
```
  --temporal-ensemble --te-m 1.0 \


## Evaluation
### Phase 1 evaluation
```bash
python -m mini_lawam.eval_prior --ckpt results/mini_lawam/prior_phase1.pt --hdf5 dataset/multi_egg_114ep.hdf5
```

### Phase 2 evaluation
```bash
python -m mini_lawam.rollout --mode eval --ckpt results/mini_lawam/ckpt_phase2.pt \
    --hdf5 dataset/multi_egg_114ep.hdf5
```

### Evaluate subgoal
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.viz_subgoal \
    --ckpt results/mini_lawam/prior_phase1.pt \
    --hdf5 dataset/multi_egg.hdf5 --demo demo_0 --t 40 80 120
```
