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

## Evaluation

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

## Real robot evaluation
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

### Run camera + robot 
```bash
cd /home/iclu200/reinaldoyang/LaWAM
CUDA_VISIBLE_DEVICES=0 /home/iclu200/miniconda3/envs/lawam/bin/python -m mini_lawam.rollout_ur7e \
  --table-cam-serial 244422300964 \
  --robot-ip 140.96.93.125 \
  --execute --use-gripper-control \
  --trace-dir results/mini_lawam/traces
```

### Run camera + robot, with interpolation for smoother movement
```bash
CUDA_VISIBLE_DEVICES=0 /home/iclu200/miniconda3/envs/lawam/bin/python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_phase2.pt \
  --table-cam-serial 244422300964 \
  --robot-ip 140.96.93.125 --execute --use-gripper-control \
  --max-reach 0.005 --target-deadband 0.004 --target-ema 0.3 \
  --servol-max-pos-step 0.001 \
  --trace-dir results/mini_lawam/traces --show-camera
```

### use wrist cam + table cam
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
  --ckpt results/mini_lawam/ckpt_mult_egg_30_moved_256.pt \
  --table-cam-serial 244422300964 --wrist-cam-serial 252122300792 \
  --robot-ip 140.96.93.125 --execute --use-gripper-control \
  --train-frame-hw 0 0 \
  --temporal-ensemble --te-m 0.1 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 --servol-max-pos-step 0.002 \
  --trace-dir results/mini_lawam/traces --show-camera
```

## Train 2 Phase: ConvPrior and Action expert
### Phase 1: distill the ConvPrior (run once; reused by both phase-2 variants)
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train --hdf5 dataset/multi_egg_30_moved_256.hdf5 \
    --phase 1 --steps 10000 --out results/mini_lawam/phase1_moved_256.pt
```

### Phase 2 (without wrist cam) — reuse the phase-1 prior above
```bash
CUDA_VISIBLE_DEVICES=1 python -m mini_lawam.train --hdf5 dataset/multi_egg_30_moved_256.hdf5 \
    --phase 2 --prior-ckpt results/mini_lawam/phase1_moved_256.pt \
    --steps 10000 --batch 32 \
    --out results/mini_lawam/ckpt_moved_256.pt \
    --csv-log results/mini_lawam/train_log_moved_256.csv
```

### Phase 2 (with wrist cam) — SAME phase-1 prior
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train --hdf5 dataset/multi_egg_30_moved_256.hdf5 \
    --phase 2 --use-wrist --prior-ckpt results/mini_lawam/phase1_moved_256.pt \
    --steps 10000 --batch 32 \
    --out results/mini_lawam/ckpt_moved_256_wrist.pt \
    --csv-log results/mini_lawam/train_log_moved_256_wrist.csv
```

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


## Attention head experiment

Experiment 1 — attn head, isolate the un-pooling fix (your existing phase-1 prior, no proprioception, lower LR since transformers are LR-sensitive):
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.train \
  --hdf5 dataset/multi_egg_30_moved_256.hdf5 --phase 2 --head attn --use-wrist \
  --prior-ckpt results/mini_lawam/phase1_moved_256.pt \
  --steps 10000 --batch 32 --lr 1e-4 \
  --out results/mini_lawam/ckpt_attn_256.pt --csv-log results/mini_lawam/log_attn.csv
```

Experiment 2: add proprioception



Deployment: run the rollout
```bash
CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e \
  --ckpt results/mini_lawam/ckpt_attn_256.pt \
  --table-cam-serial 244422300964 --wrist-cam-serial 252122300792 \
  --robot-ip 140.96.93.125 --execute --use-gripper-control --train-frame-hw 0 0 \
  --temporal-ensemble --te-m 0.1 --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.02 --servol-max-pos-step 0.002 --trace-dir results/mini_lawam/traces --show-camera
```