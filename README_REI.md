REI LAWAM Documentation

## Installation
Follow the official README for installation and setup

Only do this step if your application for DINOv3 on huggingface is pending or not accepted!
If you don't have DINOv3 access, you can apply in the facebook webpage, and then use this converter to convert to huggingface format, note that this doesn't guarantee that it will become the same as the original DINOv3 from huggingface
```bash
cd /home/iclu200/reinaldoyang/LaWAM
CUDA_VISIBLE_DEVICES="" /home/iclu200/miniconda3/envs/lawam/bin/python \
  scripts/dinov3_convert/convert_local.py \
  --pth weights/dinov3-vitb16-pretrain-lvd1689m/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --save-dir weights/dinov3-vitb16-pretrain-lvd1689m
```

This will convert pth file to a huggingface folder, dir: weights/dinov3-vitb16-pretrain-lvd1689m

## Smoke Test
```bash
cd /home/ovxuser02@itriovx.local/reinaldoyang/lawam_rei
CUDA_VISIBLE_DEVICES=0 /home/ovxuser02@itriovx.local/miniconda3/envs/lawam/bin/python \
  -m mini_lawam.train \
  --hdf5 /home/ovxuser02@itriovx.local/reinaldoyang/dataset/multi_egg_83ep.hdf5 \
  --steps 300 --eval-every 100 --log-every 20

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