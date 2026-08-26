# Stage-1 LAM Fine-Tuning

This guide covers domain fine-tuning of the released LaWAM latent action model
(LAM) on local UR7e demonstrations. It applies only to the Stage-1 code under
`latent_action_model/`; it does not use or modify the Stage-2 `mini_lawam`
workflow.

The fine-tuning setup:

- initializes model weights from the released `dino_large_vae` LAM checkpoint;
- keeps the DINOv3 vision backbone frozen;
- trains the LAM encoder, VAE, visual decoder, and state decoder;
- converts the local robomimic HDF5 recordings into the LeRobot v3 layout used
  by the Stage-1 dataloader;
- holds out the last 10% of trajectories for validation; and
- saves the three checkpoints with the lowest `val_loss`, plus `last.ckpt`.

## 1. Prerequisites

Run commands from the repository root:

```bash
cd /home/iclu200/reinaldoyang/LaWAM
conda activate lawam
pip install -r requirements.txt
```

The following pretrained artifacts must exist:

```text
weights/dinov3-vitb16-pretrain-lvd1689m/
latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt
```

Check them before converting or training:

```bash
test -d weights/dinov3-vitb16-pretrain-lvd1689m
test -f latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt
```

The released checkpoint is loaded strictly as model weights. Optimizer,
scheduler, epoch, global-step, loop, and callback state are not restored when a
new fine-tuning run starts.

## 2. Selected dataset and expected source data

This guide uses:

```text
dataset/vr_teleop/new_vr_teleop_egg_rz_103ep_256.hdf5
```

Its recorded metadata and contents have been checked:

- 103 VR-teleoperation episodes and 23,011 total frames;
- 164 to 299 frames per episode;
- 256 x 256 table-camera and wrist-camera images;
- 20 Hz recording frequency;
- seven-dimensional actions; and
- end-effector quaternions stored in **WXYZ** order.

With `val_tail_ratio: 0.1`, episodes 0-92 are used for training and the final 10
episodes, 93-102, are used for validation.

The converter expects one robomimic HDF5 file with episodes under
`data/demo_*`. Every episode must provide:

| HDF5 path | Expected shape | Meaning |
| --- | --- | --- |
| `obs/table_cam` | `[T, H, W, 3]`, `uint8` | Table-camera RGB frames |
| `obs/eef_pos_base` | `[T, 3]` | End-effector XYZ position in the base frame |
| `obs/eef_quat_base` | `[T, 4]` | End-effector quaternion; this dataset uses WXYZ |
| `actions` | `[T, >=7]` | Six motion commands plus gripper command |

The selected file has already been resized to `256 x 256`, which matches the
Stage-1 training configuration.

### Gripper-state limitation

The current recordings do not contain measured gripper position. The converter
therefore uses `actions[:, 6]`, the latched gripper command, as the seventh
state value. This is a proxy rather than measured proprioception, and the
generated `meta/info.json` records that limitation.

## 3. Convert HDF5 to LeRobot v3

The registered dataset mixture expects the converted dataset at
`dataset/ur_lam_finetune`.

First, optionally convert one episode to a temporary directory as a smoke test:

```bash
python -m latent_action_model.convert_robomimic_to_lerobot \
  --input dataset/vr_teleop/new_vr_teleop_egg_rz_103ep_256.hdf5 \
  --output /tmp/ur_lam_smoke \
  --fps 20 \
  --task "VR teleoperation egg manipulation" \
  --quaternion-order wxyz \
  --max-episodes 1
```

Then convert the complete dataset:

```bash
python -m latent_action_model.convert_robomimic_to_lerobot \
  --input dataset/vr_teleop/new_vr_teleop_egg_rz_103ep_256.hdf5 \
  --output dataset/ur_lam_finetune \
  --fps 20 \
  --task "VR teleoperation egg manipulation" \
  --quaternion-order wxyz
```

Do not omit `--quaternion-order wxyz` for this dataset. Its HDF5 metadata states
that `eef_quat_base` is `(w, x, y, z)`, while the converter's default is XYZW.
For a different recording, replace `--input` and `--task`, then verify its
quaternion convention before conversion.

```bash
python -c 'import h5py; f=h5py.File("YOUR_DATASET.hdf5"); print(dict(f["meta"].attrs))'
```

The converter is intentionally non-destructive: it refuses to run when the
output path already exists. Choose a new output path, or move the previous
conversion elsewhere before trying again.

The output contains:

```text
dataset/ur_lam_finetune/
├── data/chunk-000/file-000.parquet
├── meta/episodes/chunk-000/file-000.parquet
├── meta/info.json
├── meta/modality.json
├── meta/stats_gr00t.json
├── meta/tasks.parquet
└── videos/observation.images.table_cam/chunk-000/file-*.mp4
```

### Inspect the converted LeRobot dataset

Before training, decode an episode and print its schema and values without
opening a window:

```bash
python scripts/view_lerobot_gui.py \
  --dataset dataset/ur_lam_finetune \
  --episode 0 \
  --check
```

Open the interactive viewer:

```bash
python scripts/view_lerobot_gui.py \
  --dataset dataset/ur_lam_finetune
```

The viewer displays the raw camera image, task, timestamp, named state vector,
and named action vector. Use the buttons or keyboard controls:

- Left/Right: previous/next frame
- Up/Down or Page Up/Page Down: previous/next episode
- Home/End: first/last frame
- Space: play/pause at the recorded FPS
- Episode text box: jump directly to an episode index

Choose an initial episode or a different playback rate with:

```bash
python scripts/view_lerobot_gui.py \
  --dataset dataset/ur_lam_finetune \
  --episode 25 \
  --playback-fps 10
```

The viewer reads raw Parquet and MP4 data. It intentionally does not apply the
training-time normalization, random crop, or augmentation.

At 20 Hz, the configured `frame_dt_sec: 1.6` samples frames 32 timesteps apart.
Episodes must be long enough to contain those two-frame training pairs.

## 4. Choose a fine-tuning configuration

Two configurations are provided:

| Configuration | Learning rate | Recommended use |
| --- | ---: | --- |
| `latent_action_model/config/ur_lam_finetune_lr1e5.yaml` | `1e-5` | First and safer run |
| `latent_action_model/config/ur_lam_finetune_lr3e5.yaml` | `3e-5` | Learning-rate comparison |

Both configurations match the released LAM architecture and losses. On one
GPU, `batch_size: 2` with `accumulate_grad_batches: 32` gives an effective batch
size of 64. With multiple GPUs, the global effective batch size is
`2 x 32 x number_of_GPUs`.

Before a long run, confirm that Lightning can parse the configuration:

```bash
python -m latent_action_model.main fit \
  --config latent_action_model/config/ur_lam_finetune_lr1e5.yaml \
  --print_config
```

## 5. Measure the pretrained baseline

Validate the released checkpoint on the held-out part of the converted dataset:

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
python -m latent_action_model.main validate \
  --config latent_action_model/config/ur_lam_finetune_lr1e5.yaml
```

Record `val_loss` and the `val/*` component losses. This is the baseline to
compare against fine-tuned checkpoints.

## 6. Run a one-batch training smoke test

Use Lightning's fast-development mode before committing to a full run:

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
python -m latent_action_model.main fit \
  --config latent_action_model/config/ur_lam_finetune_lr1e5.yaml \
  --trainer.fast_dev_run=true
```

This verifies model construction, strict checkpoint loading, data decoding, a
forward/backward pass, and validation. It does not produce a usable checkpoint.

## 7. Start fine-tuning

Stage-1 training consumes the converted LeRobot directory, not the source
HDF5 file directly. The two data arguments below resolve to:

```text
data_root_dir (`dataset`) + mixture dataset name (`ur_lam_finetune`)
= dataset/ur_lam_finetune
```

These values are already present in each YAML configuration, but they are
repeated on the command line so the selected training dataset is explicit.

Start with the `1e-5` run:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash latent_action_model/train.sh \
  --config latent_action_model/config/ur_lam_finetune_lr1e5.yaml \
  --data.data_root_dir dataset \
  --data.data_mix ur_lam_finetune
```

After reviewing that run, optionally launch the `3e-5` comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash latent_action_model/train.sh \
  --config latent_action_model/config/ur_lam_finetune_lr3e5.yaml \
  --data.data_root_dir dataset \
  --data.data_mix ur_lam_finetune
```

`train.sh` launches one process per CUDA device visible to PyTorch. For example,
to use GPUs 0 and 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash latent_action_model/train.sh \
  --config latent_action_model/config/ur_lam_finetune_lr1e5.yaml \
  --data.data_root_dir dataset \
  --data.data_mix ur_lam_finetune
```

The fine-tuning configurations use Lightning loggers for both TensorBoard and
W&B. The older manual W&B path is disabled to prevent duplicate metrics and
runs.

## 8. Monitor training and find checkpoints

Start TensorBoard from another terminal:

```bash
conda activate lawam
tensorboard --logdir latent_action_model/logs
```

### Weights & Biases

Both fine-tuning configurations log the following metrics to the W&B project
`lawam-stage1-lam`:

- `train_loss`, `train/recon_loss`, `train/state_loss`, and `train/lr`
- `val_loss`, `val/recon_loss`, and `val/state_loss`

`train.sh` defaults to `WANDB_MODE=offline`, so a login or network connection is
not required and an outage cannot stop training. Offline runs are stored under:

```text
latent_action_model/wandb/
```

When connectivity is available, upload an offline run with:

```bash
wandb login
wandb sync latent_action_model/wandb/offline-run-*
```

For live dashboard logging during training, authenticate first and override the
default mode:

```bash
wandb login

WANDB_MODE=online \
CUDA_VISIBLE_DEVICES=0 \
bash latent_action_model/train.sh \
  --config latent_action_model/config/ur_lam_finetune_lr1e5.yaml
```

Never put a W&B API key in `train.sh` or a YAML file. Use `wandb login` or an
environment variable. Model checkpoint upload is disabled because each LAM
checkpoint is several gigabytes; local checkpoint saving remains enabled.

The two experiments write checkpoints to separate directories:

```text
latent_action_model/logs/ur_lam_finetune_lr1e5/checkpoints/
latent_action_model/logs/ur_lam_finetune_lr3e5/checkpoints/
```

Use the checkpoint with the lowest `val_loss`, rather than assuming that the
last epoch is best. Compare it with the pretrained baseline from step 5.

## 9. Validate a fine-tuned checkpoint

Use both evaluations below. Lightning validation measures losses on the held-out
episodes, while the HDF5 diagnostic checks whether the learned latent action
actually improves future-feature prediction.

### 9.1 Select the best checkpoint

List the saved checkpoints:

```bash
ls -lh latent_action_model/logs/ur_lam_finetune_lr1e5/checkpoints/
```

Choose the checkpoint with the lowest `val_loss` in its filename or training
logs. Do not assume that `last.ckpt` is the best. Replace `YOUR_BEST.ckpt` in
the commands below with that filename.

### 9.2 Evaluate held-out validation losses

Load the fine-tuned checkpoint as weights only and evaluate it on the same last
10% of episodes held out during training:

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
python -m latent_action_model.main validate \
  --config latent_action_model/config/ur_lam_finetune_lr1e5.yaml \
  --model.pretrained_ckpt \
    "latent_action_model/logs/ur_lam_finetune_lr1e5/checkpoints/YOUR_BEST.ckpt"
```

This replaces the released initialization path for that command and still uses
strict weights-only loading. Compare `val_loss`, `val/recon_loss`, and
`val/state_loss` with the released-checkpoint baseline from Step 5. Lower is
better, and the comparison is meaningful only when both runs use the same YAML
and validation split.

### 9.3 Evaluate latent-action and world-model behavior

Run the fine-tuned LAM directly on frame pairs from the original HDF5 dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_lam_on_dataset.py \
  --hdf5 dataset/vr_teleop/new_vr_teleop_egg_rz_103ep_256.hdf5 \
  --ckpt \
    "latent_action_model/logs/ur_lam_finetune_lr1e5/checkpoints/YOUR_BEST.ckpt" \
  --yaml latent_action_model/config/ur_lam_finetune_lr1e5.yaml \
  --gaps 32 \
  --num-pairs 256 \
  --batch 32 \
  --seed 0
```

If evaluation runs out of GPU memory, reduce `--batch 32` to `--batch 8`.
Focus on the `MOTION-REGION` table:

- `rollout_vs_gt > init_vs_gt`: the fDM predicts the future better than copying
  the current observation.
- `rollout_vs_gt > shuffled_vs_gt`: the iDM latent action contains information
  specific to the observed transition.
- `roll-init` should be positive and preferably larger than for the released
  checkpoint.

This HDF5 diagnostic samples across the complete source dataset. Treat the
held-out Lightning validation above as the primary generalization measurement.

For an apples-to-apples pretrained comparison, keep `--gaps`, `--num-pairs`,
and `--seed` unchanged and run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_lam_on_dataset.py \
  --hdf5 dataset/vr_teleop/new_vr_teleop_egg_rz_103ep_256.hdf5 \
  --ckpt latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt \
  --yaml latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml \
  --gaps 32 \
  --num-pairs 256 \
  --batch 32 \
  --seed 0
```

### 9.4 Inspect predicted subgoals visually

Generate six heatmap panels using the fine-tuned checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_lam_on_dataset.py \
  --hdf5 dataset/vr_teleop/new_vr_teleop_egg_rz_103ep_256.hdf5 \
  --ckpt \
    "latent_action_model/logs/ur_lam_finetune_lr1e5/checkpoints/YOUR_BEST.ckpt" \
  --yaml latent_action_model/config/ur_lam_finetune_lr1e5.yaml \
  --gaps 32 \
  --num-pairs 256 \
  --batch 32 \
  --seed 0 \
  --dump-heatmaps 6 \
  --heatmap-gap 32 \
  --out-dir results/lam_check/ur_finetuned
```

The panels are written under `results/lam_check/ur_finetuned/`. Check that the
predicted subgoal emphasizes the moving robot/egg regions instead of only the
static background.

## 10. Evaluate LaWM rollout quality (`roll-init`)

Lightning validation reports reconstruction and state losses. For the
feature-space LaWM diagnostics used elsewhere in this repository, run
`scripts/eval_lam_on_dataset.py`. It reports:

- `rollout_vs_gt`: cosine similarity between the predicted and true future
  visual features; higher is better.
- `init_vs_gt`: the copy-current-frame baseline.
- `shuffled_vs_gt`: prediction after assigning each sample another sample's
  latent action; this should be lower than the correctly paired rollout.
- `roll-init`: `rollout_vs_gt - init_vs_gt`; positive and larger means the
  LaWM predicts future dynamics better than copying the current frame.

Evaluate the fine-tuned epoch-9 checkpoint on the original VR HDF5:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m scripts.eval_lam_on_dataset \
  --hdf5 dataset/base_vr_teleop/new_vr_teleop_egg_rz_103ep_256.hdf5 \
  --ckpt "latent_action_model/logs/ur_lam_finetune_lr1e5/checkpoints/epoch=epoch=09-val_loss=val_loss=0.21701.ckpt" \
  --yaml latent_action_model/config/ur_lam_finetune_lr1e5.yaml \
  --gaps 16 24 32 48 \
  --num-pairs 256 \
  --batch 8 \
  --topk 24 \
  --seed 0
```

Run the released checkpoint with the same data, gaps, sample count, batch size,
and seed for a before/after comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m scripts.eval_lam_on_dataset \
  --hdf5 dataset/base_vr_teleop/new_vr_teleop_egg_rz_103ep_256.hdf5 \
  --ckpt latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt \
  --yaml latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml \
  --gaps 16 24 32 48 \
  --num-pairs 256 \
  --batch 8 \
  --topk 24 \
  --seed 0
```

Use the **MOTION-REGION** table for the main comparison because whole-frame
similarity is diluted by the static background. Gap 32 corresponds to the
fine-tuning interval of 1.6 seconds at 20 Hz. A useful fine-tuned model should
improve `rollout_vs_gt` and `roll-init` over the released checkpoint while
keeping `rollout_vs_gt` clearly above `shuffled_vs_gt`. `init_vs_gt` should be
the same for both runs because it does not use the learned rollout.

Run this command from the repository root with `python -m` as shown. Directly
executing `python scripts/eval_lam_on_dataset.py` can fail to resolve the local
`latent_action_model` package in environments where the repository has not
been installed as a package.

## 11. Resume an interrupted run

Resuming is different from pretrained initialization. Set `CKPT_PATH` when you
need to restore the full Lightning training state, including optimizer,
scheduler, epoch, and global step:

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=offline \
CKPT_PATH="latent_action_model/logs/ur_lam_finetune_lr1e5/checkpoints/last.ckpt" \
bash latent_action_model/train.sh \
  --config latent_action_model/config/ur_lam_finetune_lr1e5.yaml
```

Do not set `CKPT_PATH` merely to start from the released checkpoint. The
configuration's `model.pretrained_ckpt` already performs the correct
weights-only initialization for a new run.

## 12. Focused verification tests

The implementation includes standalone tests for checkpoint behavior and data
conversion:

```bash
python latent_action_model/test_pretrained_loading.py
python latent_action_model/test_robomimic_to_lerobot.py
```

The conversion test creates synthetic robomimic episodes, converts them, and
loads the result through the actual Stage-1 `LeRobotLAMDataset`.

## Common problems

### `No CUDA devices are visible to PyTorch`

Check the active environment and visibility:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
echo "$CUDA_VISIBLE_DEVICES"
```

### Output path already exists

The converter will not overwrite a prior dataset. Use a different `--output`
path. If the final directory is not named `ur_lam_finetune`, update the dataset
name in `starVLA/dataloader/gr00t_lerobot/mixtures.py` or move the completed
conversion to the configured path.

### Strict checkpoint loading reports missing or unexpected keys

The checkpoint architecture and the YAML configuration do not match. Use the
released `dino_large_vae` checkpoint with one of the supplied UR fine-tuning
configurations, or make architecture changes deliberately and create a matching
checkpoint.

### Validation dataset is empty

The configurations reserve the final 10% of trajectories using
`val_tail_ratio: 0.1`. Convert enough episodes for both the training and
validation partitions, and ensure each episode is long enough for the 1.6-second
frame gap.
