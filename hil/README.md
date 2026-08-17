# Mini-LaWAM VR HIL correction collector

This folder is intentionally self-contained. It does not modify the normal
`mini_lawam/rollout_ur7e.py` or `mini_lawam/rollout_ur7e_vr.py` entrypoints.
It uses `MiniLaWAMPolicy` only as the frozen autonomous base policy.

Coding agents and maintainers should also read [`HANDOFF.md`](HANDOFF.md) for
the design invariants, module map, safety sequencing, current limitations,
artifact snapshot, and remaining gated-rollout work.

## Ownership behavior

At every 20 Hz collection step, Mini-LaWAM predicts the action it would take.
The Quest side grip then selects who controls the robot:

```text
side grip released -> execute Mini-LaWAM action
side grip held     -> execute VR target with full manual ownership
side grip released -> resume Mini-LaWAM from the last VR command target
```

The existing Quest APK protocol is used unchanged. In that protocol `grip` is
the selected controller's side-grip signal. The front index trigger toggles the
gripper while VR owns control.

Following the `lapa_kunn/real_world/hil` labeling logic, ownership and active
correction are separate. Every side-grip-held frame has
`manual_control_mask=True`, but `intervene_mask=True` only when the executed VR
XYZ/RZ step exceeds its configured deadband or the front trigger produces a
gripper-toggle edge. The robot remains fully under VR control while the grip is
held even when `intervene_mask=False`.

When requested, the autonomous branch uses the same temporal ensemble,
gripper-open lookahead/latch, target EMA/deadband, command-relative joystick
anchor, RZ composition, workspace clamp, and measured-TCP reach clamp as
`mini_lawam.rollout_ur7e_vr`. These settings continue to update during VR
takeover so every correction retains a same-frame rollout-policy baseline.

Quest face buttons:

| Button | Command |
|---|---|
| A | Start episode |
| Y | Save episode and go home |
| X | Discard episode and go home |
| B | Go home while idle |

Keyboard fallbacks are `Z`/`S` start, `V` save, `X` discard, `H` home, and
`Q`/Escape quit.

## Quest setup

Use the APK and ADB tunnel from `QuestUR7eTeleop`:

```bash
adb forward tcp:5555 tcp:5555
adb shell am start -n \
  com.DefaultCompany.QuestUR7eTeleop/com.unity3d.player.UnityPlayerActivity
```

The expected TCP row is:

```text
px,py,pz,qx,qy,qz,qw,trigger,grip,a,b,x,y
```

## Dry run first

Dry run still requires the cameras, Quest stream, and Mini-LaWAM checkpoint,
but does not connect to the UR7e:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hil.collect_corrections \
  --ckpt results/mini_lawam/checkpoint/vr_controller/ckpt_new_vr_teleop_egg_rz_103ep_256_attn_rz_binary_grip_t1.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --train-frame-hw 240 320 \
  --action-scale 1.0 \
  --enable-rz \
  --output-dir dataset/hil_mini_lawam_vr_active_dry_run \
  --output-file dry_run_active_v2.hdf5
```

Check the axis mapping, side-grip handoff, RZ sign, gripper toggle, workspace,
and saved HDF5 structure before enabling the arm.

## Robot collection

The HIL equivalent of the working `rollout_ur7e_vr` configuration is:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hil.collect_corrections \
  --ckpt results/mini_lawam/checkpoint/vr_controller/ckpt_new_vr_teleop_egg_rz_103ep_256_attn_rz_binary_grip_t1.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 \
  --wrist-exposure 100 --wrist-gain 16 \
  --train-frame-hw 240 320 \
  --image-height 256 --image-width 256 \
  --robot-ip 140.96.93.7 \
  --execute --use-gripper-control \
  --temporal-ensemble --te-m 0.2 \
  --gripper-open-lead-steps 0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --position-scale 1.2 \
  --max-linear-speed 0.2 \
  --rz-scale -1.0 \
  --max-angular-speed 0.5 \
  --intervention-translation-deadband 0.0005 \
  --intervention-rz-deadband 0.002 \
  --max-reach 0.015 \
  --ws-min -0.165 -0.164 0.158 \
  --ws-max 0.54 0.63 0.518 \
  --servol-max-pos-step 0.002 \
  --servol-max-rot-step 0.005 \
  --show-camera \
  --action-scale 1.0 --enable-rz \
  --output-dir dataset/hil_mini_lawam_vr_active \
  --output-file hil_corrections_active1.hdf5
```

`--max-reach` and `--max-target-lead` are aliases in this collector. The
Quest-only mapping and velocity options have no counterpart in the autonomous
rollout. Their HIL defaults match the VR demonstration recorder (`1.2`
position scale, `0.2 m/s` linear limit, `-1.0` RZ scale, and `0.5 rad/s`
angular limit). The `--vr-*` spellings and recorder spellings shown above are
aliases.

The intervention deadbands affect labels only; they do not suppress or scale
VR robot commands. A gripper toggle is always labeled as a correction while VR
owns control. Use the separate `hil_mini_lawam_vr_active` output so these
active-input labels are not mixed with older files where every held frame was
positive. The writer also refuses to append this schema to an old file.

Robot motion requires the literal `--execute` flag. A stale Quest stream stops
the servo worker and saves the valid partial episode with outcome
`aborted_vr_watchdog`; it never silently hands control back to the policy.

These Python checks are not safety-rated. Keep the UR safety planes, reduced
mode, speed slider, physical E-stop, and collision-free workspace configured.

## HDF5 layout

Each output file contains append-only `data/demo_N` groups:

```text
data/demo_N/
  obs/
    table_cam             uint8 [N,H,W,3]
    wrist_cam             uint8 [N,H,W,3]
    eef_pos_base          float32 [N,3]
    eef_quat_base         float32 [N,4], wxyz
    joint_pos             float32 [N,6]
    gripper_state         float32 [N,1]
    base_policy_action    float32 [N,7]
    bc_action             compatibility hard link to base_policy_action
    quest_controller      float32 [N,9]
  base_policy_actions     float32 [N,7]
  bc_actions              compatibility hard link to base_policy_actions
  base_actions            legacy hard link to base_policy_actions
  human_delta_actions     float32 [N,7]
  executed_actions        float32 [N,7]
  actions                 hard link to executed_actions
  residual_targets        float32 [N,7]
  manual_control_mask     bool [N]
  intervene_mask          bool [N]
  gripper_labels          int64 [N], 0=open / 1=close
  timestamps              float64 [N]
```

Actions use forward commanded-pose deltas:
`[dx,dy,dz,dRx,dRy,dRz,gripper]`, with metres, radians, and gripper
`-1=open/+1=close`. `manual_control_mask` records VR ownership.
`intervene_mask` records active VR motion or a gripper-toggle edge. Arm
`residual_targets` is `executed_actions - base_policy_actions` only on active
arm-correction frames; its gripper channel is populated only on toggle frames,
and all other residual components are zero.

Inspect these values frame by frame with:

```bash
python3 scripts/view_hdf5_gui.py \
  --input dataset/hil_mini_lawam_vr_active/hil_corrections_32ep.hdf5
```

For HIL files, the side panel shows the current controller owner, raw Quest
controller vector, base-policy action, executed action, human VR action,
residual target, intervention/manual masks, and binary gripper label. Older
files containing only the `bc_actions` name are displayed the same way.

### Analyze correction activity

Report correction counts per episode and across the complete HDF5 file:

```bash
python3 -m hil.analyze_hdf5 \
  dataset/hil_mini_lawam_vr_active/hil_corrections_30ep.hdf5
```

A `correction event` is one contiguous run of `intervene_mask=True`, not one
individual 20 Hz frame. The report separately counts VR takeovers as contiguous
`manual_control_mask=True` side-grip holds, because one takeover can contain
several correction bursts separated by short pauses. Use `--summary-only` to
hide episode rows, or `--json-output results/hil/correction_report.json` to
save the complete report.

### Removing bad episodes

In the viewer, navigate to an unwanted episode and click **Delete demo** (or
press the Delete key), then confirm. The first deletion creates a sibling file
such as `corrections_pruned.hdf5` and leaves the original dataset untouched.
Further deletions in that viewer session safely update the pruned copy shown in
the window.

For batch removal from a terminal, first list the episode names and their
frame/intervention counts:

```bash
python3 -m hil.delete_episodes dataset/hil_mini_lawam_vr/corrections.hdf5 --list
```

Remove individual episodes or inclusive ranges. By default this leaves the
input untouched and writes `corrections_pruned.hdf5`:

```bash
python3 -m hil.delete_episodes \
  dataset/hil_mini_lawam_vr/corrections.hdf5 \
  --episodes 2 5-7 demo_10
```

Use `--output PATH` to select another output. `--renumber` makes the retained
groups consecutive from `demo_0`. If an in-place replacement is needed, pass
`--in-place`; the successful operation keeps the original file as
`corrections.hdf5.bak` and refuses to proceed if that backup already exists.

## Stage 1: arm and gripper correction

The Stage 1 correction space reflects the robot's controllable task space:

```text
arm residual       [dx, dy, dz, dRz]       canonical indices [0,1,2,5]
gripper correction OPEN | CLOSE            class labels 0 | 1
```

Roll and pitch stay locked, so `dRx` and `dRy` are deliberately excluded from
the learned arm head. The default five-dimensional low-dimensional context is
the Mini-LaWAM base command `[dx,dy,dz,dRz]` plus the current gripper state.
Both table and wrist images provide the visual context. Mini-LaWAM itself is
not fine-tuned: its same-frame action was already recorded as
`base_policy_actions` (`bc_actions` remains a compatibility hard link).
The gripper target is the human-executed absolute state: executed value `-1`
maps to OPEN (`0`) and `+1` maps to CLOSE (`1`). It is not defined by whether
the human gripper command differs from the policy command, so there is no
third no-change class. The loader derives this target from `executed_actions`,
which also makes previously collected HDF5 files usable without recollection.

Train the two Stage 1 heads with:
first run with the deterministic output with one frame context first, and check if the validation is better
```bash
CUDA_VISIBLE_DEVICES=0 python -m hil.train_stage1 \
  --data-dir dataset/hil_mini_lawam_vr_active/hil_corrections_active_v2.hdf5 \
  --output-dir results/hil/mini_lawam_stage1_active_v2_det_ctx1_xyz005 \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 8 \
  --residual-samples intervention_only \
  --split-unit demo \
  --low-dim-mode image_bc_xyz_rz_grip \
  --image-encoder resnet18_spatial \
  --image-pretrained \
  --freeze-image-backbone \
  --spatial-keypoints 32 \
  --temporal-context 1 \
  --action-head-type deterministic \
  --gripper-loss-weight 0.01 \
  --selection-metric arm_physical_mae \
  --max-xyz-residual-per-step 0.005 \
  --max-rz-residual-per-step 0.05
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m hil.train_stage1 \
  --data-dir dataset/hil_mini_lawam_vr_active/hil_corrections_active_v2.hdf5 \
  --output-dir results/hil/mini_lawam_stage1_active_v2_gmm_xyz005 \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 8 \
  --residual-samples intervention_only \
  --split-unit demo \
  --low-dim-mode image_bc_xyz_rz_grip \
  --image-encoder resnet18_spatial \
  --image-pretrained \
  --freeze-image-backbone \
  --spatial-keypoints 32 \
  --temporal-context 4 \
  --action-head-type gmm \
  --num-gmm-modes 5 \
  --gripper-loss-weight 0.01 \
  --selection-metric arm_physical_mae \
  --max-xyz-residual-per-step 0.005 \
  --max-rz-residual-per-step 0.05
```

The trainer splits whole demonstrations by default, then selects intervention
frames for Stage 1. Arm targets are divided by the configured XYZ/RZ clips and
clamped to `[-1,1]`; this experiment uses a five-mode GMM arm head trained by
negative log likelihood, while the gripper head uses two-class cross-entropy.
The fresh output directory deliberately avoids resuming or overwriting the
deterministic checkpoint. The trainer writes `residual_stage1.pt` (best
validation arm MAE), `residual_stage1_last.pt`, and a JSON training history
under the output directory.

Pass one exact HDF5 file or an isolated directory. The loader recursively reads
every `.hdf5`/`.h5` below a directory, so a folder containing source, merged,
resized, or pruned variants would count those variants as separate data.

## Stage 2: intervention gate

Stage 2 follows the reference HIL implementation: it loads the Stage 1
checkpoint and freezes the image encoders, fusion network, temporal GRU, arm
head, and gripper head. Only a new two-class `gate_head` is trained:

```text
gate class 0 = use the Mini-LaWAM base action
gate class 1 = apply the Stage 1 arm/gripper correction
```

Unlike Stage 1, Stage 2 uses every frame. Its target is `intervene_mask`, which
is true for active VR motion or a gripper-toggle edge while the side grip is
held. To prevent the much more common non-intervention frames from dominating
training, every positive is retained and negatives are randomly retained at
the positive rate of each batch.

Train Stage 2 with the same seed, validation fraction, and demo split used by
Stage 1:

```bash
CUDA_VISIBLE_DEVICES=0 python -m hil.train_stage2 \
  --data-dir dataset/hil_mini_lawam_vr_active/hil_corrections_active_v2.hdf5 \
  --stage1-checkpoint results/hil/mini_lawam_stage1_active_v2_gmm_xyz005/residual_stage1.pt \
  --output-dir results/hil/mini_lawam_stage2_gate_active_v2 \
  --epochs 50 \
  --batch-size 8 \
  --num-workers 8 \
  --split-unit demo \
  --seed 101 \
  --val-fraction 0.1 \
  --gate-eval-threshold 0.3 \
  --selection-metric f1
```

The trainer writes `residual_gate_stage2.pt` (best validation F1),
`residual_gate_stage2_last.pt`, and a JSON history. Stage 2 only learns when a
correction should be used.

## Gated rollout

`hil.rollout_gated` runs Mini-LaWAM and the Stage 2 checkpoint together. When
the gate is active, it adds the Stage 1 XYZ/RZ residual to the safe current
Mini-LaWAM action and replaces the gripper with the predicted OPEN/CLOSE class.
The composed target is clamped again before it reaches the servo worker.

Run without `--execute` first, then use the same command with `--execute` only
after checking the camera inputs, gate transitions, residuals, and workspace:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hil.rollout_gated \
  --ckpt results/mini_lawam/checkpoint/vr_controller/ckpt_new_vr_teleop_egg_rz_103ep_256_attn_rz_binary_grip_t1.pt \
  --stage2-checkpoint results/hil/mini_lawam_stage2_gate_active_v2/residual_gate_stage2.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 \
  --wrist-exposure 100 --wrist-gain 16 \
  --train-frame-hw 240 320 \
  --correction-source-frame-hw 256 256 \
  --correction-frame-hw 256 256 \
  --robot-ip 140.96.93.7 \
  --use-gripper-control \
  --temporal-ensemble --te-m 0.2 \
  --gripper-open-lead-steps 0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --gate-threshold 0.3 --gate-hysteresis 0.05 \
  --max-reach 0.015 \
  --ws-min -0.165 -0.164 0.158 \
  --ws-max 0.54 0.63 0.518 \
  --servol-max-pos-step 0.002 \
  --servol-max-rot-step 0.005 \
  --show-camera --show-subgoal \
  --subgoal-update-steps 8 \
  --trace-dir results/hil/gated_rollout_traces \
  --action-scale 1.0 --enable-rz
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m hil.rollout_gated \
  --ckpt results/mini_lawam/checkpoint/vr_controller/ckpt_new_vr_teleop_egg_rz_103ep_256_attn_rz_binary_grip_t1.pt \
  --stage2-checkpoint results/hil/mini_lawam_stage2_gate/residual_gate_stage2.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 \
  --wrist-exposure 100 --wrist-gain 16 \
  --train-frame-hw 240 320 \
  --correction-source-frame-hw 256 256 \
  --correction-frame-hw 256 256 \
  --robot-ip 140.96.93.7 \
  --use-gripper-control \
  --temporal-ensemble --te-m 0.2 \
  --gripper-open-lead-steps 0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --gate-hysteresis 0.05 \
  --max-reach 0.015 \
  --ws-min -0.165 -0.164 0.158 \
  --ws-max 0.54 0.63 0.518 \
  --servol-max-pos-step 0.002 \
  --servol-max-rot-step 0.005 \
  --show-camera --show-subgoal \
  --subgoal-update-steps 8 \
  --action-scale 1.0 \
  --num-rollouts 10 \
  --trace-dir results/hil/gated_rollout_traces \
  --execute \
  --enable-rz
```

Keyboard controls are `S=start`, `E=end`, `H=stop and home`, and `Q=quit`.
Add `--execute` for real robot motion. The default gate threshold comes from
the Stage 2 checkpoint; specifying `--gate-threshold` overrides it. Hysteresis
keeps the gate active until its probability falls below `threshold - 0.05`.
`--trace-dir` saves the initial table frame and a summary JSON for every
rollout, using the same persistent `trial_N_TIMESTAMP` naming as Mini-LaWAM.
The initial table image records the object's starting position. Optional
`--save-frames 8` also saves the table frame every eight control steps, but
may add control-loop latency; it is unnecessary when only the initial position
is needed.

Run the hardware-free tests with:

```bash
python3 -m pytest hil/test_actions.py hil/test_vr.py hil/test_policy.py \
  hil/test_cli.py hil/test_data.py hil/test_stage1.py hil/test_stage2.py \
  hil/test_rollout_gated.py hil/test_analyze_hdf5.py
```
