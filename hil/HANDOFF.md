# Mini-LaWAM VR HIL engineering handoff

Last updated: 2026-08-12

Read this file before modifying `hil/`. The operator-facing setup, commands,
and complete HDF5 tree are in [`README.md`](README.md). This document records
the design intent, non-obvious invariants, implementation boundaries, current
workspace state, and unfinished work for another coding agent.

## Objective

The goal is to improve a frozen Mini-LaWAM robot policy using corrections
collected from a Meta Quest controller:

```text
normal operation       side grip released -> Mini-LaWAM owns the robot
human correction       side grip held     -> VR fully owns arm and gripper
resume autonomy        side grip released -> Mini-LaWAM resumes from the
                                             last executed VR target
```

The collected data supports a gated residual policy:

```text
Mini-LaWAM base action
        +
Stage 1: predict the arm/gripper correction
        +
Stage 2: decide whether the correction should be applied
```

Mini-LaWAM remains frozen. The HIL code is deliberately isolated in `hil/` and
must not change the behavior of `mini_lawam.rollout_ur7e.py` or
`mini_lawam.rollout_ur7e_vr.py`.

## Current implementation status

Implemented:

- Real and dry-run collection with Mini-LaWAM as the autonomous policy.
- Momentary VR takeover using the Quest side grip.
- Quest front-trigger binary gripper toggle during takeover.
- Same-observation base-policy actions and explicit intervention labels.
- Buffered episode writing to HDF5.
- Stage 1 arm-residual and binary gripper training.
- Stage 2 binary intervention-gate training.
- Gated dry-run/real-robot rollout with Stage 1 arm/gripper correction,
  Stage 2 gate hysteresis, temporal context, and post-composition safety clamp.
- HDF5 episode listing/removal and HIL-aware GUI inspection.
- Hardware-free unit tests for action math, VR clutching, policy ensembling,
  storage, and both training stages.

Known limitation:

- Collection is not dual-rate. VR commands are selected in the same blocking
  20 Hz loop that runs Mini-LaWAM inference, so takeover is less responsive
  than the direct 100 Hz VR teleoperation recorder.

The gated rollout is implemented but has not been declared real-robot
validated by the operator. It must be tested in dry run before use under the
robot's external safety systems.

## Module map

| File | Responsibility |
|---|---|
| `constants.py` | Canonical action dimensions, correction indices, gripper labels. |
| `actions.py` | Pose/action conversion, SO(3) rotation deltas, workspace and target-lead clamps. |
| `vr.py` | Quest TCP parser, button-edge queue, background receiver, side-grip clutch and speed limiting. |
| `robot.py` | RealSense readers, RTDE state, 500 Hz `servoL` worker, gripper latch, home sequencing. |
| `policy.py` | Frozen Mini-LaWAM adapter, temporal ensemble, rollout smoothing and safe one-step base action. |
| `data.py` | In-memory episode buffer and append-only HDF5 writer. |
| `collect_corrections.py` | 20 Hz ownership state machine and collection entrypoint. |
| `stage1_dataset.py` | HDF5 indexing, demo/frame split, temporal windows, low-dimensional inputs. |
| `stage1_model.py` | Two-view correction model and arm/gripper heads. |
| `train_stage1.py` | Stage 1 validation, training, metrics, resume checks and checkpoints. |
| `stage2_model.py` | Frozen Stage 1 model plus trainable binary gate. |
| `train_stage2.py` | Stage 2 split checks, negative sampling, training and checkpoints. |
| `gated_policy.py` | Stage 2 checkpoint loading, temporal inference, gate hysteresis and action composition. |
| `rollout_gated.py` | Separate gated Mini-LaWAM rollout and robot-control entrypoint. |
| `delete_episodes.py` | Recoverable episode removal with aggregate-count repair. |
| `test_*.py` | Focused hardware-free checks. |
| `../scripts/view_hdf5_gui.py` | Frame viewer with HIL actions, residuals, masks and Quest state. |

Hardware-heavy imports are intentionally deferred so CLI help and unit tests
can run without RTDE, RealSense, pygame, or the gripper driver.

## Collection control flow and invariants

The main loop in `collect_corrections.py` runs at `--control-hz` (20 Hz by
default). Camera and Quest reception use background threads; the RTDE servo
worker runs separately at 500 Hz.

For each recording step:

1. Read the latest camera pair, robot observation, and queued Quest samples.
2. Update the VR clutch for every queued Quest sample so button edges and
   velocity integration are retained.
3. On the side-grip rising edge, reset `command_pose` to the measured TCP and
   queue that hold pose **before inference**. This prevents the previous policy
   target from racing the takeover.
4. Run Mini-LaWAM even during takeover. This is required to record the base
   action for the same observation as the human correction.
5. Select the VR target while the side grip is held; otherwise select the
   Mini-LaWAM target.
6. Queue the selected pose, command the selected gripper state, and append one
   aligned frame to the in-memory episode.

Preserve these invariants:

- Ownership is determined only by the explicit side-grip state. Never infer
  intervention from `executed_action != base_policy_action`.
- A held side grip with zero human movement is still a positive intervention
  and may intentionally teach a zero residual.
- The base policy must continue from the last manual target after release.
  `policy.commit_manual_target()` and the shared `command_pose` provide this
  re-anchoring.
- `residual_targets` is zero outside intervention. During intervention it is
  exactly `executed_actions - base_policy_actions`.
- Roll and pitch remain locked. The stored canonical action is 7D, but the
  learned arm correction uses only indices `(0, 1, 2, 5)` = XYZ/RZ.
- Gripper commands are exact states: `-1=open`, `+1=close`. Training labels are
  binary absolute states: `0=open`, `1=close`; there is no `keep` class.
- `action_from_target()` measures motion from the previous commanded target to
  the next commanded target. It is not a delta from the measured TCP.
- Rotation changes must be composed in SO(3); do not subtract UR rotation
  vectors component-wise.
- The policy checkpoint may predict an action chunk, but collection stores
  only the rollout-selected current action for each 20 Hz frame.

## Safety sequencing

The Python safeguards are useful but are not safety-rated.

- Robot motion requires the literal `--execute` flag.
- Starting an episode arms the servo from the current measured TCP.
- Saving or discarding calls `stop_servo()` before writing data or calling
  `moveJ` home. `move_home()` also defensively stops the servo first. This
  ordering prevents a stale `servoL` target from racing the home command.
- The takeover rising edge publishes a measured-pose hold before inference.
- Workspace XYZ and maximum target lead from measured TCP are clamped for both
  policy and human targets.
- The servo worker independently interpolates and limits each 500 Hz position
  and rotation step.
- A stale Quest stream stops the servo and saves a nonempty partial episode as
  `aborted_vr_watchdog`; it does not silently return ownership to the policy.
- Runtime exceptions attempt to stop the servo and save a nonempty partial
  episode as `aborted_runtime_error`.
- Home is ignored during an active episode; the operator must save or discard.

Any new rollout must retain these properties and still rely on UR safety
planes, reduced mode/speed slider, a collision-free workspace, and a physical
E-stop.

## Action and HDF5 semantics

Canonical action order:

```text
[dx, dy, dz, dRx, dRy, dRz, gripper_state]
 metres                  radians      -1/open or +1/close
```

The essential per-frame relationship is:

```python
if intervene_mask[k]:
    executed_actions[k] = human_delta_actions[k]
    residual_targets[k] = executed_actions[k] - base_policy_actions[k]
else:
    executed_actions[k] = base_policy_actions[k]
    residual_targets[k] = 0
```

Important names:

- `base_policy_actions`: canonical name for the safe, post-processed
  Mini-LaWAM action at the same observation.
- `executed_actions`: selected command; VR during takeover, otherwise base.
- `human_delta_actions`: the VR forward-command action during takeover. Away
  from takeover its motion channels are zero and its gripper value is the
  observed state; use `intervene_mask` before interpreting it as a correction.
- `intervene_mask`: authoritative side-grip ownership label.
- `manual_control_mask`: currently identical to `intervene_mask`.
- `gripper_labels`: binary absolute executed gripper state.
- `obs/quest_controller`: raw
  `[px,py,pz,qx,qy,qz,qw,front_trigger,side_grip]`. A/B/X/Y are handled as
  controls but are not stored in this vector.

Compatibility:

- New files use `base_policy_actions` and `obs/base_policy_action`.
- `bc_actions`, `base_actions`, and `obs/bc_action` are HDF5 hard-link aliases,
  not duplicated arrays.
- Existing files in this workspace predate the rename and contain only the
  legacy `bc_actions` name. The Stage 1 loader, GUI, deletion utility, and
  aggregate counter deliberately support both schemas.
- `actions` is a hard-link alias of `executed_actions`.

Do not remove the aliases until all previously collected datasets and external
consumers have been migrated.

## Image sizing: three different concepts

Do not conflate these settings:

1. RealSense acquisition is `640x480` RGB at 30 Hz.
2. `--train-frame-hw 240 320` reproduces the original recording-size pre-resize
   before Mini-LaWAM receives its `256x256` tensor. It does not affect HDF5.
3. `--image-height` and `--image-width` control images saved in HDF5. Their
   defaults are `168x224`. Pass both as `256` to store square `256x256` images.

The Stage 1 encoders accept variable image sizes, but checkpoint provenance
must record which dataset variant was used.

## Stage 1 residual policy

The recommended configuration in `README.md` uses four frames of context:

```text
per timestep:
  table RGB -> independent frozen pretrained ResNet-18 -> 32 spatial keypoints -> 128D
  wrist RGB -> independent frozen pretrained ResNet-18 -> 32 spatial keypoints -> 128D
  low-dimensional input = base [dx,dy,dz,dRz] + current gripper = 5D
  concatenate -> fusion MLP -> 512D

four fused features -> GRU -> latest 512D temporal feature
                               |-> deterministic arm MLP -> 4 normalized residuals
                               `-> gripper MLP -> OPEN/CLOSE logits
```

The two ResNet instances do not share weights. With
`--freeze-image-backbone`, the ResNet backbones remain frozen, while spatial
projection, fusion, GRU, and heads train.

The temporal window is left-padded with the earliest available frame at the
start of a demonstration. It produces one correction for the newest frame,
not an action chunk.

Default training behavior:

- Split complete demonstrations before selecting frames.
- `--residual-samples intervention_only` trains both Stage 1 heads only on
  frames where `intervene_mask=True`.
- Arm target is the residual projected to `[dx,dy,dz,dRz]`, divided by
  `action_clip=[max_xyz,max_xyz,max_xyz,max_rz]`, then clipped to `[-1,1]`.
- Deterministic arm head: `tanh` output and Smooth-L1 loss (`beta=0.1`).
- Gripper head: two logits and cross-entropy against the absolute executed
  OPEN/CLOSE state.
- Total loss is `arm_loss + gripper_loss_weight * gripper_cross_entropy`;
  the documented command uses a gripper weight of `0.01`.
- A GMM arm head is available but is not used by the current checkpoint.

## Stage 2 intervention gate

Stage 2 loads the Stage 1 checkpoint and freezes the entire correction policy.
A three-hidden-layer MLP (128 units by default) maps the frozen 512D temporal
feature to two logits:

```text
class 0 -> use Mini-LaWAM unchanged
class 1 -> apply Stage 1 arm/gripper correction
```

It trains on every frame with `intervene_mask` as the target. All positives are
kept; batch negatives are randomly retained at the batch positive rate.
Training uses cross-entropy. Validation uses all frames and the configured
positive-probability threshold (documented as `0.3`). Stage 2 must reuse the
Stage 1 seed, validation fraction, and split unit; code checks this.

The Stage 2 checkpoint embeds the full frozen Stage 1 state plus
`low_dim_mean`, `low_dim_std`, `action_clip`, model configs, and gate metadata.
Its forward pass returns a normalized one-step arm residual, gripper logits,
and gate logits. It does not itself compose or safety-clamp a robot target.

## Gated rollout behavior

`rollout_gated.py` and `gated_policy.py` implement the deployment path:

1. Load Mini-LaWAM and Stage 2 and verify base checkpoint name, target mode,
   and action schema.
2. Reproduce correction-image resizing (`640x480 -> 168x224 -> 256x256` for
   the current converted training file), low-dimensional normalization,
   temporal left-padding, and the checkpoint's GRU context.
3. Run Mini-LaWAM with the same rollout post-processing used during collection.
4. Compute Stage 1 residual/gripper and Stage 2 gate from the same observation
   and safe base action.
5. Apply probability hysteresis: gate on at the checkpoint/CLI threshold and
   off at `threshold - gate_hysteresis`.
6. If gated off, preserve the complete Mini-LaWAM action and gripper state.
7. If gated on, denormalize using `action_clip`, add only XYZ and optional RZ,
   leave Rx/Ry unchanged, and use the Stage 1 OPEN/CLOSE class.
8. Convert the action to a command-relative target using SO(3), reapply
   workspace and measured-TCP lead clamps, then queue it to the servo worker.
9. Re-anchor Mini-LaWAM smoothing to an executed corrected target so autonomy
   continues from the command that actually won ownership.

Avoid applying a residual to the raw Mini-LaWAM chunk: Stage 1 was trained
against the safe, rollout-selected `base_policy_actions` stored by the HIL
collector.

The rollout emits one correction and one gate decision per 20 Hz step; neither
Stage 1 nor Stage 2 emits an action chunk. Use `S/E/H/Q` for start/end/home/quit.
`H` and shutdown stop and join the servo worker before `moveJ` or exit. Start
without `--execute`; real-robot validation remains an operator decision.

## Latency limitation

The direct VR recorder commands the robot around 100 Hz while saving images at
20 Hz. The HIL collector receives Quest poses in a background thread but
publishes selected targets in its 20 Hz main loop. It also runs Mini-LaWAM
before publishing subsequent human targets because same-frame base actions are
needed for residual labels.

Consequences:

- Direct VR command sampling delay is roughly `0-10 ms`.
- HIL sampling delay is roughly `0-50 ms + inference/preprocessing time`.
- The 500 Hz servo smooths the latest target but cannot remove target-update
  latency.
- Display/subgoal work can increase loop overrun; `--no-display` is a useful
  diagnostic.

Do not simply change collection to 100 Hz: model/data action semantics are 20
Hz and inference may not sustain 100 Hz. A proper improvement is a synchronized
dual-rate architecture: immediate 100 Hz VR/ownership commands plus 20 Hz
camera, base inference, and recording. It must retain exact observation/base/
human timestamp alignment and all takeover/home race protections.

## Dataset and artifact snapshot

As of this handoff, the local workspace contains:

| File | Episodes | Frames | Actual intervention frames | Stored RGB size |
|---|---:|---:|---:|---|
| `dataset/hil_mini_lawam_vr/hil_corrections_11ep.hdf5` | 11 | 3512 | 1241 | 168x224 |
| `dataset/hil_mini_lawam_vr/hil_corrections_13ep.hdf5` | 13 | 4110 | 1441 | 168x224 |
| `dataset/hil_mini_lawam_vr/hil_corrections_24ep.hdf5` | 24 | 7622 | 2682 | 168x224 |
| `dataset/hil_mini_lawam_vr/hil_corrections_24ep_256.hdf5` | 24 | 7622 | 2682 | 256x256 |

The 24-episode files are cumulative/derived variants of the smaller files.
The merged 24-episode files do not currently have the aggregate
`data.attrs[total_interventions]`, although their per-frame masks are intact
and the trainers count those masks directly.

**Training-data trap:** `find_hdf5_files()` recursively loads every `.hdf5` and
`.h5` under `--data-dir`. Passing `dataset/hil_mini_lawam_vr` therefore loads
overlapping cumulative and resized copies as separate training data. Pass one
exact file, for example:

```bash
--data-dir dataset/hil_mini_lawam_vr/hil_corrections_24ep_256.hdf5
```

The same warning applies to `_pruned.hdf5` outputs stored beside their source.
Use an exact file path or an isolated directory.

Current local checkpoints:

- Stage 1: `results/hil/mini_lawam_stage1_xyz_rz_grip/residual_stage1.pt`
- Stage 2: `results/hil/mini_lawam_stage2_gate/residual_gate_stage2.pt`

Both report provenance from the single 24-episode 256x256 file. The current
Stage 1 best checkpoint is epoch 6 with arm physical MAE about `0.00258` and
only about `3.2%` improvement over predicting zero residual; arm cosine
similarity is about `0.116`, while gripper accuracy is about `0.989`. This is a
warning that arm correction quality is modest despite the low absolute MAE.
The current Stage 2 best checkpoint is epoch 6 with validation F1 about `0.883`
at threshold `0.3`. Treat these as experimental metrics, not evidence of safe
hardware deployment.

Artifact paths and metrics are local snapshots, not portable source-code
guarantees. Reinspect checkpoint metadata before relying on them.

## Verification and utilities

Operator commands and training commands are maintained in `README.md`.

Focused test command:

```bash
python3 -m pytest \
  hil/test_actions.py hil/test_vr.py hil/test_policy.py hil/test_cli.py \
  hil/test_data.py hil/test_delete_episodes.py \
  hil/test_stage1.py hil/test_stage2.py
```

Useful inspection commands:

```bash
python3 scripts/view_hdf5_gui.py --input PATH_TO_DATASET.hdf5
python3 -m hil.delete_episodes PATH_TO_DATASET.hdf5 --list
```

Episode deletion writes a new `_pruned` file by default. `--in-place` performs
a staged replacement and retains the original as `.bak`. This is intentionally
recoverable because collected robot data is expensive.

Before changing hardware behavior, run the narrow tests relevant to the
change, inspect a dry-run HDF5 episode, and repeat the no-`--execute` collection
path. Full real-robot validation must remain an explicit operator decision.

## Guidance for future coding agents

- Read `README.md`, this handoff, and the touched module completely before
  editing control or action semantics.
- Preserve unrelated work in the dirty worktree. Do not assume all untracked
  HIL files are disposable.
- Prefer schema aliases and explicit migrations over silently breaking old
  datasets.
- Keep base-policy action, executed action, residual, and ownership as separate
  concepts.
- Never infer side-grip ownership from action magnitude or action difference.
- Do not run real-robot commands or add `--execute` during automated testing.
- Keep any future deployment entrypoint separate from collection and normal
  Mini-LaWAM rollout scripts.
- When behavior and documentation disagree, treat the code plus focused tests
  as current truth, then update both documents together.
