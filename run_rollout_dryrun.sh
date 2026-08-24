#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python -m mini_lawam.rollout_ur7e_vr \
  --ckpt results/mini_lawam/ckpt_combined_256_attn_ry_rz_binary_grip_t1.pt \
  --table-cam-serial 244422300964 \
  --wrist-cam-serial 252122300792 \
  --table-exposure 180 --table-gain 16 \
  --wrist-exposure 100 --wrist-gain 16 \
  --robot-ip 140.96.93.7 \
  --use-gripper-control \
  --train-frame-hw 240 320 \
  --temporal-ensemble --te-m 1.0 \
  --gripper-open-lead-steps 0 \
  --target-ema 1.0 --target-deadband 0.0 \
  --max-reach 0.015 \
  --ws-min -0.072 -0.15 0.158 \
  --ws-max 0.81 0.427 0.518 \
  --servol-max-pos-step 0.002 \
  --servol-max-rot-step 0.005 \
  --show-camera --show-subgoal \
  --subgoal-update-steps 8 \
  --action-scale 1.0 \
  --enable-rz \
  --enable-ry \
  --min-rot-angle 2.2 \
  --max-rot-angle 3.5 \
  --trace-dir results/mini_lawam/traces
