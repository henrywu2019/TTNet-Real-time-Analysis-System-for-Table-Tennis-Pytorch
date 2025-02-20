#!/bin/bash

set -x

python demo.py \
  --working-dir '../' \
  --saved_fn 'demo' \
  --arch 'ttnet' \
  --pretrained_path ../checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_3.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --thresh_ball_pos_mask 0.05 \
  --video_path /home/ubuntu/w.mp4 \
  --output_format video \
  --save_demo_output
