#!/bin/bash

python train_simple.py \
  --train_tars "./1k-coins-dataset-no-pr/train-dataset-{0000..0029}.tar" \
  --test_tars "./1k-coins-dataset-no-pr/test-dataset-{0000..0003}.tar" \
  --meta_csv "./1k-coins-dataset-no-pr.csv" \
  --model "convnext_tiny" \
  --batch_size 4 \
  --epochs 30 \
  --lr 1e-3 \
  --out_dir "convnext_tiny"