#!/bin/bash

python train_sota.py \
  --train_tars "./1k-coins-dataset-no-pr/train-dataset-{0000..0029}.tar" \
  --test_tars "./1k-coins-dataset-no-pr/test-dataset-{0000..0003}.tar" \
  --meta_csv "./1k-coins-dataset-no-pr.csv" \
  --model "convnext_base" \
  --batch_size 4 \
  --epochs 30 \
  --lr 1e-3 \
  --alpha 0 \
  --out_dir "convnext_base"