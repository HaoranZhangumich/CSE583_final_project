#!/bin/bash

set -e


python3 main.py \
  --repo-root .. \
  --out-dir ./out_ir_transformer \
  --input-mode ir \
  --model-type transformer \
  --task all \
  --d-model 128 \
  --num-layers 4 \
  --nhead 8 \
  --dim-feedforward 256

  python3 main.py \
  --repo-root .. \
  --out-dir ./out_ir_lstm \
  --input-mode ir \
  --model-type lstm \
  --task all \
  --d-model 64 \
  --num-layers 2

  python3 main.py \
  --repo-root .. \
  --out-dir ./out_source_transformer \
  --input-mode source \
  --model-type transformer \
  --task all \
  --d-model 128 \
  --num-layers 4 \
  --nhead 8 \
  --dim-feedforward 256

  python3 main.py \
  --repo-root .. \
  --out-dir ./out_source_lstm \
  --input-mode source \
  --model-type lstm \
  --task all \
  --d-model 64 \
  --num-layers 2