#!/usr/bin/bash

nsys profile \
  --trace=cuda,nvtx,osrt \
  --duration=30 \
  --output=profiles/val_$(date +%Y%m%d_%H%M) \
  -p $(pgrep -f train_gpt.py)
