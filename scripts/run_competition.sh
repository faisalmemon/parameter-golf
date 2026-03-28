#!/bin/bash

# 1. Clean up any broken compiler artifacts
echo "Cleaning Inductor cache..."
rm -rf /tmp/torchinductor_ubuntu/*

# 2. Patch the code for DGX Spark (Blackwell GB10) compatibility
# Reference 736: Disable functional compile on the optimizer (Triton crash on GB10)
sed -i 's/zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")/# zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")/g' train_gpt.py

# 3. Set Hardware-Specific Environment Variables
export TRITON_MAX_REGS_PER_THREAD=255
export TORCHINDUCTOR_REDUCE_OP_FUSION=0

# 4. Define Run Parameters
# Step timing on DGX Spark (no compile): ~6200ms/step
# 600s budget => ~97 usable steps
# WARMDOWN_ITERS=15: warmdown_ms = 15 * 6200ms = 93s, starts at ~507s elapsed (last ~15% of budget)
# WARMUP_STEPS=5: minimise time outside the training timer (~31s JIT warmup)
# VAL_LOSS_EVERY=10: get ~9 validation checkpoints across the run
export RUN_ID="spark_competition_$(date +%M%S)"
export MAX_WALLCLOCK_SECONDS=600
export WARMDOWN_ITERS=15
export WARMUP_STEPS=5
export VAL_LOSS_EVERY=10
DATA_PATH="./data/datasets/fineweb10B_sp1024/"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE=1024

mkdir -p logs

echo "Launching competition run: $RUN_ID"
echo "Log written to: logs/${RUN_ID}.txt"

torchrun --standalone --nproc_per_node=1 train_gpt.py \
    --data_path=$DATA_PATH \
    --tokenizer_path=$TOKENIZER_PATH \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "Run complete. Log: logs/${RUN_ID}.txt"
