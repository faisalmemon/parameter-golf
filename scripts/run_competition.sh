#!/bin/bash

# 1. Clean up any broken compiler artifacts
echo "Cleaning Inductor cache..."
rm -rf /tmp/torchinductor_ubuntu/*

# 2. Patch the code for DGX Spark (Blackwell GB10) compatibility
# Disable zeropower compile (Triton crash on GB10)
sed -i 's/zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")/# zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")/g' train_gpt.py

# 3. Set Hardware-Specific Environment Variables
export TRITON_MAX_REGS_PER_THREAD=255
export TORCHINDUCTOR_REDUCE_OP_FUSION=0

# 4. Probe whether partial torch.compile works on this GB10 unit.
#    mode="reduce-overhead" avoids fullgraph=True and the 101KB SRAM limit.
#    Run 3 warmup steps only; if it exits cleanly, use compile for the real run.
echo ""
echo "Probing torch.compile (reduce-overhead, no fullgraph)..."
TORCH_COMPILE_MODEL="reduce-overhead" \
MAX_WALLCLOCK_SECONDS=30 \
WARMUP_STEPS=3 \
ITERATIONS=1 \
VAL_LOSS_EVERY=0 \
RUN_ID="spark_compile_probe" \
    torchrun --standalone --nproc_per_node=1 train_gpt.py > /tmp/probe_out.txt 2>&1
PROBE_EXIT=$?
tail -5 /tmp/probe_out.txt

if [ $PROBE_EXIT -eq 0 ]; then
    echo "compile probe: SUCCESS — using torch.compile for competition run"
    COMPILE_MODE="reduce-overhead"
else
    echo "compile probe: FAILED — running without compile (~97 steps in 600s)"
    COMPILE_MODE=""
fi
echo ""

# 5. Define Run Parameters
# No-compile timing:  ~6200ms/step → ~97 steps in 600s → WARMDOWN_ITERS=15
# With-compile timing: ~800ms/step (est) → ~750 steps in 600s → WARMDOWN_ITERS=100
if [ -n "$COMPILE_MODE" ]; then
    WARMDOWN_ITERS=100
    STEP_NOTE="~750 steps (compiled)"
else
    WARMDOWN_ITERS=15
    STEP_NOTE="~97 steps (no compile)"
fi

export RUN_ID="spark_competition_$(date +%M%S)"
export MAX_WALLCLOCK_SECONDS=600
export WARMDOWN_ITERS
export WARMUP_STEPS=5
export VAL_LOSS_EVERY=10
export TORCH_COMPILE_MODEL="$COMPILE_MODE"
DATA_PATH="./data/datasets/fineweb10B_sp1024/"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE=1024

mkdir -p logs

echo "Launching competition run: $RUN_ID ($STEP_NOTE, WARMDOWN_ITERS=$WARMDOWN_ITERS)"
echo "Log written to: logs/${RUN_ID}.txt"

torchrun --standalone --nproc_per_node=1 train_gpt.py \
    --data_path=$DATA_PATH \
    --tokenizer_path=$TOKENIZER_PATH \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "Run complete. Log: logs/${RUN_ID}.txt"
