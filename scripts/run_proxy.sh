#!/bin/bash
# Architecture search script for DGX Spark (GB10).
#
# Uses a scaled-down proxy model (6L x 256-dim) so torch.compile fits in 101KB SRAM.
# Findings about activation functions, attention variants, normalization, etc. are
# scale-invariant and transfer to the full 9L x 512-dim competition model on H100s.
#
# Expected step throughput:
#   With compile:    ~400ms/step  ->  ~1200 steps in 600s
#   Without compile: ~1500ms/step ->  ~370 steps in 600s

# 1. Clean up any broken compiler artifacts
echo "Cleaning Inductor cache..."
rm -rf /tmp/torchinductor_ubuntu/*

# 2. Patch zeropower compile (Triton crash on GB10)
sed -i 's/zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")/# zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")/g' train_gpt.py

# 3. Hardware-specific env vars
export TRITON_MAX_REGS_PER_THREAD=255
export TORCHINDUCTOR_REDUCE_OP_FUSION=0

# 4. Proxy model parameters
#    Scale down from 9L/512-dim to 6L/256-dim so compiled kernels fit in GB10 SRAM.
#    Keep MLP_MULT=2 and same attention structure so architecture findings transfer.
export NUM_LAYERS=6
export MODEL_DIM=256
export NUM_HEADS=4
export NUM_KV_HEADS=2
export MLP_MULT=3
export TRAIN_SEQ_LEN=512   # shorter seq also reduces attention SRAM pressure

# 5. Probe compile with proxy model size
echo ""
echo "Probing torch.compile with proxy model (6L x 256-dim)..."
TORCH_COMPILE_MODEL="reduce-overhead" \
MAX_WALLCLOCK_SECONDS=30 \
WARMUP_STEPS=3 \
ITERATIONS=1 \
VAL_LOSS_EVERY=0 \
RUN_ID="proxy_compile_probe" \
NUM_LAYERS=$NUM_LAYERS \
MODEL_DIM=$MODEL_DIM \
NUM_HEADS=$NUM_HEADS \
NUM_KV_HEADS=$NUM_KV_HEADS \
TRAIN_SEQ_LEN=$TRAIN_SEQ_LEN \
    torchrun --standalone --nproc_per_node=1 train_gpt.py > /tmp/proxy_probe_out.txt 2>&1
PROBE_EXIT=$?
tail -5 /tmp/proxy_probe_out.txt

if [ $PROBE_EXIT -eq 0 ]; then
    echo "compile probe: SUCCESS — ~1200 steps in 600s"
    COMPILE_MODE="reduce-overhead"
    WARMDOWN_ITERS=120
    STEP_NOTE="~1200 steps (compiled, proxy model)"
else
    echo "compile probe: FAILED — ~370 steps in 600s (no compile)"
    echo "  (check /tmp/proxy_probe_out.txt for details)"
    COMPILE_MODE=""
    WARMDOWN_ITERS=37
    STEP_NOTE="~370 steps (no compile, proxy model)"
fi
echo ""

# 6. Run parameters
export RUN_ID="proxy_$(date +%M%S)"
export MAX_WALLCLOCK_SECONDS=600
export WARMDOWN_ITERS
export WARMUP_STEPS=3
export VAL_LOSS_EVERY=20    # ~55 checkpoints with compile, ~17 without
export TORCH_COMPILE_MODEL="$COMPILE_MODE"
DATA_PATH="./data/datasets/fineweb10B_sp1024/"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE=1024

mkdir -p logs

echo "Launching proxy run: $RUN_ID ($STEP_NOTE, WARMDOWN_ITERS=$WARMDOWN_ITERS)"
echo "Log written to: logs/${RUN_ID}.txt"
echo "To compare variants: change MLP_MULT, activation fn, attention type in train_gpt.py"
echo "  then rerun this script and compare final val_bpb"
echo ""

torchrun --standalone --nproc_per_node=1 train_gpt.py \
    --data_path=$DATA_PATH \
    --tokenizer_path=$TOKENIZER_PATH \
    --vocab_size=$VOCAB_SIZE

echo ""
echo "Run complete. Log: logs/${RUN_ID}.txt"
echo "Final score: $(grep 'final_int8_zlib_roundtrip val_bpb' logs/${RUN_ID}.txt | tail -1)"
