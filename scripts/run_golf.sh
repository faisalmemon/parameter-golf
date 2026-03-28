#!/bin/bash

# 1. Clean up any broken compiler artifacts
echo "Cleaning Inductor cache..."
rm -rf /tmp/torchinductor_ubuntu/*

# 2. Patch the code for DGX Spark (Blackwell GB10) compatibility
# Reference 736: Disable functional compile on the optimizer
sed -i 's/zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")/# zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5, mode="reduce-overhead")/g' train_gpt.py

# Reference 843: Disable structural model compile to stay under 101KB SRAM
sed -i 's/compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)/compiled_model = base_model/g' train_gpt.py

# 3. Set Hardware-Specific Environment Variables
export TRITON_MAX_REGS_PER_THREAD=255
export TORCHINDUCTOR_REDUCE_OP_FUSION=0

# 4. Define Run Parameters (exported so Hyperparameters picks them up via os.environ)
export RUN_ID="spark_smoke_$(date +%M%S)"
export MAX_WALLCLOCK_SECONDS=60
export WARMUP_STEPS=20
export VAL_LOSS_EVERY=0
DATA_PATH="./data/datasets/fineweb10B_sp1024/"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE=1024

mkdir -p /workspace/profiles

echo "Launching Run: $RUN_ID"
echo "Profile output: /workspace/profiles/roundtrip_${RUN_ID}.nsys-rep"

# 5. Execute under nsys.
# --capture-range=cudaProfilerApi means nsys records nothing until cudaProfilerStart()
# is called inside eval_val (first 5 batches of the roundtrip eval only).
nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output=/workspace/profiles/roundtrip_${RUN_ID} \
    torchrun --standalone --nproc_per_node=1 train_gpt.py \
        --run_id=$RUN_ID \
        --data_path=$DATA_PATH \
        --tokenizer_path=$TOKENIZER_PATH \
        --vocab_size=$VOCAB_SIZE

echo ""
echo "Profile written to: /workspace/profiles/roundtrip_${RUN_ID}.nsys-rep"
echo "Log written to: logs/${RUN_ID}.txt"
