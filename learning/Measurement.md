# Blackwell GB10: Parameter Golf Experimentation & Profiling Guide

## 1. The Strategy: The 600-Second Sprint
In this competition, the script is governed by a `max_wallclock_seconds: 600` hard limit. Because each training step on the DGX Spark takes ~6.2 seconds, you only have room for approximately 97 steps.

**The "Silence Phase" Paradox:**
Once the 600-second timer hits, the training stops, and the "Silence Phase" begins. This is a massive validation loop (62M tokens). On the Spark's LPDDR5X memory, this pass takes ~10-14 minutes. If you are using **Test-Time Training (TTT)**, the model is actually performing mini-training updates *during* this validation, which is why the GPU stays pinned at 96% utilization despite no logs appearing.

---

## 2. Code Instrumentation (NVTX)
To make the Nsight Systems profile readable, you must instrument the `train_gpt.py` validation loop. This allows you to distinguish between standard inference and TTT updates in the timeline.

```python
from torch.cuda import nvtx

# Wrap your validation loop inside train_gpt.py
for i, (x, y) in enumerate(val_loader):
    with nvtx.range("Val_Step"):
        # Profile Forward Pass
        with nvtx.range("Forward"):
            logits = model(x)
            loss = criterion(logits, y)

        # Profile TTT Backprop/Optimizer
        with nvtx.range("TTT_Update"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

## 3. The Experimentation Workflow

### Step A: Launch the Sprint
Run the script with aggressive settings. We reduce the warmup to 5 steps to ensure more time is spent at the maximum learning rate.

```bash
export RUN_ID="spark_v8_aggro"

torchrun --standalone --nproc_per_node=1 train_gpt.py \
    --run_id=$RUN_ID \
    --warmup_steps=5 \
    --learning_rate=0.08 \
    --max_wallclock_seconds=600
```

### Step B: Identify the PID during the Silence
When the 600-second mark is reached and the terminal output stops, the script is in the Validation loop. Find the process ID:

```bash
pgrep -f train_gpt.py
```

### Step C: Capture the Nsight Profile
While the GPU is still at 96% utilization, capture a 30-second slice of the math to see exactly what is happening under the hood.

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --duration=30 \
  --output=profiles/val_capture_$(date +%Y%m%d_%H%M) \
  -p <YOUR_PID_HERE>
```

## 4. Measurement & Analysis Skills
Once you load the .nsys-rep file into the Nsight Systems GUI, check these specific markers:

- **NVTX Rows:** Look for the "Forward" and "TTT_Update" labels. If TTT_Update is significantly wider than Forward, your bottleneck is the backpropagation math.
- **Kernel Occupancy:** Right-click a kernel. If occupancy is low (<50%), the Blackwell Streaming Multiprocessors (SMs) are underutilized. Solution: Increase micro_batch_size.
- **Memory Gaps:** If there are white gaps between kernels, the GPU is waiting on the Grace CPU to feed it data from the LPDDR5X RAM.
- **SM Efficiency:** If you see "Compute Bound," you are successfully utilizing the Blackwell Tensor Cores.

## 5. Maintenance Commands

- **Force Stop:** `pkill -9 python` — Use this if the validation takes too long and you want to start a new experiment.
- **Live Monitor:** `watch -n 1 nvidia-smi`
- **Log Check:** `tail -f logs/spark_v8_aggro.txt`


**Would you like me to help you set up a second "Fast-Mode" script that caps the validation tokens for your daytime testing, so you only do the full "Silence Phase" for your overnight runs?**
