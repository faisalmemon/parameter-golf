#!/bin/bash
# run_spark.sh

docker run --gpus all -it --rm \
  --user $(id -u):$(id -g) \
  -v /home/faisalm/dev:/workspace \
  -w /workspace/parameter-golf \
  --ipc=host \
  -e HOME=/tmp \
  -p 6006:6006 \
  nvcr.io/nvidia/pytorch:24.03-py3 \
  /bin/bash -c "pip install --user kernels tiktoken datasets sentencepiece huggingface-hub && /bin/bash"
