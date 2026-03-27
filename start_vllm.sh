#!/bin/bash

# Configuration
MODEL_PATH="/n/data1/hms/dbmi/farhat/aryan/AL/models/qwen35-27b"
SERVED_NAME="qwen35-27b"
PORT=8000
CONDA_ENV="/home/ars3983/miniforge/envs/al-agent"

echo "Starting vLLM server for Claude Code..."
echo "Model: $MODEL_PATH"
echo "Serving as: $SERVED_NAME"
echo "Port: $PORT"

# Ensure environment is active
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Optimized command for qwen35-27b on quad GPU nodes (~44GB VRAM)
# Using quantization to ensure fit and memory utilization headroom
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port "$PORT" \
    --trust-remote-code \
    --quantization bitsandbytes \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager
