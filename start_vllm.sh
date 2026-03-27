#!/bin/bash

# Configuration
MODEL_PATH="/n/data1/hms/dbmi/farhat/aryan/AL/models/qwen35-27b"
SERVED_NAME="qwen35-27b"
PORT=8080

CONDA_ENV="/home/ars3983/miniforge/envs/al-agent"

echo "Starting vLLM server for Claude Code..."
echo "Model: $MODEL_PATH"
echo "Serving as: $SERVED_NAME"
echo "Port: $PORT"

# Ensure environment is active
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Fix for CXXABI_1.3.15 version mismatch on O2
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"

# Get the internal node IP for binding (ignoring external IPs)
NODE_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n 1)

# Request a GPU node via srun and start vLLM
# Binding specifically to $NODE_IP ensures it's reachable on the private O2 network
srun -p gpu --gres=gpu:1 --mem=60G --time=08:00:00 --pty python -m vllm.entrypoints.openai.api_server \
    --host "$NODE_IP" \
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



