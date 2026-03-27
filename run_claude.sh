#!/bin/bash

# Configuration
PROJECT_DIR="/n/data1/hms/dbmi/farhat/aryan/AL/agent_active_learning"
cd "$PROJECT_DIR" || exit

SERVED_NAME="qwen35-27b"

PORT=8080


# 1. Automatically find the compute node running vLLM
echo "Searching for active vLLM compute node..."
# Using -o %N to get the nodelist, and tr to handle potential comma-separated lists
COMPUTE_NODE=$(squeue -u $USER -p gpu -t RUNNING -h -o %N | head -n 1 | tr -d ' ')

if [ -z "$COMPUTE_NODE" ]; then
    echo "❌ Error: No running GPU job found for $USER."
    echo "Please start vLLM first using 'bash start_vllm.sh' and wait for it to allocate a node."
    exit 1
fi

# Ensure port 8080 is used
# If we are ALREADY on the compute node, use 127.0.0.1 to avoid DNS/hostname issues
# Using partial match because 'hostname' often includes more than the Slurm node name
CURRENT_HOST=$(hostname -s | tr -d ' ')
CLEAN_COMPUTE_NODE=$(echo "$COMPUTE_NODE" | cut -d'.' -f1 | tr -d ' ')

if [[ "$CURRENT_HOST" == *"$CLEAN_COMPUTE_NODE"* ]] || [[ "$CLEAN_COMPUTE_NODE" == *"$CURRENT_HOST"* ]]; then
    VLLM_URL="http://127.0.0.1:${PORT}/v1"
    echo "🏠 Using local loopback for vLLM (Detected: $CURRENT_HOST matches $CLEAN_COMPUTE_NODE)"
else
    VLLM_URL="http://${COMPUTE_NODE}:${PORT}/v1"
    echo "🌐 Using remote address for vLLM ($CURRENT_HOST != $CLEAN_COMPUTE_NODE)"
fi



echo "✅ Found vLLM running on $COMPUTE_NODE"
echo "🚀 Launching Claude Code pointing to $VLLM_URL..."

# 2. Environment variables for redirecting Claude Code to vLLM
export ANTHROPIC_BASE_URL="$VLLM_URL"

export ANTHROPIC_API_KEY="sk-ant-vllm-dummy"

# Map all Claude models to our local served model
export ANTHROPIC_DEFAULT_OPUS_MODEL="$SERVED_NAME"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$SERVED_NAME"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$SERVED_NAME"

# Run Claude Code
claude

