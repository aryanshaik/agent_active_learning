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

# Ensure port 8000 is used on that specific node
VLLM_URL="http://${COMPUTE_NODE}:${PORT}/v1"

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

