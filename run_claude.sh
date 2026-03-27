#!/bin/bash

# Configuration
VLLM_URL="http://localhost:8000/v1"
SERVED_NAME="qwen35-27b"

echo "Launching Claude Code pointing to local vLLM at $VLLM_URL..."

# Environment variables for redirecting Claude Code to vLLM
export ANTHROPIC_BASE_URL="$VLLM_URL"
export ANTHROPIC_API_KEY="sk-ant-vllm-dummy"

# Map all Claude models to our local served model
export ANTHROPIC_DEFAULT_OPUS_MODEL="$SERVED_NAME"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$SERVED_NAME"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$SERVED_NAME"

# Run Claude Code in the current repository
claude .
