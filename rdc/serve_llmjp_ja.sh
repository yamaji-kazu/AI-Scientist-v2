#!/bin/bash
# 2026-10-13: 日本語化だけを llm-jp に任せる比較のための vLLM。GPU 2(空いている 1 枚)・8002 番。
# Qwen(GPU 0,1・8001)とは別に動かす。隔離の網からは socat(172.30.0.1:8002)で届ける。
cd ~/llm-jp-4-cookbook/llmjp4_vllm || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2
exec .venv/bin/python example_cli.py serve "$HOME/models/llm-jp-4.1-8b-thinking" \
  --served-model-name llm-jp/llm-jp-4.1-8b-thinking \
  --reasoning-parser llmjp4 --trust-remote-code \
  --gpu-memory-utilization 0.90 --host 127.0.0.1 --port 8002 --dtype bfloat16 --max-model-len 32768 --max-num-seqs 8
