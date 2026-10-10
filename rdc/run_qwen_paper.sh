#!/bin/bash
# 論文執筆(⑤ writeup)/自己査読(⑥ review)を、研究実行(④)の作業フォルダに対して 1 段回す。
#   bash ~/run_qwen_paper.sh <writeup|review> <~/ais-smoke-out の下の作業フォルダ>
# 研究実行と同じ隔離の網(ais-egress)・同じ LLM(Qwen3-Coder, 8001)。GPU は使わない。
# 外向き通信は使わない(文献の検索はしない)。像は TeX を足した ais-exp:rdc-tex。
PHASE="$1"; DIR="$2"
NAME=$(basename "$DIR")
cd ~/AI-Scientist-v2
sudo docker rm -f ais-paper 2>/dev/null
sudo docker run --rm --name ais-paper --network ais-egress \
  -e no_proxy=172.30.0.1,localhost,127.0.0.1 -e NO_PROXY=172.30.0.1,localhost,127.0.0.1 \
  -e OPENAI_BASE_URL=http://172.30.0.1:8001/v1 -e OPENAI_API_KEY=EMPTY -e LLMJP_TIMEOUT=1800 \
  -v ~/ais-smoke-out:/ais/experiments -v ~/AI-Scientist-v2/rdc_writeup.py:/ais/rdc_writeup.py:ro -v ~/AI-Scientist-v2/ai_scientist/perform_icbinb_writeup.py:/ais/ai_scientist/perform_icbinb_writeup.py:ro \
  ais-exp:rdc-tex \
  bash -c "cd /ais && timeout 10800 python rdc_writeup.py $PHASE experiments/$NAME"
echo "PAPER_EXIT=$? $(date +%H:%M:%S)"
