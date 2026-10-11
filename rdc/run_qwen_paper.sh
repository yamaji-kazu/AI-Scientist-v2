#!/bin/bash
# 題材づくり(④ の前段 ideate)/論文執筆(⑤ writeup)/自己査読(⑥ review)を 1 段回す。
#   bash ~/run_qwen_paper.sh ideate <run>            … ~/rdc-exchange/<run>/input から題材を作る
#   bash ~/run_qwen_paper.sh <writeup|review> <dir>  … ~/ais-smoke-out の下の作業フォルダ
# 研究実行と同じ隔離の網(ais-egress)・同じ LLM(Qwen3-Coder, 8001)。GPU は使わない。
# 外向き通信は使わない(文献は DR の調査から実行器が渡す)。像は TeX と和文(CJK)を足した ais-exp:rdc-tex-cjk。
PHASE="$1"; DIR="$2"
NAME=$(basename "$DIR")
cd ~/AI-Scientist-v2
if [ "$PHASE" = "ideate" ]; then
  MOUNT="-v $HOME/rdc-exchange/$NAME:/rdc -v $HOME/rdc-exchange/$NAME/input:/rdc/input:ro -v $HOME/AI-Scientist-v2/rdc_ideate.py:/ais/rdc_ideate.py:ro"
  CMD="python rdc_ideate.py /rdc"
else
  MOUNT="-v $HOME/ais-smoke-out:/ais/experiments -v $HOME/AI-Scientist-v2/rdc_writeup.py:/ais/rdc_writeup.py:ro -v $HOME/AI-Scientist-v2/ai_scientist/perform_icbinb_writeup.py:/ais/ai_scientist/perform_icbinb_writeup.py:ro"
  CMD="python rdc_writeup.py $PHASE experiments/$NAME"
fi
sudo docker rm -f ais-paper 2>/dev/null
sudo docker run --rm --name ais-paper --network ais-egress \
  -e no_proxy=172.30.0.1,localhost,127.0.0.1 -e NO_PROXY=172.30.0.1,localhost,127.0.0.1 \
  -e OPENAI_BASE_URL=http://172.30.0.1:8001/v1 -e OPENAI_API_KEY=EMPTY -e LLMJP_TIMEOUT=1800 \
  $MOUNT \
  ais-exp:rdc-tex-cjk \
  bash -c "cd /ais && timeout 10800 $CMD"
echo "PAPER_EXIT=$? $(date +%H:%M:%S)"
