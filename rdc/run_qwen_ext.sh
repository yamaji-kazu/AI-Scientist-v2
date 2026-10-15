#!/bin/bash
# 研究実行(④)。AI-Scientist の実験(BFTS)を、隔離の網(ais-egress)と GPU 2 で回す。
#   AIS_RUN_ID=<run> [AIS_IDEA=<idea.json>] bash ~/run_qwen_ext.sh
# AIS_RUN_ID があるとき: 交換の場所 ~/rdc-exchange/<run> を /rdc に、そのうち入力(input/: DG が渡した
#   研究計画・maDMP・取得済みデータ)を読み取り専用で /rdc/input に渡す。
# AIS_IDEA があるとき: 研究計画から作った題材(rdc_ideate.py の出力)を回す。無いときは従来の固定の
#   題材(ext_data_demo.json)。計画の無い run だけがこちらになる(2026-10-11)。
cd ~/AI-Scientist-v2
M="gpt-4o"
RDCMOUNT=""
if [ -n "$AIS_RUN_ID" ]; then
  X=$HOME/rdc-exchange/$AIS_RUN_ID
  mkdir -p $X/requests $X/staged $X/input
  RDCMOUNT="-e RDC_EXCHANGE=/rdc -e PYTHONPATH=/opt -v $X:/rdc -v $X/input:/rdc/input:ro -v $HOME/rdc_data.py:/opt/rdc_data.py:ro"
fi
IDEA_HOST=~/AI-Scientist-v2/ai_scientist/ideas/ext_data_demo.json
IDEA=ai_scientist/ideas/ext_data_demo.json
if [ -n "$AIS_IDEA" ]; then
  # 実行器は ~/rdc-exchange/... の形で渡す。docker は絶対パスでないとボリューム名と読むので展開する(2026-10-11)
  IDEA_HOST="${AIS_IDEA/#\~/$HOME}"
  [ -f "$IDEA_HOST" ] || { echo "題材のファイルがありません: $IDEA_HOST"; echo "QWEN_EASY_EXIT=2 $(date +%H:%M:%S)"; exit 2; }
  IDEA=ai_scientist/ideas/rdc_plan_idea.json
fi
sudo docker rm -f ais-qwen-ext 2>/dev/null
sudo docker run --rm --name ais-qwen-ext --gpus '"device=2"' --network ais-egress \
  $RDCMOUNT \
  -e HTTP_PROXY=http://172.30.0.1:3128 -e HTTPS_PROXY=http://172.30.0.1:3128 \
  -e no_proxy=172.30.0.1,localhost,127.0.0.1 -e NO_PROXY=172.30.0.1,localhost,127.0.0.1 \
  -e OPENAI_BASE_URL=http://172.30.0.1:8001/v1 -e OPENAI_API_KEY=EMPTY -e LLMJP_TIMEOUT=1800 \
  -v $IDEA_HOST:/ais/$IDEA:ro -v ~/bfts_config.qwen_easy.yaml:/ais/bfts_config.yaml -v ~/ais-smoke-out:/ais/experiments \
  -v $HOME/AI-Scientist-v2/ai_scientist/perform_plotting.py:/ais/ai_scientist/perform_plotting.py:ro -v $HOME/AI-Scientist-v2/ai_scientist/llm.py:/ais/ai_scientist/llm.py:ro \
  ais-exp:rdc \
  bash -c "cd /ais && timeout 28800 python launch_scientist_bfts.py --load_ideas $IDEA --idea_idx 0 --skip_writeup --skip_review --model_agg_plots $M"
echo "QWEN_EASY_EXIT=$? $(date +%H:%M:%S)"
