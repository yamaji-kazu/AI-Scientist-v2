#!/bin/bash
# 題材づくり(④ の前段 ideate)/論文執筆(⑤ writeup)/自己査読(⑥ review)を 1 段回す。
#   bash ~/run_qwen_paper.sh ideate <run>            … ~/rdc-exchange/<run>/input から題材を作る
#   bash ~/run_qwen_paper.sh <writeup|review> <dir>  … ~/ais-smoke-out の下の作業フォルダ
# 研究実行と同じ隔離の網(ais-egress)・同じ LLM(Qwen3-Coder, 8001)。GPU は使わない。
# 外向き通信は使わない(文献は DR の調査から実行器が渡す)。像は TeX と和文(CJK)を足した ais-exp:rdc-tex-cjk に、日本語版を組む luatexja と原ノ味を足した ais-exp:rdc-tex-ja。
#
# 2026-10-13: 日本語化だけは llm-jp-4.1-8B に任せる(Qwen は簡体字・ハングルを混ぜ、用語も誤った)。
# 日本語化を含む段(writeup・revise・translate)の間だけ、空いている GPU 2 に llm-jp の vLLM(8002)と
# 隔離の網からの中継(172.30.0.1:8002)を起こし、段が終わったら止める。GPU 2 は実験のコンテナも使うので
# 置きっぱなしにしない。GPU 2 が使われていれば起こさず、rdc_writeup.py が執筆と同じモデルで訳して
# そう書き残す。AIS_PAPER_ROOT で作業フォルダの置き場を差し替えられる(試験用)
PHASE="$1"; DIR="$2"
NAME=$(basename "$DIR")
ROOT="${AIS_PAPER_ROOT:-$HOME/ais-smoke-out}"
JA_MODEL=llmjp/llm-jp/llm-jp-4.1-8b-thinking
JA_GPU=${AIS_JA_GPU:-2}  # 空きを見る GPU(試験で使用中の GPU を指して、起こさない道を通すため)
cd ~/AI-Scientist-v2
if [ "$PHASE" = "ideate" ]; then
  MOUNT="-v $HOME/rdc-exchange/$NAME:/rdc -v $HOME/rdc-exchange/$NAME/input:/rdc/input:ro -v $HOME/AI-Scientist-v2/rdc_ideate.py:/ais/rdc_ideate.py:ro"
  CMD="python rdc_ideate.py /rdc"
else
  MOUNT="-v $ROOT:/ais/experiments -v $HOME/AI-Scientist-v2/rdc_writeup.py:/ais/rdc_writeup.py:ro -v $HOME/AI-Scientist-v2/ai_scientist/perform_icbinb_writeup.py:/ais/ai_scientist/perform_icbinb_writeup.py:ro -v $HOME/AI-Scientist-v2/ai_scientist/llm.py:/ais/ai_scientist/llm.py:ro"
  CMD="python rdc_writeup.py $PHASE experiments/$NAME"
fi

JA_ENV=""
VLLM_PID=""   # この段で起こした llm-jp の vLLM(起こしていなければ空)
SOCAT_PID=""  # この段で起こした中継
ja_stop() {
  if [ -n "$SOCAT_PID" ]; then
    kill -TERM "$SOCAT_PID" 2>/dev/null; SOCAT_PID=""
  fi
  [ -z "$VLLM_PID" ] && return
  kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null
  for i in $(seq 30); do
    used=$(nvidia-smi -i $JA_GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ "${used:-0}" -lt 1000 ] && break; sleep 2
  done
  if [ "${used:-0}" -lt 1000 ]; then
    echo "[ja] llm-jp を止めた(GPU $JA_GPU: ${used:-?} MiB)"
  else
    echo "[ja] llm-jp を止めたが GPU $JA_GPU が空かない(${used} MiB)"
  fi
  VLLM_PID=""
}
case "$PHASE" in writeup|revise|translate)
  JA_WAIT=300  # llm-jp の起動を待つ秒数(読み込みは約 1 分)
  if ss -ltn | grep -q "127.0.0.1:8002 "; then
    echo "[ja] llm-jp は起動済み(この段では止めない)"
  else
    used=$(nvidia-smi -i $JA_GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ "${used:-99999}" -lt 1000 ]; then
      setsid ~/serve_llmjp_ja.sh > ~/serve_llmjp_ja.log 2>&1 < /dev/null & VLLM_PID="$!"
      echo "[ja] llm-jp を GPU $JA_GPU で起動(pid $!)"
    else
      echo "[ja] GPU $JA_GPU が使用中(${used:-?} MiB)のため llm-jp を起こさない"
      JA_WAIT=0  # 起こさないので待たない。rdc_writeup.py が執筆と同じモデルで訳し、そう書き残す
    fi
  fi
  if ! ss -ltn | grep -q "172.30.0.1:8002 "; then
    setsid socat TCP-LISTEN:8002,bind=172.30.0.1,fork,reuseaddr TCP:127.0.0.1:8002 < /dev/null > /dev/null 2>&1 &
    SOCAT_PID="$!"
  fi
  JA_ENV="-e RDC_JA_MODEL=$JA_MODEL -e LLMJP_BASE_URL=http://172.30.0.1:8002/v1 -e RDC_JA_WAIT=$JA_WAIT"
  trap ja_stop EXIT
  trap "exit 143" TERM INT HUP
  ;;
esac

sudo docker rm -f ais-paper 2>/dev/null
sudo docker run --rm --name ais-paper --network ais-egress \
  -e no_proxy=172.30.0.1,localhost,127.0.0.1 -e NO_PROXY=172.30.0.1,localhost,127.0.0.1 \
  -e OPENAI_BASE_URL=http://172.30.0.1:8001/v1 -e OPENAI_API_KEY=EMPTY -e LLMJP_TIMEOUT=1800 \
  $JA_ENV $MOUNT \
  ais-exp:rdc-tex-ja \
  bash -c "cd /ais && timeout 10800 $CMD"
rc=$?
ja_stop
echo "PAPER_EXIT=$rc $(date +%H:%M:%S)"
