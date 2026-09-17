#!/bin/bash
# Push lanes one at a time as Kaggle's 2-concurrent-GPU-session slots free up.
K=/home/huhu/.venvs/kaggle/bin/kaggle; HERE=$(cd "$(dirname "$0")" && pwd); U=x1ngyu
QUEUE=("$@"); LOG=/tmp/kaggle-queue.log
running() { n=0; for l in lane0 lane1 lane2 lane3; do s=$($K kernels status $U/areno-kaggle-$l 2>/dev/null | grep -oE 'RUNNING|QUEUED'); [ -n "$s" ] && n=$((n+1)); done; echo $n; }
while [ ${#QUEUE[@]} -gt 0 ]; do
  if [ "$(running)" -lt 2 ]; then
    l=${QUEUE[0]}; out=$($K kernels push -p "$HERE/kernels/$l" --accelerator NvidiaTeslaT4 2>&1 | tail -1)
    echo "$(date +%H:%M) push $l: $out" | tee -a $LOG
    echo "$out" | grep -q "successfully" && QUEUE=("${QUEUE[@]:1}")
  fi
  sleep 300
done
echo "$(date +%H:%M) queue drained" | tee -a $LOG
