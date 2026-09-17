#!/bin/bash
# kg.sh push <role> | status <role> | log <role> | pull <role> [dir] | quota
K=/home/huhu/.venvs/kaggle/bin/kaggle; HERE=$(cd "$(dirname "$0")" && pwd); U=x1ngyu
case "$1" in
  push)   python3 "$HERE/gen.py" >/dev/null && $K kernels push -p "$HERE/kernels/$2" --accelerator NvidiaTeslaT4 ;;
  status) $K kernels status "$U/areno-kaggle-$2" ;;
  log)    $K kernels logs "$U/areno-kaggle-$2" 2>/dev/null || $K kernels output "$U/areno-kaggle-$2" -p /tmp/kg-log-$2 -q && cat /tmp/kg-log-$2/$2.log ;;
  pull)   $K kernels output "$U/areno-kaggle-$2" -p "${3:-/mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/kaggle-20260918/$2}" ;;
  quota)  $K quota ;;
  *) sed -n 2p "$0" ;;
esac
