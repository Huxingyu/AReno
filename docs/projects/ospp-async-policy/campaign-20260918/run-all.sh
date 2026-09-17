#!/bin/bash
# 用法：bash run-all.sh   （lane0 先串行跑完，再并行 lane1-3；日志在本目录 laneN.log）
cd /mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/campaign-20260918
bash lane0.sh > lane0.log 2>&1 || { echo "lane0 失败，先看 lane0.log 再决定是否继续"; exit 1; }
for n in 1 2 3; do nohup bash lane$n.sh > lane$n.log 2>&1 & done
wait
