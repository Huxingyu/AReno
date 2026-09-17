#!/bin/bash
# T6 严格全参数续训 seed43 + T7 最终版本 GPU 回归。先跑，20 分钟内能暴露代码问题。
set -e
cd /mnt/areno-lab/async-policy-batched
MP=/mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/modal-venv/bin/python
$MP examples/async_policy/tools/modal_run.py --phase resume-full --output-dir /mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/campaign-20260918/resume-full-seed43 --budget-file /mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/campaign-20260918/budget-lane0.json --budget-usd 10 --task-timeout-s 3600 --controller-timeout-s 4200 --benchmark-args --seed 43 --deterministic --audit-each-step
for phase in regression faults extended-faults; do
  $MP examples/async_policy/tools/modal_run.py --phase $phase --output-dir /mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/campaign-20260918/$phase --budget-file /mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/campaign-20260918/budget-lane0.json --task-timeout-s 1800 --controller-timeout-s 2400
done
