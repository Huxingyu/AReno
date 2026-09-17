#!/bin/bash
# T3 quality：3 seed × 64/256 token × sync/lag1/lag0/offpolicy = 24 job
cd /mnt/areno-lab/async-policy-batched
/mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/venv/bin/python examples/async_policy/tools/matrix.py --suite quality --output-dir /mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/campaign-20260918/quality --budget-file /mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/campaign-20260918/budget-lane1.json --model-path Qwen/Qwen3-0.6B --steps 55 --warmup 5 --backend modal --modal-python /mnt/areno-lab/async-policy-rewrite/runs/async-policy-rewrite/modal-venv/bin/python --budget-usd 60 --job-timeout-s 3600 --campaign-timeout-s 43200 --max-attempts 3 --execute --seeds 41 42 43
