# G1 evidence: async policy CPU closed loop

日期：2026-09-14。分支：`feat/async-policy-trainer`（Fork `Huxingyu/AReno`）。
功能基线：上游 `60e2fa73f331ea5f5f7245f0793084fbebe9d76c`。
提交：`1f73e23`（协议基础 F1）、`ca97cad`（CPU pipeline / bridge / metrics）。

本文件是 T02–T06、T07（协议部分）、T08a 的证据索引，对应工作在第 5 节的
`areno/experimental/async_policy/` 与 `tests/test_async_policy_*_cpu.py`。

## 验证命令

```bash
source /mnt/areno-lab/env.sh
cd /mnt/areno-lab/feat-async-trainer
python -m pytest -q \
  tests/test_async_policy_pipeline_cpu.py \
  tests/test_async_policy_batch_cpu.py \
  tests/test_async_policy_coordinator_cpu.py \
  tests/test_async_policy_equivalence_cpu.py \
  tests/test_async_policy_bridge_cpu.py \
  -p no:cacheprovider
# 结果：43 passed（连续 8 次运行全部通过）
```

## 真实重叠证据

证据脚本用事件门控让 producer 线程与主线程同时停在 backend 调用内部，再从事件
时间线取区间相交，而不是用 sleep 推断：

```
both gated calls active at once: rollout#1 True train#0 True
overlap(rollout,train) : True
rollout intervals: [(51428.475, 51428.507), (51431.585, 51431.636)]
train intervals  : [(51431.585, 51431.636), (51431.636, 51431.636)]
producer thread still alive: False
new threads left behind    : []
```

## 覆盖的场景

| 场景 | 测试 |
|---|---|
| 数据耗尽：drain 剩余批次后 EOF 退出 | `test_data_exhaustion_drains_and_exits` |
| `max_steps` 提前停止 | `test_max_steps_stops_early` |
| reward 抛错经控制通道传播并清理 | `test_reward_error_propagates` |
| rollout 引擎失败 | `test_rollout_failure_propagates` |
| 权重同步失败不推进 Vr | `test_sync_failure_does_not_advance_rollout_version` |
| 用户停止 `stopped`、无残留线程 | `test_request_stop_is_clean` |
| 重复 close 幂等、close 后 run 报错 | `test_double_close_is_idempotent_and_run_after_close_fails` |
| stale 整批丢弃后同步恢复进展 | `test_stale_batch_is_dropped_and_sync_recovers_progress` |
| K=2 两个生产任务同时在途（inflight_peak=2） | `test_k_greater_than_one_runs_multiple_production_tasks` |
| 未 stepped 不推进版本、不触发同步 | `test_non_stepped_update_does_not_advance_version_or_sync` |
| 批次指标含 Vt/Vr/Vb、lag、决策、阶段时间 | `test_batch_metrics_capture_versions_lag_and_decision` |
| T02 与同步 `_materialize_train_batch` 逐字段一致 | `test_async_envelope_matches_sync_materialize` |
| T04 pending-sync 关闭准入、4 线程压力下仍达安全点 | `test_pending_sync_closes_admission_before_draining`、`test_sync_reaches_safe_point_under_continuous_load` |

## 重复运行发现的缺陷（已修）

`request_stop` 关闭队列后，train loop 把 `QueueClosed` 一律判为"数据耗尽"，
退出原因偶发 `data_exhausted`（15 次中 5 次）。修复为按 `_stop` 区分
`stopped` / `data_exhausted`（`areno/experimental/async_policy/pipeline.py`），
修复后 pipeline 用例 20/20、全量 8/8 稳定。

## 边界（未验证，不计入）

- T06 只验证了 fake dual-backend 的准入协议；真实 CUDA backend、原生 NCCL 权重
  传输未接入，A07（真实多 GPU 并发）未取得。
- T07 只验证了 stepped / 版本推进的控制语义；真实 optimizer 更新计数与两端
  publish / receive 未接入。
- T08b 完整指标未实现。
- G1 的评审点（把证据发导师确认）尚未完成，故 G1 记为"待评审"。
- 同步回归（T13 / A09）未在本轮运行。
