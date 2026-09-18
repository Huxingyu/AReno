# Kaggle 双 T4 阶段验收报告（2026-09-18）

## 1. 结论

本轮验证支持以下阶段性判断：

- 双 T4、LoRA、单机双 worker 的异步训练主路径已经可用。
- 推荐的 `lag=1`、单 prompt group 路径在全部吞吐配对中快于同步路径，
  且本轮短程评测没有观察到质量退化。
- 容量、GSPO、异常关闭和 checkpoint 保留均通过预定矩阵。
- 多 prompt group 批处理仍属于实验能力：生成吞吐提高，但陈旧样本大量丢弃，
  尚未稳定转化成端到端训练收益。
- strict full-parameter resume 未在 T4 上完成；该路径需要更大显存或明确的
  optimizer offload 设计。

因此，当前完成度可表述为：核心异步 LoRA 路径约 85%--90%，性能调优约
75%--80%，包含全参数恢复和更广硬件覆盖的生产级能力约 70%。这些百分比是
工程成熟度判断，不是测试覆盖率。

## 2. 固定版本与执行环境

- 实验代码：`feat/async-policy-batched` @
  `bb8a0637bbf13c46454fff2ed4620e330e3d95f1`
- deterministic 对照：`feat/async-policy-deterministic` @
  `17a9cb19bfaf5cc5b019afc9749f168421dcd3cc`
- 模型：`Qwen/Qwen3-0.6B`，T4 上使用 FP32
- 硬件：Kaggle `Tesla T4 x2`，每卡 15 GiB
- attention backend：`native`
- 训练长度：55 updates，前 5 updates 作为 warmup
- seeds：41、42、43

结果来自以下私有 Kaggle kernels：

- `x1ngyu/areno-kaggle-lane1`
- `x1ngyu/areno-kaggle-lane2`
- `x1ngyu/areno-kaggle-lane3`
- `x1ngyu/areno-kaggle-capacity-rest`
- `x1ngyu/areno-kaggle-throughput-rest`
- `x1ngyu/areno-kaggle-faults-rerun`

补跑 kernel 恢复原 campaign 后只执行缺项。每个补跑任务写顶层 manifest，
若 job 数、attempt 状态或子任务返回码不满足预期，kernel 会以非零状态退出。

## 3. 完整度验收

| suite | 完成 | 失败 attempt | 状态 |
|---|---:|---:|---|
| quality | 24 / 24 | 0 | complete |
| GSPO | 12 / 12 | 0 | complete |
| capacity | 57 / 57 | 0 | complete |
| throughput | 24 / 24 | 0 | complete |
| batching | 15 / 15 | 0 | complete |

合计 132 个训练/评测 job 全部成功。补跑 manifest 的关键内容为：

```json
{
  "capacity": {"expected_jobs": 57, "completed_jobs": 57,
               "failed_attempts": [], "return_code": 0, "complete": true},
  "throughput": {"expected_jobs": 24, "completed_jobs": 24,
                 "failed_attempts": [], "return_code": 0, "complete": true},
  "faults": {"hardware_target": "T4 x2", "attention_backend": "native",
             "outcomes": [{"mode": "faults", "return_code": 0, "ok": true},
                          {"mode": "extended-faults", "return_code": 0,
                           "ok": true}],
             "complete": true}
}
```

## 4. 性能与质量

### 4.1 推荐异步路径

吞吐矩阵包含 `tokens={256,512}`、`n_samples={4,8}`、三个 seed 的 12 组
sync/lag1 配对：

- 12 / 12 组中，lag1 的训练阶段都快于 sync。
- 配对训练时长比 `sync / lag1`：均值 1.368，中位数 1.373，范围
  1.265--1.448。
- 平均训练 token 吞吐：sync 54.7 tokens/s，lag1 76.7 tokens/s，约提高
  40%。
- 12 组 256-token 评测中，lag1 相对 sync 的准确率差值均不为负；平均绝对
  差值为 +0.0013。
- 所有运行的 `sync_model_overlap_violations` 和 `monitor_errors` 均为 0。

quality 矩阵中的 lag1 结果也一致：6 / 6 训练配对更快，平均时长比 1.428；
12 个评测配对的平均准确率差值为 +0.015，10 / 12 持平或更好。三个 seed、
55 steps 只能支持短程“不退化”结论，不能声称长期质量提升或收敛优势。

GSPO 的 6 / 6 配对更快，平均时长比 1.415；质量差值与对应 GRPO 矩阵一样
没有系统性退化。该结果证明插件路径可训练、同步、保存和评测，不等于所有
GSPO 工作负载都有同样加速。

### 4.2 容量矩阵

capacity 覆盖 54 个异步配置和 3 个同步基线，包括：

- `queue_capacity` 取 1、2、4；
- `max_inflight_rollouts` 取 1、2；
- `weight_sync_interval_updates` 取 1、2、4；
- `max_policy_lag=4`。

57 / 57 均成功，异步配置未产生 stale drop、同步重叠违规或监控错误；两卡
显存峰值不超过约 5.1 GiB。队列和 inflight 增大没有带来明确吞吐变化，说明
该 64-token 算术负载的主要瓶颈不在就绪队列容量。

### 4.3 batching 瓶颈

6 组 `rollout_batch_groups=2` 与单 group 配对显示：

- 生成 sampled-token 吞吐比均值 1.699，6 / 6 提高；
- updates/s 比均值 0.941，中位数 0.899，仅 2 / 6 提高；
- batched 路径 stale drop rate 为 49.5%--61.0%，均值约 53.4%；
- 准确率差值 5 / 6 不为负，但有一组下降 0.0859。

这说明生成侧 batching 本身有效，但调度器允许同一或相近 policy version 的
结果生产过多；训练消费期间，后续 group 越过 lag 窗口并被丢弃。生成卡完成
了更多工作，却没有稳定增加 optimizer update 数。当前应保留单 group 默认值，
将 group=2 标为实验开关。

## 5. 故障与恢复

`faults` 和 `extended-faults` 共执行 9 个 case，覆盖：

- policy receive failure；
- worker 非正常退出；
- operation timeout；
- SIGINT；
- SIGTERM；
- 故障后的 restart / 正常完成对照。

所有预期故障都被观察到；成功 checkpoint 和 optimizer checkpoint 得到保留；
退出后 `live_children` 为空。该结果证明异常传播、关闭和 LoRA checkpoint 保留
路径可用，不代表 strict full-parameter resume 已通过。

## 6. 未覆盖与发布边界

- strict full-parameter resume 在 T4 上反向阶段 OOM，尚未验收。
- 当前主要证据属于 LoRA/adapter、单机双 GPU、native attention。
- 尚未覆盖更大模型、长时间训练、多节点、TP/DP、Flash Attention/L4。
- 55 steps 和三个 seed 足以检查工程链路及短程收益，不足以证明长期收敛。
- `lag0` 在本轮 quality 矩阵中 6 / 6 都慢于 sync，不应作为性能默认值。

## 7. 下一阶段

优先解决 batching 的无效过量生产：

1. 为 group batch 的 admission 建立基于 policy version、lag budget 和当前在途
   group 数的容量约束。
2. 在启动 rollout 前保留可消费的版本窗口，避免结果返回后才批量丢弃。
3. 保持 EOF、取消、同步优先级和 permit 归还语义不变，并增加对应 CPU 测试。
4. 用原 6 组 T4 配对复验。验收目标是 generation throughput 不显著回退，
   stale drop rate 从约 53% 降至 10% 以下，且 6 组 updates/s 中位数高于
   单 group。
5. batching 稳定后，再在更大显存服务器完成 full-parameter resume。
