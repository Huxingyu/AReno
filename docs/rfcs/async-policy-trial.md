# RFC：AReno 原生异步策略训练器试用

- 状态：Draft / 请求导师评审
- 日期：2026-09-19
- 评审代码：`review/async-v2-06-batching` @ `8713e5c`，基于 `origin/main` @ `1052b58`
- GPU 证据代码：`feat/async-policy-batched` @ `bb8a0637`（见「已有证据」）

## 摘要

提议以 `areno.experimental.async_policy` 的形式，小范围试用单机双 GPU 的异步策略
训练器：rollout/reward 与训练分别占用一块 GPU 并重叠执行，用有界队列、策略版本和
最大 policy lag 约束资源占用与样本陈旧度。同步训练路径的默认行为不变；本次改动对
共享 engine 的修改在「改动范围」中单列。

## 动机

同步 GRPO 中 rollout、reward 和 optimizer step 串行执行，两块 GPU 交替空闲，长输出
或多采样时尤为明显。本方案在不引入 vLLM/SGLang 等外部 runtime 的前提下，用 AReno
原生 backend 提高单机吞吐。

## 方案

```text
rollout GPU -> reward/advantage -> bounded queue -> train GPU
      ^                                      |
      +------ versioned native weight sync --+
```

约束：

- 一块 train GPU、一块 rollout GPU，各自 TP=DP=1，设备不重叠；
- ready queue（Q）、inflight 组数（K）、policy lag 均有硬上限；
- 每个 prompt group 独立评分、独立 optimizer update；
- 权重同步与任何模型操作互斥，策略版本由协调器单点持有；
- 初始化、异常、超时、信号退出和 worker 回收由统一生命周期管理；
- 默认 Q=2、K=1、`lag=1`、单 group；多 group batching 仅作可选实验。

## 改动范围

按六个堆叠提交逐层 review，每层一个分支：

| 层 | 分支 | 内容 |
| --- | --- | --- |
| 1 | `review/async-v2-01-seed` | `cuda/generation.py` 透传采样 seed（修复） |
| 2 | `review/async-v2-02-engine` | `engine/protocol.py`：`TPCluster.close` 支持超时与 kill 升级、新增 `request_shutdown`、关闭时未完成调用抛 RuntimeError；`worker.py` 新增 save/load training-state Op；新增 `engine/checkpoints/training_state.py` |
| 3 | `review/async-v2-03-core` | `experimental/async_policy/` 协调器、队列、pipeline、loss |
| 4 | `review/async-v2-04-native` | 原生 backend 适配、`examples/async_policy/train.py`、cookbook |
| 5 | `review/async-v2-05-validation` | `tests/async_policy_validation/` 与 benchmark 工具 |
| 6 | `review/async-v2-06-batching` | 多 group 生成会话，生成前预留 lag 容量 |

第 1、2 层修改共享路径。`close()` 不传 timeout 时保留原有 5 秒 grace 行为，同步
训练不受影响；相关 CPU 测试（protocol、policy sync、training state、generation）
均通过。未修改 `areno/api` 配置类、CLI 选项和 `pyproject.toml`。

## 已有证据

CPU：`pytest tests/test_async_policy_*.py` 187 passed（8713e5c）。

GPU：来自 `feat/async-policy-rewrite` @ `be8e1e8` 中的
`docs/projects/ospp-async-policy/evidence/kaggle-20260918-stage-report.md`，
双 T4、Qwen3-0.6B LoRA、FP32、native attention、55 updates（前 5 为 warmup）、
seeds 41/42/43，共 132 个训练/评测 job，全部成功。

| 指标 | 结果 |
| --- | --- |
| 吞吐配对（12 组，`lag=1` vs sync） | 12/12 更快；训练时长比均值 1.368，范围 1.265–1.448 |
| 训练 response-token 吞吐 | 54.7 → 76.7 tokens/s，约 +40% |
| 256-token 评测准确率差值（12 组） | 均不为负，平均 +0.0013 |
| quality 矩阵（6 组） | 6/6 更快；12 个评测配对平均 +0.015，10/12 持平或更好 |
| 同步/模型操作重叠违规、monitor 错误 | 全部为 0 |
| 故障注入 | 无遗留子进程 |

边界与负面结果：

- `lag=0` 在 6/6 配对中慢于 sync，不作为默认；
- 多 group batching 修复前 stale drop 约 53%，第 6 层为此修复，尚未 GPU 复测；
- full-parameter strict resume 在 T4 反向阶段 OOM，未验收；
- 3 seeds × 55 steps 只支持短程「不退化」结论。

证据提交 `bb8a0637` 与评审 tip `8713e5c` 的差异仅在第 6 层的 coordination/pipeline
（约 145 行）。默认单 group 路径的 GPU 数字来自 `bb8a0637`；若导师要求，可在双 T4
上以相同 seeds 对 `8713e5c` 复跑 6 组配对。

## 试用范围与验收

首轮配置：单机双 GPU、Qwen3-0.6B、LoRA、GRPO、文本 completion；同步基线与异步
`lag=1` 使用相同模型、数据、seed 和步数；3 seeds、55 updates。

记录并以下列门槛判定：

| 指标 | 门槛 |
| --- | --- |
| updates/s、response tokens/s | 每个 seed 的 `lag=1` 不低于 sync |
| stale drop rate | 单 group < 5%；多 group 实验 < 10% |
| 256-token 评测准确率 | 相对 sync 的平均差值 ≥ −0.01 |
| 策略版本违规、模型操作重叠 | 0 |
| 退出完整性 | 每次运行 `worker_exits` 为两个已退出 worker |

任一门槛不满足即停止试用，回到同步路径。

## 本轮不包含

- full-parameter strict resume；
- 多节点、TP/DP 大于 1；
- multimodal 与完整 agentic workload；
- 大模型与长期收敛结论；
- 并入稳定 API 或主 CLI。

## 请求决策

1. 是否同意以 experimental 能力进入小范围双 GPU 试用；
2. 是否同意按上表六层逐层 review，其中第 1、2 层作为共享路径改动优先审查；
3. 是否接受 full-parameter resume 和更大拓扑放入后续工作。
