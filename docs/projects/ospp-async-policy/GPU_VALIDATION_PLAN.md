# GPU 分阶段验收计划

日期：2026-09-16。**本页是待执行计划，没有本轮 GPU 成绩。**
候选采用 `review/async-native` 的 `2b0575a`，与工程实现 `f217bac` 的代码相同。
资源、费用上限及 OSPP 状态待用户确认；本地仅一张 GPU，不能直接运行双 GPU 配方。

## 0. 固定来源与资源

建议先复用已有 Modal 双 L4 环境和缓存。也可选用户指定的双 GPU 主机；其中
`gpu_run.py` 的历史 acceptance probe 显式限定双 L4，换卡需先适配其环境断言，
不能直接当作同资源复现。通用训练/benchmark 示例没有这项 L4 型号限制。

运行前确认：使用哪个主机/账户、首轮费用上限、是否允许推送该候选分支。
这是 [AGENTS.md](../../../AGENTS.md) 的明确要求："Ask first" 包含
"Running GPU training or serving"。远端必须 fetch 已提交的分支；不能复制未提交源码。

批准后在对应本地 worktree 推送 `review/async-native`。Modal 工具会核对本地 clean、
分支远端 HEAD 与候选 SHA，并在远端 fetch 后再次核对。用户指定主机也遵守相同
的 commit/fetch/checkout 流程。模型资产固定 ModelScope `Qwen/Qwen3-0.6B`。

本次未改 `areno/accel`。Modal 镜像复用在上游提交构建的相同扩展，候选 Python
源码不重新安装；若后续 CUDA 构建输入改变，现有 runner 会拒绝执行，届时应重建。

## 1. 六项验收，逐项通过后继续

| 顺序 | phase | 检查目标 |
|---|---|---|
| 1 | `example` | 默认 native attention 的 fresh/resumed 示例、更新版本与退出记录 |
| 2 | `regression` | 独立干净上游的 SDK GRPO/GSPO 对照 |
| 3 | `resume` | LoRA 连续训练与 checkpoint continuation 的参数/optimizer 一致性 |
| 4 | `resume-full` | 全参数 continuation 的同类对照 |
| 5 | `faults` | 接收失败、worker 强退、重启与退出证据 |
| 6 | `extended-faults` | NCCL 等待超时、SIGINT/SIGTERM 和重启 |

每项单独分配双 L4，串行执行，任务 timeout 设为 1200 秒（其中子进程预算 1050 秒）。
六项的任务 timeout 合计 2 小时，等价于 4 GPU 小时的任务执行预算；这不是费用报价，
不含构建、资源启动、存储等账单项。失败即停止后续项，不自动扩 timeout 或重复计费。

以下为 `example` 的可检查命令；其他项只替换 phase 与全新的输出目录：

```bash
python examples/async_policy/tools/modal_run.py \
  --phase example --task-timeout-s 1200 \
  --output-dir runs/async-policy-rewrite/next-gpu/example --dry-run
```

获得资源批准并推送后，去掉 `--dry-run` 才启动远端任务。缓存不可用时先单独运行
`--phase prepare` 下载 ModelScope 模型；该阶段不申请 GPU。每次重跑使用新目录。

验收读取 `request.json`、runner 的 `result.json`、`artifacts/` 内真实任务结果与
`worker_exits`，不能仅凭 Modal 调用返回成功就宣布模型训练成功。

## 2. 种子 43 小矩阵，先确认预算和记录完整性

先跑 8 次更新、2 次 warmup 的四组 pilot。下列参数已可用本地 benchmark dry-run
检查；pilot 只检查执行和资源，不承担 Q1 的退化归因结论。

```bash
python examples/async_policy/tools/modal_run.py \
  --phase benchmark --task-timeout-s 1800 \
  --output-dir runs/async-policy-rewrite/next-gpu/pilot-43-64 --dry-run \
  --benchmark-args --seed 43 --steps 8 --warmup 2 \
  --max-new-tokens 64 --eval-max-new-tokens 64 256 \
  --eval-path examples/async_policy/eval128.jsonl \
  --attn-backend native --cases sync lag1 lag0 offpolicy
```

通过后，再用独立目录跑训练上限 256 的相同 pilot。根据实际耗时、显存和 artifact
体积调整全矩阵资源预算，不能把两项 pilot 当作完整质量证据。

## 3. 完整矩阵

| suite | 配置 | jobs / 训练次数 |
|---|---|---:|
| `quality` | 5 seeds × 2 训练长度 × sync/lag1/lag0/offpolicy；每个 checkpoint 评测 64/256 | 10 / 40 |
| `throughput` | 3 seeds × 256/512 tokens × 4/8 samples × sync/async | 12 / 24 |
| `capacity` | 3 seeds × 18 个 Q/K/C 组合，lag=4；另有同步对照 | 57 / 57 |
| `gspo` | 3 seeds × 2 长度 × sync/async | 6 / 12 |

默认均为 55 次更新、5 次 warmup。以下命令只生成计划：

```bash
python examples/async_policy/tools/matrix.py \
  --suite quality --model-path MODEL_CHECKPOINT \
  --output-dir runs/async-policy-rewrite/quality-v2
```

在已批准且完成源码 checkout 的双 GPU 主机上，用解析后的本地模型路径替换
`MODEL_CHECKPOINT`，加入 `--execute` 才会逐 job 执行。`--summarize` 仅汇总已有结果。
Modal runner 目前按单个 benchmark job 提交；可按 manifest 转发参数。其本地
`result.json` 是 runner 元数据，实际 benchmark `result.json` 位于解包后的
`artifacts/`，汇总时必须使用后者并对应 manifest 中的 job 目录。

## 4. 判读与停止条件

- 每个结果保留源 SHA、模型/数据来源、数据 hash、参数、更新 prompt IDs、版本、
  原始回答、长度、token 上限命中与 reward 曲线。结果不足或设置不符时不汇总为完成。
- 分开比较评测长度效应、训练长度效应、同预算下的调度和目标差异。
- 非有限 loss/梯度、参数或 optimizer 不一致、sync 与模型操作违规重叠、子进程
  无法确认退出时，停止矩阵，保存失败证据并先修复。
- 同步与异步按 seed 配对；保留每个 seed 的结果。速度和正确性分别报告。
- 更大矩阵未复现退化只支持“该矩阵未复现”；合成 128 题与短程训练不证明长期收敛。
- P1 生成 batching 在瓶颈实测后选择方案；P2/E3/E4 仍是设计范围，不随矩阵自动升级为实现。
