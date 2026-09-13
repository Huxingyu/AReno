# AReno 本机同步 GRPO 基线

日期：2026-09-13。目标是先验证现有同步训练，再以它作为异步改造的对照。

## 代码与环境

- 仓库：<https://github.com/inclusionAI/AReno>
- 基线提交：`60e2fa73f331ea5f5f7245f0793084fbebe9d76c`
- 分支：`exp/sync-smoke-20260913`
- 测试在独立的同步基线 worktree 中执行；Fork 的资料分支保存其记录。
- GPU：RTX 3060 Laptop，6 GiB，Compute Capability 8.6。
- 主机内存：约 16 GiB。
- Python 3.12.3；复用本机 PyTorch 2.11.0+cu128。
- 外盘安装 Transformers 5.17.0、ModelScope 1.40.0、safetensors 0.8.0 等依赖。
- NVIDIA CUDA 12.8.93 编译器安装在镜像内，官方 redistrib 文件均核对 SHA-256。
- AReno 自有 CUDA 扩展编译成功，`areno check` 返回 ready。
- 使用 native attention，关闭 torch.compile，保留 decode CUDA graphs。
- FlashAttention 与 FLA 未安装；本次 Qwen3 native 路径不依赖它们。

## 实验存储

本机使用外置存储上的 100 GiB 稀疏 ext4 镜像，文件系统可用约 98 GiB，
本次运行后底层实际占用约 4.6 GiB。环境、模型、工具链、缓存和 checkpoint 保存在该实验空间。
仓库保存文档、算术输入和小型结果摘要；运行产生的模型与完整日志保存在实验环境。

## 工作负载

- 模型：`Qwen/Qwen3-0.6B`，由 ModelScope `snapshot_download` 下载，实际权重文件 1,503,300,328 字节。
- 算法：仓库原有同步 GRPO，复用 `grpo_loss_fn`。
- 调参方式：LoRA rank 8，alpha 16，覆盖 attention 和 MLP 的 7 类投影。
- 可训练 adapter 参数：5,046,272；其余基础模型参数冻结。
- 数据：本地生成的 8 道简单算术题，[arithmetic-smoke.jsonl](./data/arithmetic-smoke.jsonl)；通过仓库数据检查脚本。
- 奖励：仓库原有 `examples/math/math_verify_reward.py`。
- 单卡，TP=1；每步 1 个 prompt，每个 prompt 采样 4 个回答；最多同时生成 2 个回答。
- 训练 microbatch=1，最多输入 128 tokens，最多生成 64 tokens，关闭 thinking。
- 学习率 1e-4，恒定学习率，运行 3 个 optimizer steps。
- 当前 CLI 数据流只在单张卡上按顺序 rollout/reward/train，权重由生成和训练共用。

## 验证结果

1. `smoke-infer` 成功。
2. `smoke-train` 成功。
3. 真实 GRPO 三步完整完成，保存 adapter 成功。
4. adapter 成功重新加载；重导出的 392 个张量与原始保存值完全一致。
5. 所有保存的 adapter 张量均为有限值；196 个初始为零的 LoRA B 张量都变为非零。
6. 重载后对 `Calculate 2 + 2` 生成包含 `\boxed{4}` 的答案，原数学奖励函数评分为 1。

下表来自 TensorBoard 事件，由仓库 `read_metrics.py` 提取：

| 训练步 | adapter version | 平均奖励 | 梯度范数 | 生成时间 / 秒 | 训练时间 / 秒 | step 总时间 / 秒 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.50 | 1.29397 | 2.34075 | 0.83544 | 4.79215 |
| 1 | 2 | 0.50 | 1.12459 | 1.45463 | 0.62884 | 2.11386 |
| 2 | 3 | 1.00 | 0 | 1.30159 | 0.61172 | 2.06785 |

第三步同组回答奖励相同，group-relative advantages 为零；因此该步策略梯度为零。
当前 loss 的标量值均为零，但前两步梯度非零，不能通过 loss 数值为零判断没有训练。
这些仅是短任务上的运行数据，不能用来证明奖励提升、算法收敛或异步性能。

真实训练的 GPU 用量由 `nvidia-smi` 每 500 ms 采样，169 个样本中的
峰值设备显存用量为 **3572 MiB，约 3.49 GiB**。采样可能遗漏瞬时峰值，
该数据不是全参数训练或更长序列的容量承诺。

## 复现参数与证据

本节来自 2026-09-13 已完成的同步训练。先按主仓库的 CUDA 安装说明准备可用环境，
从 ModelScope 获取 Qwen/Qwen3-0.6B，并设置 `ARENO_MODEL_PATH` 为其本地权重目录。
在仓库根目录执行以下步骤；命令保留了原测试的参数，仅调整数据和输出位置。
本次归档没有重新执行 GPU 训练。

```bash
set -euo pipefail
: "${ARENO_MODEL_PATH:?Set ARENO_MODEL_PATH to the local Qwen3-0.6B model directory}"
mkdir -p runs
ARENO_RUN_DIR="$(mktemp -d runs/sync-smoke-XXXXXX)"
python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
  --dataset-path docs/projects/ospp-async-policy/data/arithmetic-smoke.jsonl \
  --model-hub modelscope --algo grpo
export TORCHDYNAMO_DISABLE=1
python -m areno.cli.main train \
  --ckpt "$ARENO_MODEL_PATH" \
  --base-model-name-or-path Qwen/Qwen3-0.6B \
  --model-hub modelscope \
  --dataset-path docs/projects/ospp-async-policy/data/arithmetic-smoke.jsonl \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo grpo \
  --world-size 1 \
  --tp-size 1 \
  --epochs 1 \
  --max-steps 3 \
  --batch-size 1 \
  --n-samples 4 \
  --max-running-prompts 2 \
  --mini-bs 1 \
  --max-prompt-tokens 128 \
  --max-new-tokens 64 \
  --disable-thinking \
  --drop-rollout-state \
  --attn-backend native \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lr 0.0001 \
  --min-lr 0.0001 \
  --lr-decay-style constant \
  --metrics-log-dir "$ARENO_RUN_DIR/metrics" \
  --save-path "$ARENO_RUN_DIR/adapter" \
  --save-interval 3
```

首次三步训练没有逐条记录 completion；下面的回答样本来自 checkpoint 重载后的生成。
结果目录标识为 `sync-train-nkIhSy`。已归档的证据包括：

- [TensorBoard 指标摘要](./evidence/sync-baseline-metrics.json)：原事件文件提取的全部标量序列。
- [GPU 采样摘要](./evidence/sync-baseline-gpu.json)：采样数量、间隔与峰值。
- [adapter 重载校验](./evidence/sync-baseline-reload.json)：参数数量、逐张量重导出与 logprob 检查。
- [重载生成样本](./evidence/sync-baseline-sample.json)：算术回答及奖励。
- [证据范围说明](./evidence/README.md)：原始文件来源与路径规范化规则。

完整 TensorBoard events、GPU 逐条采样、训练日志和约 20 MB 的 adapter checkpoint
保存在实验环境；上述摘要用于复核本报告的数据与范围。

## 接下来的异步工作

先保持这个同步 recipe 作为回归基线。下一步是用 CPU fake backend 串起
worker、完整 prompt group、有界队列、训练、版本同步与退出；修复现有协调器
等待同步时仍接收新工作的缺口，再接真实 backend。

本机能够验证小模型单卡训练和 CPU 调度逻辑。独立生成 GPU 与训练 GPU 的并行
吞吐需要至少两个设备组实测。建议后续准备 2×24 GB GPU、64 GB 主机内存；
更低配置可以尝试同样的 0.6B LoRA 短任务，但要重新实测容量。
LoRA 的同步体积与全量权重不同，最终应根据课题要求补充相应同步路径的验证。
