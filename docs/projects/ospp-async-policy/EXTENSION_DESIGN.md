# 后续性能与范围扩展设计

日期：2026-09-16。本文是 P1/P2/E3/E4 的设计草案，不表示能力已实现。

## P1：先考虑一次生成多个完整 prompt group

现状是单一 rollout lease 和逐组 session。K 允许生成之外的 CPU 工作重叠，
不增加 GPU 上并发 session。GPU 利用率低不自动说明 Python 互斥是唯一瓶颈；
应先使用长输出/大 n_samples 矩阵区分小工作负载、prefill/decode 与同步等待。

方案 A 把多个组展开成一个 `RolloutPayload`，通常比多个 session 更容易保留
一致版本。实现必须同时满足：

- 每个 prompt group 仍占一个 K 许可；不能把 K 偷换成 RPC 数量。
- 从唯一数据源读取、展开的 prompt indices 和返回序列必须可逆映射，保留各组边界。
- 一个生成 lease 内捕获同一个 Vb；组内评分/advantage 独立，保持各组完整。
- Q 仍按训练 batch 计数；若合并训练 batch，必须先定义单位和一次 optimizer step 的边界。
- max_running_prompts、KV blocks、prefill tokens 按真实行数计算，设置可测内存上限。
- 部分生成失败、停止及同步请求必须归还所有已获许可，不能只归还一个 RPC 对应的许可。

应先加入 CPU 的多组展开/还原、版本、许可泄漏、异常和同步优先级用例，随后用
相同 seed、设备、长度、n_samples、更新窗口比较生成 token/s、训练 token/s、
stale drop 比例、显存峰值与质量。当前没有性能数据支持采用方案 A 或 B。

方案 B 允许多个 session，需要改变 coordinator 的 bool lease、native 的 session
begin/end 所有权和 worker 端缓存生命周期。引擎能调度多 prompt 不等于它支持
多个互不干扰的外层 session；未完成此验证前不能删除单 session 守卫。

## P2：同步触发点与双缓冲

优先研究在 rollout 完成事件处交接同步：训练端提交“同步已到期”，生成端在当前
session 结束时阻止下一次准入，协调器等训练端可读后独占复制。它可能减少空转或
无效准入，但不会消除实际复制时间，也不能在训练写参数时读取不一致的权重。

双缓冲则需要 rollout 的 active/staging 两份权重，以及明确的发布事件：

1. 捕获训练侧一致快照；训练 lease 或等价的 CUDA event 必须覆盖快照读取。
2. 将完整目标版本写入 staging，接收端确认后才能发布。
3. 在 session 边界切换 active；旧 session 保持原版本直到完成。
4. KV cache、LoRA adapter version、CUDA graph 中捕获的指针必须对应同一版本。
5. 部分复制失败不能污染 active；退出须回收两套 buffer 与未完成传输。

预算包括额外权重显存、快照复制、graph 重建/重捕获和同步通信。收益上限取决于
“准入等待”中多少原本不能被有效工作覆盖；不能把所有等待指标相加后直接当成
可回收 wall time。首版继续使用现有独占同步，不实现双缓冲。

## E3：Agentic / 多模态轨迹

数据层可快照 CPU features，但 `NativeCudaEnginePair.generate` 仍显式拒绝
features。接入需沿公开 `RolloutSession` / `RewardRecord` / `TrainSequence`
契约设计，而不只把 `prompt_features_by_dp` 从 None 改成字典：

- 多轮工具 token、模型 token 和环境 token 的 loss mask 需贯穿 materialization。
- 明确整条轨迹是否固定一个策略版本；若轮间换权重，单一 Vb 不再足以描述样本。
- 每组轨迹完整评分后计算 advantage，保留实际生成 token 与工具结果的对应关系。
- 慢工具/环境应有可取消协议；任意 Python 回调线程仍不能被安全强杀。
- 多模态 features、位置编码、路由 tensor 的所有权和 CPU 无图约束需保持。

验收至少包含多轮 mask 数值对照、工具失败与取消、轨迹中断、版本变更策略，以及
真实 backend 的两步闭环。本轮没有解除原生 text-only 限制。

## E4：TP/DP 大于一

需要同时改动原生适配的分区校验、rank/DP 行映射、训练 pack 的 shard/梯度累积、
同步布局，以及 `areno/engine/checkpoints/training_state.py` 的 `_identity`。
不能只放开构造参数：当前优化器/RNG manifest 是单 rank 文件，多个 rank 直接写
同一路径会产生覆盖和不完整 checkpoint。

后续格式应有每 rank state shard、全局提交 manifest、拓扑与参数分片身份、每 rank
RNG 以及全部 rank 完成后原子发布的协议。续训先限制同拓扑，跨拓扑重分片另做设计。
GPU 验收需覆盖部分 rank 启动/训练/同步失败与 TP/DP shard 的保存恢复一致性。
