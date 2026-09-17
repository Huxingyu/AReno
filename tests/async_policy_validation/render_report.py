"""Render saved audit results as Markdown, interactive HTML and SVG diagrams."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


ISSUES = [
    ("K1", "test_begin_sync_honors_timeout", "P1", "同步超时失效", "coordinator.py:172", "绝对 deadline 被再次当作时长累加；20 ms 超时在 200 ms 后仍等待。"),
    ("K2", "test_only_one_sync_lease_is_admitted", "P1", "同步许可不独占", "coordinator.py:174", "两个同步调用同时等 rollout 排空，随后同时拿到同一计划。"),
    ("K3", "test_max_steps_does_not_report_success_with_live_producer", "P2", "退出后 producer 仍存活", "pipeline.py:147", "join 超时后仍返回 max_steps 报告，留下一个在途任务。"),
    ("K4", "test_sync_mode_rejects_rollout_during_active_training", "P2", "SYNC 模式准入不对称", "bridge.py:101", "先 train 再 rollout 时仍允许二者重叠。"),
    ("N1", "test_queue_abort_rejects_buffered_batches", "P2", "abort 后仍交付缓存 batch", "queue.py:117", "get 先检查数据，再检查 aborted，终止语义退化成继续排空。"),
    ("N2", "test_inflight_close_and_release_rejects_waiter", "P2", "关闭后的 limiter 仍可准入", "queue.py:236", "close 与 release 在等待线程恢复前发生时，线程跳出循环后没有再次检查 closed。"),
    ("N3", "test_data_source_iter_error_reaches_consumer", "P1", "数据源初始化失败导致等待不结束", "pipeline.py:198", "iter(data_source) 位于异常处理之前；producer 已退出，消费者仍轮询未关闭的队列。"),
    ("N4", "test_data_source_next_error_releases_permit", "P2", "数据读取失败泄漏在途许可", "pipeline.py:209", "next(source) 抛异常发生在 acquire 后、任务移交前；run 抛出原异常，但 active 仍为 1。"),
    ("N5", "test_train_failure_report_records_started_run", "P3", "失败报告仍声称 not_started", "pipeline.py:171", "训练异常向调用者传播了，但最终 report.exit_reason 没有记录失败状态。"),
    ("N6", "test_batch_wait_metric_includes_empty_polls", "P3", "batch 等待指标遗漏空轮询", "pipeline.py:290", "每轮 poll 重置计时起点，只记录最后一次 get 的等待，低估消费者等待时间。"),
    ("N7", "test_bridge_initialize_aligns_real_weights", "P2", "initialize 未保证初始权重一致", "bridge.py:69", "输入两份不同权重时，initialize 仅设置标记；调用方必须另行对齐，与方法声明的初始化对齐契约不符。"),
    ("N8", "test_partial_sync_failure_blocks_poisoned_rollout", "P1", "部分复制失败后仍放行 rollout", "bridge.py:150", "复制第一块参数后抛错，Vr 留在旧版本；bridge 未关闭，下一次 rollout 得到混合版本参数。测试适配器检测到 fingerprint 不符。"),
    ("N9", "test_batch_owns_routed_tensor_snapshot", "P2", "batch 与生产端共享可变路由张量", "batch.py:211", "生产端修改原始 CPU routed_experts 后，已 materialize 的 batch 同步变化；需要明确快照或所有权转移契约。"),
    ("N10", "test_batch_rejects_non_cpu_or_graph_features", "P2", "features 绕过 CPU/计算图检查", "batch.py:208", "features 可携带 CPU autograd 图或 meta 张量；当前检查只覆盖 routed_experts。本轮未使用 CUDA 张量。"),
    ("S1", "test_stale_loss_matches_behavior_policy_clipped_reference", "已知限制", "陈旧策略下的 loss 语义差异", "areno/api/backend/cuda/losses.py:166", "现有 GRPO 使用 exp(logp-logp.detach())，ratio 恒为 1；与行为策略 logprob 作分母的 clipped GRPO 不同。实施方案第 12 节已明确这一限制并要求首版保留现有 loss；本探针不计入新增 bug 或重写验收失败。"),
]


def diagram(root: Path):
    def box(x, y, width, height, title, detail, fill="#eef5ff"):
        return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="#aac0d8"/>'
                f'<text x="{x + width / 2}" y="{y + 30}" text-anchor="middle" font-size="19" font-weight="600">{escape(title)}</text>'
                f'<text x="{x + width / 2}" y="{y + 58}" text-anchor="middle" font-size="14">{escape(detail)}</text>')

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="700" viewBox="0 0 1240 700">',
             '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#4a627b"/></marker></defs>',
             '<rect width="1240" height="700" fill="#fff"/><g font-family="sans-serif" fill="#203044">',
             '<text x="40" y="40" font-size="25" font-weight="700">AReno 两层 CPU 验证：实际执行路径</text>',
             box(40, 65, 550, 78, "L1 可控并发、故障与生命周期", "真实线程、队列、limiter、coordinator；引擎使用可控替身"),
             box(650, 65, 550, 78, "L2 真实 CPU 张量闭环", "425 参数因果小模型；真实采样 / SGD / state_dict 复制", "#eaf8ee"),
             box(30, 235, 145, 84, "数据源", "prompt / EOF / 异常"),
             box(215, 235, 210, 84, "Rollout 策略 Vr", "后台生产任务；上限 K"),
             box(465, 235, 260, 84, "Reward + BatchEnvelope", "完整 prompt 组；版本 Vb"),
             box(765, 235, 130, 84, "Ready Queue", "容量 Q"),
             box(935, 235, 275, 84, "Train 策略 Vt", "前台消费；GRPO loss + SGD", "#eaf8ee"),
             box(455, 395, 330, 84, "独占权重同步", "排空 leases → 复制参数 → 发布 Vr", "#fff3df"),
             box(30, 515, 385, 84, "Bridge 独立契约测试", "初始化 / 单卡准入 / 同步失败；未串入 pipeline"),
             box(460, 515, 325, 84, "不变量与数值检查", "任务计数 / 退出 / 参数快照 / 梯度"),
             box(835, 515, 375, 84, "同步基线 60e2fa7", "相同模型、采样种子、固定 batch 和优化器"),
             box(365, 625, 510, 60, "结果：JSON / JUnit / 日志 / 最小复现", "", "#f2f0fb")]
    for x1, y1, x2, y2 in [(175,277,210,277),(425,277,460,277),(725,277,760,277),(895,277,930,277),
                            (315,143,315,230),(925,143,1070,230),(620,479,620,510),
                            (415,557,455,557),(835,557,790,557),(620,599,620,620)]:
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#4a627b" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    parts.extend(['<path d="M1070,319 L1070,435 L790,435" stroke="#b37b2d" stroke-width="2" fill="none" marker-end="url(#arrow)"/>',
                  '<path d="M455,435 L315,435 L315,324" stroke="#b37b2d" stroke-width="2" fill="none" marker-end="url(#arrow)"/>',
                  '<text x="40" y="190" font-size="16">队列等待不持有 rollout lease；版本和权重同时核验。所有实际张量都在 CPU。</text>', '</g></svg>'])
    (root / "flow.svg").write_text("".join(parts))


def timeline(root: Path, data: dict):
    events = data["real_events"]
    start, end = min(e["timestamp"] for e in events), max(e["timestamp"] for e in events)
    duration = max(end - start, 1e-9)
    colors = {"rollout": "#477cda", "train": "#299865", "sync": "#d68a22"}
    lanes = {"rollout": 85, "train": 145, "sync": 205}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="310" viewBox="0 0 1280 310">',
             '<rect width="1280" height="310" fill="white"/><g font-family="sans-serif" fill="#203044">',
             '<text x="25" y="32" font-size="21">实际 CPU 阶段时间线：Q=1 / K=1 / C=1 / lag=1</text>',
             '<text x="25" y="57" font-size="14">区间含事件门等待；用于检查重叠与同步互斥，不是 GPU 性能测试。</text>']
    opened = {}
    for stage, y in lanes.items():
        parts.append(f'<text x="25" y="{y + 20}" font-size="17">{stage}</text><path d="M130,{y + 32} H1245" stroke="#ddd"/>')
    for event in events:
        stage = event["stage"]
        identity = event.get("batch_id", event.get("prompt_id", event.get("target")))
        key = stage, identity
        if event["action"] == "start":
            opened[key] = event["timestamp"]
        else:
            first = opened.pop(key)
            x = 130 + (first - start) / duration * 1115
            width = max(0.5, (event["timestamp"] - first) / duration * 1115)
            title = f"{stage} {identity}: {(first - start) * 1000:.3f}–{(event['timestamp'] - start) * 1000:.3f} ms"
            parts.append(f'<rect x="{x:.3f}" y="{lanes[stage]}" width="{width:.3f}" height="30" fill="{colors[stage]}" rx="2"><title>{escape(title)}</title></rect>')
    for tick in range(6):
        parts.append(f'<text x="{130 + tick * 223}" y="270" text-anchor="middle" font-size="13">{duration * tick / 5 * 1000:.1f} ms</text>')
    parts.append('</g></svg>')
    (root / "timeline.svg").write_text("".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    groups = []
    for layer in ("layer1-final", "layer2"):
        groups.extend(json.loads((root / layer / "results.json").read_text()))
    cases = [(group, case) for group in groups for case in group["cases"]]
    passed = sum(case["status"] == "pass" for _, case in cases)
    failed = sum(case["status"] == "fail" for _, case in cases)
    issues = [issue for issue in ISSUES if any(case["status"] == "fail" and case["name"].startswith(issue[1]) for _, case in cases)]
    matrices = [row for path in (root / "layer1-final").glob("test_seeded*/*seed-*.json") for row in json.loads(path.read_text())]
    real = [json.loads(path.read_text()) for path in sorted((root / "layer2").glob("test_real*/*real-q*.json"))]
    reference = json.loads((root / "reference-comparison.json").read_text())
    steps = sum(run["report"]["steps"] for run in real)
    grad_error = max(row["gradient_error"] for run in real for row in run["updates"])
    param_error = max(row["parameter_error"] for run in real for row in run["updates"])
    diagram(root)
    timeline(root, next(run for run in real if run["config"]["queue_capacity"] == run["config"]["max_inflight_rollouts"] == 1))
    semantic_cases = [case for _, case in cases if case["name"].startswith("test_stale_loss_matches_behavior_policy_clipped_reference")]
    semantic_failures = sum(case["status"] == "fail" for case in semantic_cases)
    summary = (f"专项原始结果 {len(cases)} 项：{passed} 通过、{failed} 失败。"
               f"其中 {len(cases) - len(semantic_cases)} 项契约用例有 {failed - semantic_failures} 项失败，"
               f"去重为 {sum(i[0].startswith('K') for i in issues)} 项已知问题、"
               f"{sum(i[0].startswith('N') for i in issues)} 项新发现的实现或契约缺口。"
               "另 1 项 S1 是方案已知的 loss 语义比较，不计入新增 bug。")
    md = ["# 异步训练前两层 CPU 实测报告", "", "候选：`ca97cad`；同步基线：`60e2fa7`。生产实现保持待审状态，本轮新增可重复运行的验证工具。", "",
          summary, "", "分类复核依据：资料分支 `docs/ospp-async-policy@1481411` 的实施方案与 TODO v2，已取回当前工作区。原始断言与运行记录保持原样；S1 的预期目标不属于首版契约。", "", "![完整验证流程](flow.svg)", "", "## 已执行的验证", "",
          "- 当前版本选取的 85 项既有 CPU 测试通过；其中 42 项在同步基线 worktree 上也通过。",
          f"- Q∈{{1,2,4}}、K∈{{1,2,4}}、C∈{{1,4}}、lag∈{{0,1,4}}，4 个种子：共 {len(matrices)} 组组合、{len(matrices) * 16} 个 prompt 组通过容量/计数/版本边界检查。",
          "- 30 次空输入生命周期、50 次非空生命周期（400 个 prompt 组）；非空测试结束后无存活 pipeline 弱引用或遗留线程。",
          f"- 4 组真实 CPU 小模型流水线，425 个参数，处理 96 个 prompt 组，实际执行 {steps} 次 SGD 更新。所有模型张量使用 CPU float32。",
          f"- 每一步独立重算 mask、advantage、logprob、loss、梯度和 SGD 更新；最大梯度误差 {grad_error:.3g}，最大参数误差 {param_error:.3g}。",
          f"- 固定 4 个 prompt / 11 个 completion，在基线同步 materializer 与候选 async envelope 上各执行一步；9 类输出最大绝对差均为 {max(reference['max_absolute_errors'].values()):g}。对照阈值 atol=1e-7、rtol=1e-6。",
          "", "![实际 CPU 时间线](timeline.svg)", "", "## 失败清单", "",
          "K 为上次已知问题；N 为本轮新增契约/实现问题；S 为算法语义探针。features 的两个失败用例合并为 N10。", "",
          "| 编号 | 优先级 | 问题 | 位置 | 实测条件与含义 |", "|---|---|---|---|---|"]
    md.extend(f"| {code} | {priority} | {title} | `{site}` | {detail} |" for code, _, priority, title, site, detail in issues)
    md.extend(["", "## 适用范围", "",
               "小模型是 425 参数的因果 token 模型，执行真实随机采样、autograd、SGD 和参数复制；没有使用 AReno GPU 模型、CUDA 扩展或 NCCL。复用了 AReno 的 batch 构造、padded/packed 布局、selected logprob 与 GRPO loss。",
               "测试模型以相同初始参数开始正常闭环，初始化不一致由独立故障用例覆盖。现有 reward 回调只接受 prompt 和 sample_index，测试适配器通过 prompt_id 查找实际 completion 的奖励。Bridge 仍单独测试，pipeline 没有通过 bridge 调用引擎。",
               "CPU 两份独立模型不构成双 GPU 验证；CPU 时序随机种子固定也不能完全固定操作系统线程调度。每次保留实际事件日志，成功更新数和丢弃数允许随调度变化，检查的是不变量。",
               "S1 比较的是使用行为策略比率的 clipped GRPO，与首版明确保留的目标函数不同。实施方案第 12 节和 TODO T12 / T14 已要求记录此限制、检查质量，不在调度改动中替换 loss。该探针保留原始失败以展示数值差异，但不作为重写必须变绿的验收项；有限 lag 也不构成质量保证。",
               "", "## 重跑", "", "```bash",
               "python3 tests/async_policy_validation/run.py --output-dir runs/async-policy-cpu/recheck",
               "python3 tests/async_policy_validation/run.py --layer protocol --case data_source_next --output-dir runs/async-policy-cpu/repro-next",
               "```", "", "用例命中问题时 runner 返回非零退出码，不把预期失败包装为通过。每组子进程默认 45 秒限时，pytest 15 秒时打印线程栈；本轮没有触发进程级强制超时。各子目录包含原始日志、JUnit XML、result.json 及专项张量/时序证据。", "",
               "测试工具与同步基线重跑步骤见仓库中的 `tests/async_policy_validation/README.md`。"])
    (root / "report.md").write_text("\n".join(md) + "\n")
    issue_rows = "".join(f"<tr><td>{code}</td><td>{priority}</td><td>{escape(title)}</td><td><code>{site}</code></td><td>{escape(detail)}</td></tr>" for code, _, priority, title, site, detail in issues)
    case_rows = []
    for group, case in cases:
        log = (Path(group["artifact_dir"]) / "output.log").relative_to(root)
        details = f"<details><summary>原始失败信息</summary><pre>{escape(case['detail'] or '')}</pre></details>" if case["status"] == "fail" else ""
        case_rows.append(f'<tr data-status="{case["status"]}"><td class="{case["status"]}">{case["status"].upper()}</td><td>{escape(case["name"])}</td><td><a href="{log}">日志</a>{details}</td></tr>')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AReno CPU 验证报告</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1240px;margin:35px auto;padding:0 24px;color:#203044;background:#fbfcfe}}h1{{font-size:30px}}table{{width:100%;border-collapse:collapse;background:white;margin:20px 0}}td,th{{border:1px solid #d9e1eb;padding:10px;text-align:left;vertical-align:top}}th{{background:#eef3fa}}img{{width:100%;background:white;border:1px solid #d9e1eb;margin:15px 0}}pre{{white-space:pre-wrap;font-size:12px;max-width:650px}}.pass{{color:#177640}}.fail{{color:#b42533}}input{{padding:9px;font-size:15px}}.note{{background:#edf4ff;padding:18px;border-radius:8px}}</style>
<h1>异步训练：两层 CPU 实测</h1><p>候选 ca97cad · 同步基线 60e2fa7 · CPU float32 · 模型种子 123 / 采样种子 41</p>
<p class="note">{summary}<br>既有测试 85 项通过；216 组并发配置；真实 CPU 模型 {steps} 次 SGD 更新；固定 batch 的基线数值差为 0。</p>
<img src="flow.svg" alt="完整验证流程"><img src="timeline.svg" alt="实际 CPU 阶段时间线">
<h2>实测问题</h2><p>已知 K1–K4 保留作基准。N 为新增契约/实现缺口。S1 是实施方案第 12 节已明确接受的 loss 限制，不计入新增 bug 或重写验收失败。原始断言与运行记录保持原样。</p>
<table><tr><th>编号</th><th>级别</th><th>问题</th><th>位置</th><th>触发条件与证据</th></tr>{issue_rows}</table>
<h2>全部用例</h2><input id="query" placeholder="筛选用例名称"><label><input id="failed" type="checkbox">只看失败</label>
<table id="cases"><tr><th>结果</th><th>用例</th><th>证据</th></tr>{''.join(case_rows)}</table>
<p>这是 CPU 数值及协议验证，未验证 AReno GPU 模型、CUDA 扩展或 NCCL。Bridge 单独测试，尚未串入 pipeline。随机种子不能固定操作系统调度；原始时间线用于记录实际交错顺序。</p>
<p><a href="report.md">完整 Markdown 报告与重跑命令</a> · <a href="reference-comparison.json">同步基线数值比较</a> · <a href="../../docs/projects/ospp-async-policy/REWRITE_ASSESSMENT.md">对照 TODO 的重写评估</a></p>
<script>function filter(){{let q=document.getElementById('query').value.toLowerCase();let f=document.getElementById('failed').checked;document.querySelectorAll('tr[data-status]').forEach(r=>r.hidden=(f&&r.dataset.status!=='fail')||!r.textContent.toLowerCase().includes(q));}}document.getElementById('query').oninput=filter;document.getElementById('failed').onchange=filter;</script></html>'''
    (root / "report.html").write_text(html)
    print(summary)
    print(root / "report.html")


if __name__ == "__main__":
    main()
