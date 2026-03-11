# LaYER — Layered Agent Yielding Evaluated Results

分层竞争式 Agent 编排系统。上层 Agent 拆解任务、发起竞标，多个下层 Agent 提交方案；选出最优方案后在独立 Git Worktree 中并行执行，最终择优合并。递归直到深度上限或方案趋同。

## 核心机制

1. **分解** — Agent 0 将任务拆分为子任务，每个子任务形成独立竞争池
2. **竞标** — 至少 N 个 Agent 逐个提交方案，后者可见前者所有方案并给出批判性改进；方案不再有显著差异时停止
3. **选拔** — 上层 Agent 从中选出 M 个最具多样性的方案
4. **执行** — 被选方案在各自的 Git Worktree 中并行实施（未达深度限制则递归）
5. **评估** — 上层 Agent 对比所有结果，选出最优，合并 Worktree，清理其余

## 设计原则

- **多模型混用** — 通过 `--agent` 配置多种后端（Copilot、Codex、Claude 等），轮询分配以获得多样性
- **全局资源控制** — 信号量限制最大并发 Agent 数，防止资源耗尽
- **趋同短路** — 所有方案实质相同时，跳过递归直接执行
- **透明简洁** — 单文件实现，无框架依赖

## 使用

```bash
# 默认 (copilot)
layer run "实现用户认证模块"

# 多后端竞争
layer run "重构数据库层" \
  -a "copilot -s --model claude-sonnet-4-20250514 -p {prompt}" \
  -a "codex --full-auto -q -p {prompt}"

# 调整参数
layer run "优化搜索算法" -n 5 -m 3 --max-depth 2 --max-agents 32
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--agent, -a` | `copilot -s -p {prompt}` | Agent 命令模板，`{prompt}` 为占位符，可重复 |
| `-n` | 3 | 每个任务最少竞标方案数 |
| `-m` | 2 | 最少并行执行方案数 |
| `--max-depth` | 3 | 最大递归层数 |
| `--max-agents` | 16 | 全局最大并发 Agent 数 |

## 构建

```bash
uv sync
pyinstaller --onefile --name layer main.py
./dist/layer run "your task"
```
