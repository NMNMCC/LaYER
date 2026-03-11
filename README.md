# LaYER — Layered Agent Yielding Evaluated Results

A layered competitive agent orchestration system. Upper-layer agents decompose tasks and initiate bidding; multiple lower-layer agents submit proposals. The best proposals are selected and executed in parallel within isolated Git worktrees, with the optimal result merged. This process recurses until reaching the depth limit or proposal convergence.

## Core Mechanisms

1. **Decomposition** — Agent 0 breaks tasks into subtasks, each forming an independent competition pool
2. **Bidding** — At least N agents submit proposals sequentially; later agents see all prior proposals and provide critical improvements; bidding stops when proposals no longer differ significantly
3. **Selection** — Upper-layer agent selects M most diverse proposals
4. **Execution** — Selected proposals are implemented in parallel Git worktrees (recursing if not at depth limit)
5. **Evaluation** — Upper-layer agent compares all results, selects the best, merges its worktree, and cleans up the rest

## Design Principles

- **Multi-model mixing** — Configure multiple backends via `--agent` (Copilot, Codex, Claude, etc.), assigned round-robin for diversity
- **Global resource control** — Semaphore limits maximum concurrent agents to prevent resource exhaustion
- **Convergence short-circuit** — When all proposals are essentially identical, skip recursion and execute directly
- **Transparent simplicity** — Single-file implementation, no framework dependencies

## Usage

```bash
# Default (copilot)
layer run "Implement user authentication module"

# Multi-backend competition
layer run "Refactor database layer" \
  -a "copilot -s --model claude-sonnet-4-20250514 -p {prompt}" \
  -a "codex --full-auto -q -p {prompt}"

# Layered backends
layer run "Build feature X" \
  -a "0=copilot -s --model claude-opus-4-20250514 -p {prompt}" \
  -a "1=copilot -s --model claude-sonnet-4-20250514 -p {prompt}"

# Adjust parameters
layer run "Optimize search algorithm" -n 5 -m 3 --max-depth 2 --max-agents 32
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--agent, -a` | `copilot -s -p {prompt}` | Agent command template with `{prompt}` placeholder; repeatable. Can specify depth: `"DEPTH=CMD"` |
| `-n` | 3 | Minimum competing proposals per task |
| `-m` | 2 | Minimum parallel executions |
| `--max-depth` | 3 | Maximum recursion depth |
| `--max-agents` | 16 | Global maximum concurrent agents |

### Depth-Specific Backends

The `--agent` option supports depth targeting:

```bash
# Single depth
-a "0=copilot -s --model opus -p {prompt}"

# Depth range
-a "0-2=copilot -s --model sonnet -p {prompt}"

# Multiple specific depths
-a "1,3=codex --full-auto -q -p {prompt}"

# Mixed
-a "0,2-4=claude -p {prompt}"
```

## Build

```bash
uv sync
pyinstaller --onefile --name layer main.py
./dist/layer run "your task"
```
