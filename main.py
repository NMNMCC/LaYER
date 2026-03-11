from __future__ import annotations

import itertools
import json
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

import typer

app = typer.Typer(help="LaYER — LaYERed Agent Yielding Evaluated Results")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentBackend:
    """A single agent backend — a command template with {prompt} placeholder."""

    cmd: str  # e.g. "copilot -s --model claude-sonnet-4-20250514 -p {prompt}"

    def render(self, prompt: str) -> str:
        return self.cmd.format(prompt=json.dumps(prompt))

    def __str__(self) -> str:
        return self.cmd


@dataclass
class LayerPool:
    """A pool of backends assigned to a range of depths."""

    backends: list[AgentBackend]
    start_depth: int = 0
    end_depth: int | None = None  # None means inclusive to infinity
    _cycle: itertools.cycle = field(init=False, repr=False)

    def __post_init__(self):
        self._cycle = itertools.cycle(self.backends)

    def pick(self) -> AgentBackend:
        return next(self._cycle)

    def matches_depth(self, depth: int) -> bool:
        if depth < self.start_depth:
            return False
        if self.end_depth is not None and depth > self.end_depth:
            return False
        return True


DEFAULT_CMD = "copilot -s -p {prompt}"


@dataclass
class Config:
    max_depth: int = 3
    min_competitors: int = 3
    min_selected: int = 2
    max_agents: int = 16
    # depth -> pool; depth not found falls back to highest defined layer
    layer_pools: dict[int, LayerPool] = field(default_factory=dict)

    def pool_for_depth(self, depth: int) -> LayerPool:
        if depth in self.layer_pools:
            return self.layer_pools[depth]
        # fallback: nearest lower defined layer, then nearest higher
        defined = sorted(self.layer_pools.keys())
        if not defined:
            return LayerPool(backends=[AgentBackend(cmd=DEFAULT_CMD)])
        below = [d for d in defined if d <= depth]
        return self.layer_pools[below[-1]] if below else self.layer_pools[defined[0]]

    def pick_backend(self, depth: int) -> AgentBackend:
        return self.pool_for_depth(depth).pick()


# ---------------------------------------------------------------------------
# Global agent counter
# ---------------------------------------------------------------------------


class AgentPool:
    def __init__(self, limit: int):
        self._sem = threading.Semaphore(limit)
        self._count = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        got = self._sem.acquire(timeout=0)
        if got:
            with self._lock:
                self._count += 1
        return got

    def release(self):
        self._sem.release()
        with self._lock:
            self._count -= 1

    @property
    def active(self) -> int:
        with self._lock:
            return self._count


_pool: AgentPool | None = None


def get_pool(cfg: Config) -> AgentPool:
    global _pool
    if _pool is None:
        _pool = AgentPool(cfg.max_agents)
    return _pool


# ---------------------------------------------------------------------------
# Helpers — call copilot
# ---------------------------------------------------------------------------


def call_agent(
    cfg: Config,
    prompt: str,
    depth: int = 0,
    cwd: str | None = None,
    backend: AgentBackend | None = None,
) -> str:
    """Run a sub-agent and return its stdout."""
    be = backend or cfg.pick_backend(depth)
    cmd = be.render(prompt)
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Agent [{be}] failed: {result.stderr[:500]}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Git worktree helpers
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(out)


def create_worktree(tag: str) -> Path:
    root = repo_root()
    branch = f"layer-{tag}"
    wt_dir = root.parent / f".layer-worktrees" / branch
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "branch", branch, current], check=True, capture_output=True)
    subprocess.run(
        ["git", "worktree", "add", str(wt_dir), branch], check=True, capture_output=True
    )
    return wt_dir


def remove_worktree(wt_dir: Path):
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(wt_dir)], capture_output=True
    )
    branch = wt_dir.name
    subprocess.run(["git", "branch", "-D", branch], capture_output=True)


def merge_worktree(wt_dir: Path):
    branch = wt_dir.name
    subprocess.run(
        ["git", "merge", branch, "--no-edit"], check=True, capture_output=True
    )
    remove_worktree(wt_dir)


# ---------------------------------------------------------------------------
# Core flow
# ---------------------------------------------------------------------------


@dataclass
class Proposal:
    agent_id: str
    plan: str
    key_details: str
    backend: AgentBackend | None = None


@dataclass
class ExecutionResult:
    agent_id: str
    proposal: Proposal
    worktree: Path
    output: str
    success: bool


def gather_proposals(
    cfg: Config, pool: AgentPool, task: str, depth: int
) -> list[Proposal]:
    """Phase 1: sequentially collect competing proposals until convergence."""
    proposals: list[Proposal] = []
    for i in range(cfg.min_competitors * 3):  # hard upper bound
        if not pool.acquire():
            print(
                f"  [depth={depth}] Agent pool exhausted, stopping proposals at {len(proposals)}"
            )
            break
        try:
            prior = (
                "\n---\n".join(
                    f"Proposal {j + 1}:\n{p.plan}" for j, p in enumerate(proposals)
                )
                or "(none yet)"
            )

            prompt = dedent(f"""\
                You are a competing agent bidding on a task. You MUST respond with valid JSON only.
                Task: {task}

                Previous proposals from other agents:
                {prior}

                Provide a NEW proposal that is meaningfully different from all previous ones.
                If you cannot think of a substantially different approach, set "is_novel" to false.

                Respond ONLY with JSON:
                {{"is_novel": true/false, "plan": "your high-level plan", "key_details": "critical implementation details"}}
            """)
            be = cfg.pick_backend(depth)
            raw = call_agent(cfg, prompt, depth=depth, backend=be)
            # Extract JSON from response
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Try to find JSON in the output
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(raw[start:end])
                else:
                    print(
                        f"  [depth={depth}] Agent {i} returned invalid JSON, skipping"
                    )
                    continue

            if not data.get("is_novel", True) and len(proposals) >= cfg.min_competitors:
                print(
                    f"  [depth={depth}] Proposals converged after {len(proposals)} agents"
                )
                break

            proposals.append(
                Proposal(
                    agent_id=f"d{depth}-c{i}-{uuid.uuid4().hex[:6]}",
                    plan=data.get("plan", ""),
                    key_details=data.get("key_details", ""),
                    backend=be,
                )
            )
            print(
                f"  [depth={depth}] Proposal {len(proposals)} [{be}]: {proposals[-1].plan[:80]}..."
            )
        finally:
            pool.release()

    return proposals


def select_proposals(
    cfg: Config, pool: AgentPool, task: str, proposals: list[Proposal], depth: int
) -> list[Proposal]:
    """Phase 2: upper agent selects M diverse proposals."""
    if len(proposals) <= cfg.min_selected:
        return proposals

    summary = "\n---\n".join(
        f"ID: {p.agent_id}\nPlan: {p.plan}\nDetails: {p.key_details}" for p in proposals
    )
    prompt = dedent(f"""\
        You are a manager agent selecting the best proposals for a task.
        Task: {task}

        Proposals:
        {summary}

        Select at least {cfg.min_selected} proposals that are most promising AND maximally diverse.
        Respond ONLY with a JSON list of selected agent IDs:
        ["id1", "id2", ...]
    """)

    if not pool.acquire():
        return proposals[: cfg.min_selected]
    try:
        raw = call_agent(cfg, prompt, depth=depth)
        try:
            ids = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                ids = json.loads(raw[start:end])
            else:
                return proposals[: cfg.min_selected]

        id_set = set(ids)
        selected = [p for p in proposals if p.agent_id in id_set]
        if len(selected) < cfg.min_selected:
            selected = proposals[: cfg.min_selected]
        return selected
    finally:
        pool.release()


def execute_proposal(
    cfg: Config, pool: AgentPool, task: str, proposal: Proposal, depth: int
) -> ExecutionResult:
    """Phase 3: execute a single proposal in its own worktree."""
    tag = proposal.agent_id
    wt = create_worktree(tag)
    print(
        f"  [depth={depth}] Executing {proposal.agent_id} [{proposal.backend}] in {wt}"
    )
    try:
        if depth + 1 < cfg.max_depth:
            # Recurse
            output = run_layer(cfg, pool, task, proposal.plan, depth + 1, cwd=str(wt))
        else:
            # Leaf — actually do the work
            prompt = dedent(f"""\
                You are an implementation agent. Execute this task directly.
                Task: {task}
                Plan to follow: {proposal.plan}
                Key details: {proposal.key_details}
                Implement the solution now. Write/modify files as needed.
            """)
            if not pool.acquire():
                return ExecutionResult(tag, proposal, wt, "pool exhausted", False)
            try:
                output = call_agent(cfg, prompt, cwd=str(wt), backend=proposal.backend)
            finally:
                pool.release()
        return ExecutionResult(tag, proposal, wt, output[:2000], True)
    except Exception as e:
        return ExecutionResult(tag, proposal, wt, str(e)[:500], False)


def evaluate_results(
    cfg: Config, pool: AgentPool, task: str, results: list[ExecutionResult], depth: int
) -> ExecutionResult | None:
    """Phase 4: upper agent picks the best result."""
    successes = [r for r in results if r.success]
    if not successes:
        return None
    if len(successes) == 1:
        return successes[0]

    summary = "\n---\n".join(
        f"ID: {r.agent_id}\nPlan: {r.proposal.plan}\nOutput (excerpt): {r.output[:500]}"
        for r in successes
    )
    prompt = dedent(f"""\
        You are an evaluator agent. Pick the best result for this task.
        Task: {task}

        Results:
        {summary}

        Respond ONLY with the ID of the best result (just the string, no quotes or extra text).
    """)
    if not pool.acquire():
        return successes[0]
    try:
        raw = call_agent(cfg, prompt, depth=depth).strip().strip('"').strip("'")
        for r in successes:
            if r.agent_id in raw:
                return r
        return successes[0]
    finally:
        pool.release()


def run_layer(
    cfg: Config,
    pool: AgentPool,
    task: str,
    context: str = "",
    depth: int = 0,
    cwd: str | None = None,
) -> str:
    """Main recursive entry: decompose → compete → execute → evaluate."""
    indent = "  " * depth
    print(f"{indent}[Layer {depth}] Task: {task[:100]}...")

    full_task = (
        f"{task}\n\nAdditional context from upper layer:\n{context}"
        if context
        else task
    )

    # Phase 1: gather proposals
    proposals = gather_proposals(cfg, pool, full_task, depth)
    if not proposals:
        print(f"{indent}[Layer {depth}] No proposals generated, executing directly")
        if not pool.acquire():
            return "pool exhausted"
        try:
            return call_agent(
                cfg, f"Execute this task directly:\n{full_task}", depth=depth, cwd=cwd
            )
        finally:
            pool.release()

    # Check convergence — if all proposals are essentially the same, execute directly
    if len(proposals) >= cfg.min_competitors:
        convergence_prompt = dedent(f"""\
            Are these proposals essentially identical in approach? Answer ONLY "yes" or "no".
            {chr(10).join(p.plan for p in proposals)}
        """)
        if pool.acquire():
            try:
                answer = (
                    call_agent(cfg, convergence_prompt, depth=depth).strip().lower()
                )
            finally:
                pool.release()
            if "yes" in answer:
                print(
                    f"{indent}[Layer {depth}] All proposals converged, executing directly"
                )
                prompt = dedent(f"""\
                    Execute this task directly. Follow this plan:
                    Task: {full_task}
                    Plan: {proposals[0].plan}
                    Details: {proposals[0].key_details}
                """)
                if pool.acquire():
                    try:
                        return call_agent(cfg, prompt, depth=depth, cwd=cwd)
                    finally:
                        pool.release()

    # Phase 2: select
    selected = select_proposals(cfg, pool, full_task, proposals, depth)
    print(f"{indent}[Layer {depth}] Selected {len(selected)} proposals for execution")

    # Phase 3: execute in parallel worktrees
    results: list[ExecutionResult] = []
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = {
            executor.submit(execute_proposal, cfg, pool, full_task, p, depth): p
            for p in selected
        }
        for f in as_completed(futures):
            results.append(f.result())

    # Phase 4: evaluate
    best = evaluate_results(cfg, pool, full_task, results, depth)
    if best is None:
        print(f"{indent}[Layer {depth}] All executions failed")
        for r in results:
            remove_worktree(r.worktree)
        return "all executions failed"

    print(f"{indent}[Layer {depth}] Winner: {best.agent_id}")

    # Merge winner, clean up losers
    for r in results:
        if r.agent_id != best.agent_id:
            remove_worktree(r.worktree)
    merge_worktree(best.worktree)

    return best.output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_agent_spec(spec: str) -> tuple[list[int], AgentBackend]:
    """Parse 'depth=command template' into (depths, AgentBackend).

    Format: "DEPTH=CMD" or "DEPTH_RANGE=CMD" where CMD contains {prompt}.
    If no "DEPTH=" prefix, defaults to depth 0.

    Depth can be:
        - Single value: "0=CMD" -> [0]
        - Range: "0-2=CMD" -> [0, 1, 2]
        - Comma-separated: "0,1,2=CMD" -> [0, 1, 2]
        - Mixed: "0,2-4=CMD" -> [0, 2, 3, 4]

    Examples:
        "0=copilot -s --model claude-opus-4-20250514 -p {prompt}"
        "0-2=copilot -s --model claude-sonnet-4-20250514 -p {prompt}"
        "1,3=codex --full-auto -q -p {prompt}"
        "copilot -s -p {prompt}"     (implies depth=0)
    """

    def parse_depths(prefix: str) -> list[int]:
        depths = set()
        for part in prefix.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", maxsplit=1)
                start, end = int(start.strip()), int(end.strip())
                depths.update(range(start, end + 1))
            else:
                depths.add(int(part))
        return sorted(depths)

    if "=" in spec:
        prefix, cmd = spec.split("=", maxsplit=1)
        try:
            depths = parse_depths(prefix)
        except ValueError:
            cmd = spec
            depths = [0]
    else:
        cmd = spec
        depths = [0]

    if "{prompt}" not in cmd:
        raise typer.BadParameter(
            f"Agent command must contain {{prompt}} placeholder: {cmd}"
        )
    return depths, AgentBackend(cmd=cmd)


@app.command()
def run(
    task: str = typer.Argument(..., help="The task description"),
    agent: list[str] = typer.Option(
        [],
        "--agent",
        "-a",
        help='Per-layer agent spec: "DEPTH=CMD" or just "CMD" (depth 0). '
        "{prompt} is the placeholder. Repeatable. "
        'Example: -a "0=copilot -s --model opus -p {prompt}" '
        '-a "1=copilot -s --model sonnet -p {prompt}"',
    ),
    max_depth: int = typer.Option(3, help="Maximum recursion depth"),
    min_competitors: int = typer.Option(3, "-n", help="Minimum competing proposals"),
    min_selected: int = typer.Option(2, "-m", help="Minimum parallel executions"),
    max_agents: int = typer.Option(16, help="Global max concurrent agents"),
):
    """Run the LaYER agent system on a task."""
    # Group backends by depth
    depth_backends: dict[int, list[AgentBackend]] = {}
    if agent:
        for spec in agent:
            depths, be = parse_agent_spec(spec)
            for d in depths:
                depth_backends.setdefault(d, []).append(be)
    else:
        depth_backends[0] = [AgentBackend(cmd=DEFAULT_CMD)]

    layer_pools = {d: LayerPool(backends=bes) for d, bes in depth_backends.items()}
    cfg = Config(
        max_depth=max_depth,
        min_competitors=min_competitors,
        min_selected=min_selected,
        max_agents=max_agents,
        layer_pools=layer_pools,
    )
    pool = get_pool(cfg)
    print(
        f"LaYER starting: depth={cfg.max_depth}, N={cfg.min_competitors}, "
        f"M={cfg.min_selected}, m    ax_agents={cfg.max_agents}"
    )
    print("Agent pools:")
    for d in sorted(cfg.layer_pools):
        cmds = cfg.layer_pools[d].backends
        print(f"  Layer {d}: {len(cmds)} backend(s)")
        for b in cmds:
            print(f"    - {b}")
    print(f"Task: {task}\n")

    result = run_layer(cfg, pool, task)
    print(f"\n{'=' * 60}")
    print("Final result:")
    print(result)


if __name__ == "__main__":
    app()
