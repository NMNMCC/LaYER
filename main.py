import asyncio
import itertools
import json
import os
import shlex
import shutil
import subprocess
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

import typer
from rich.console import Console
from rich.live import Live
from rich.tree import Tree

from cache import (
    CacheEntry,
    compute_task_hash,
    lookup_cache,
    save_cache,
)

app = typer.Typer(help="LaYER — LaYERed Agent Yielding Evaluated Results")

# maximum characters to show in brief summaries
SUMMARY_MAX_LEN = 80

# --- Rich-based status tree UI ------------------------------------------------


class StatusTree:
    """Thread-safe live-updating tree of current state using rich.Live.

    Messages are stored per numeric depth as a list of (text, style) tuples.
    style is any valid rich style name (e.g., 'blue'); None means default style.
    """

    def __init__(self) -> None:
        self.console = Console()
        self.lock = threading.Lock()
        self.messages: dict[int, list[tuple[str, str | None]]] = defaultdict(list)
        self.header: str = "LaYER"
        self._live: Live | None = None

    def start(self) -> None:
        with self.lock:
            self.messages.clear()
            self.header = "LaYER"
            if self._live is None:
                self._live = Live(
                    self._render(), console=self.console, refresh_per_second=4
                )
                self._live.__enter__()

    def stop(self) -> None:
        with self.lock:
            if self._live is not None:
                try:
                    self._live.__exit__(None, None, None)
                finally:
                    self._live = None

    def set_header(self, text: str) -> None:
        with self.lock:
            self.header = text
            if self._live:
                self._live.update(self._render())

    def add(self, depth: int, message: str, style: str | None = None) -> None:
        """Add a message under a numeric depth. style is a rich style (e.g. 'blue')."""
        with self.lock:
            self.messages[depth].append((message, style))
            if self._live:
                self._live.update(self._render())

    def _render(self) -> Tree:
        root = Tree(self.header)
        # do NOT render top-level meta (-1) per user preference
        for depth in sorted(k for k in self.messages.keys() if k >= 0):
            node = root.add(f"Layer {depth}")
            for text, style in self.messages[depth]:
                if style:
                    node.add(text, style=style)
                else:
                    node.add(text)
        return root


# module-level UI instance
ui = StatusTree()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentBackend:
    """A single agent backend — a command template with {prompt} placeholder."""

    cmd: str

    def render(self, prompt: str) -> list[str]:
        if "{prompt}" not in self.cmd:
            raise ValueError(f"Command must contain {{prompt}}: {self.cmd}")
        start = self.cmd.find("{prompt}")
        end = start + len("{prompt}")
        prefix = self.cmd[:start].rstrip()
        suffix = self.cmd[end:].lstrip()
        cmd_list = []
        if prefix:
            cmd_list = shlex.split(prefix)
        cmd_list.append(prompt)
        if suffix:
            cmd_list.extend(shlex.split(suffix))
        return cmd_list

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
        # Single-threaded asyncio — no lock needed
        return next(self._cycle)

    def matches_depth(self, depth: int) -> bool:
        if depth < self.start_depth:
            return False
        if self.end_depth is not None and depth > self.end_depth:
            return False
        return True


DEFAULT_CMD = "opencode run {prompt}"


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
        self._sem = asyncio.Semaphore(limit)
        self._count = 0

    async def acquire(self) -> bool:
        """Non-blocking try-acquire. Returns False immediately if pool is full."""
        if self._sem.locked():
            return False
        await self._sem.acquire()
        self._count += 1
        return True

    def release(self) -> None:
        self._sem.release()
        self._count -= 1

    @property
    def active(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Helpers — JSON extraction
# ---------------------------------------------------------------------------


def extract_json(raw: str, expect_list: bool = False):
    """Extract JSON from agent response, handling extra text."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        open_char = "[" if expect_list else "{"
        close_char = "]" if expect_list else "}"
        start = raw.find(open_char)
        end = raw.rfind(close_char) + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        return None


# ---------------------------------------------------------------------------
# Helpers — call copilot
# ---------------------------------------------------------------------------


async def call_agent(
    cfg: Config,
    prompt: str,
    depth: int = 0,
    cwd: str | None = None,
    backend: AgentBackend | None = None,
) -> str:
    """Run a sub-agent and return its stdout. Adds UI diagnostics on start/failure.

    Note: depth is used to place diagnostic messages in the status tree.
    """
    be = backend or cfg.pick_backend(depth)
    cmd_list = be.render(prompt)
    ui.add(depth, f"Starting agent [{be}]...")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except FileNotFoundError as e:
        ui.add(depth, f"Agent command not found: {e}")
        raise

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        ui.add(depth, f"Agent [{be}] timed out after 600s")
        raise subprocess.TimeoutExpired(cmd_list, 600)

    if proc.returncode != 0:
        stderr_excerpt = (stderr.decode() if stderr else "")[:400]
        ui.add(depth, f"Agent [{be}] failed (rc={proc.returncode}): {stderr_excerpt}")
        raise RuntimeError(f"Agent [{be}] failed: {stderr_excerpt}")

    stdout_text = (stdout.decode() if stdout else "").strip()
    ui.add(depth, f"Agent [{be}] finished, output len={len(stdout_text)}")
    return stdout_text


# ---------------------------------------------------------------------------
# Git worktree helpers
# ---------------------------------------------------------------------------


async def _git(*args: str, check: bool = False) -> tuple[int, bytes, bytes]:
    """Run a git command asynchronously and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (rc={proc.returncode}): {stderr.decode()[:200]}"
        )
    return proc.returncode, stdout, stderr


async def repo_root() -> Path:
    _, stdout, _ = await _git("rev-parse", "--show-toplevel", check=True)
    return Path(stdout.decode().strip())


async def create_worktree(tag: str) -> Path:
    root = await repo_root()
    branch = f"layer-{tag}"
    wt_dir = root.parent / ".layer-worktrees" / branch
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    _, stdout, _ = await _git("rev-parse", "HEAD", check=True)
    current = stdout.decode().strip()
    await _git("branch", branch, current, check=True)
    await _git("worktree", "add", str(wt_dir), branch, check=True)
    return wt_dir


async def remove_worktree(wt_dir: Path) -> None:
    """Remove a git worktree and delete its branch, then fully delete the directory.

    Safety: only remove directories under the repo's .layer-worktrees directory.
    """
    try:
        root = await repo_root()
        base = root.parent / ".layer-worktrees"
        try:
            wt_real = wt_dir.resolve()
            base_real = base.resolve()
            if not str(wt_real).startswith(str(base_real)):
                # Not under managed worktrees dir — do not recursively delete; still attempt git cleanup
                await _git("worktree", "remove", "--force", str(wt_dir))
                await _git("branch", "-D", wt_dir.name)
                return
        except Exception:
            await _git("worktree", "remove", "--force", str(wt_dir))
            await _git("branch", "-D", wt_dir.name)
            return

        await _git("worktree", "remove", "--force", str(wt_dir))
        await _git("branch", "-D", wt_dir.name)
    finally:
        try:
            if wt_dir.exists():
                shutil.rmtree(wt_dir, ignore_errors=True)
        except Exception:
            pass


async def merge_worktree(wt_dir: Path) -> None:
    branch = wt_dir.name
    await _git("merge", branch, "--no-edit", check=True)
    await remove_worktree(wt_dir)


# ---------------------------------------------------------------------------
# Core flow
# ---------------------------------------------------------------------------


@dataclass
class Proposal:
    agent_id: str
    plan: str
    key_details: str
    backend: AgentBackend | None = None
    subtasks: list[str] = field(default_factory=list)


@dataclass
class CallNode:
    """A node in the agent call tree."""

    task: str
    depth: int
    agent_id: str | None = None
    plan: str | None = None
    backend: str | None = None
    children: list["CallNode"] = field(default_factory=list)
    output: str | None = None
    success: bool | None = None

    def render_tree(self, indent: int = 0) -> str:
        """Render the call tree as a human-readable string."""
        prefix = "  " * indent
        lines = []
        node_desc = f"[Depth {self.depth}] {self.task[:80]}{'...' if len(self.task) > 80 else ''}"
        if self.agent_id:
            node_desc = f"{self.agent_id}: {node_desc}"
        if self.backend:
            node_desc = f"{node_desc} [{self.backend}]"
        lines.append(f"{prefix}{node_desc}")
        if self.plan:
            lines.append(
                f"{prefix}  Plan: {self.plan[:100]}{'...' if len(self.plan) > 100 else ''}"
            )
        if self.output:
            lines.append(
                f"{prefix}  Output: {self.output[:100]}{'...' if len(self.output) > 100 else ''}"
            )
        for child in self.children:
            lines.extend(child.render_tree(indent + 1).split("\n"))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "task": self.task,
            "depth": self.depth,
            "agent_id": self.agent_id,
            "plan": self.plan,
            "backend": self.backend,
            "children": [c.to_dict() for c in self.children],
            "output": self.output,
            "success": self.success,
        }


@dataclass
class CallTree:
    """The full call tree with a reference to the current node."""

    root: CallNode = field(default_factory=lambda: CallNode(task="<root>", depth=-1))
    current: CallNode | None = field(default=None)

    def __post_init__(self):
        if self.current is None:
            self.current = self.root

    def spawn_child(self, task: str, depth: int) -> "CallTree":
        """Create a new tree view with a child as current."""
        child = CallNode(task=task, depth=depth)
        self.current.children.append(child)
        return CallTree(root=self.root, current=child)

    def mark_complete(
        self,
        agent_id: str | None = None,
        plan: str | None = None,
        backend: str | None = None,
        output: str | None = None,
        success: bool | None = None,
    ):
        """Mark the current node with execution details."""
        if agent_id is not None:
            self.current.agent_id = agent_id
        if plan is not None:
            self.current.plan = plan
        if backend is not None:
            self.current.backend = backend
        if output is not None:
            self.current.output = output
        if success is not None:
            self.current.success = success

    def get_context_prompt(self) -> str:
        """Generate a prompt describing the call tree for child agents."""
        if self.current == self.root:
            return ""
        return self._render_for_agent(self.root, 0)

    def _render_for_agent(self, node: CallNode, indent: int) -> str:
        """Render tree in a format suitable for agent consumption."""
        lines = []
        prefix = "  " * indent
        if node.depth >= 0:
            status = ""
            if node.success is not None:
                status = "✓" if node.success else "✗"
            line = f"{prefix}{status} [Depth {node.depth}] Task: {node.task}"
            if node.plan:
                line += f"\n{prefix}  Plan: {node.plan}"
            if node.output:
                line += f"\n{prefix}  Result: {node.output[:200]}"
            lines.append(line)
        for child in node.children:
            lines.extend(self._render_for_agent(child, indent + 1).split("\n"))
        return "\n".join(lines)


@dataclass
class ExecutionResult:
    agent_id: str
    proposal: Proposal
    worktree: Path
    output: str
    success: bool
    call_tree: CallTree | None = None
    task_hash: str | None = None
    test_passed: bool | None = None
    test_output: str = ""


async def run_tests(cwd: str) -> tuple[bool, str]:
    """Run project tests in the given directory. Returns (passed, output)."""
    test_commands = ["npm test", "pytest", "make test", "cargo test"]
    for cmd in test_commands:
        try:
            parts = shlex.split(cmd)
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300.0)
            output = stdout.decode() if stdout else ""
            passed = proc.returncode == 0
            return passed, output[:5000]
        except (FileNotFoundError, asyncio.TimeoutError):
            continue
    return True, ""  # No test framework found, assume pass


async def gather_proposals(
    cfg: Config, pool: AgentPool, task: str, depth: int, call_tree: CallTree
) -> list[Proposal]:
    """Phase 1: sequentially collect competing proposals until convergence."""
    proposals: list[Proposal] = []
    for i in range(cfg.min_competitors * 3):  # hard upper bound
        if not await pool.acquire():
            ui.add(
                depth, f"Agent pool exhausted, stopping proposals at {len(proposals)}"
            )
            break
        try:
            prior = (
                "\n---\n".join(
                    f"Proposal {j + 1}:\n{p.plan}" for j, p in enumerate(proposals)
                )
                or "(none yet)"
            )

            tree_context = call_tree.get_context_prompt()
            prompt = dedent(f"""\
                You are a competing agent. Stay strictly on-task. Output valid JSON only.
                Task: {task}
                {f"\n\n=== CALL TREE (Parent Context) ===\n{tree_context}\n=== END CALL TREE ===\n" if tree_context else ""}
                Previous proposals:
                {prior}

                Return JSON with these keys:
                  - is_novel: true/false
                  - plan: short high-level plan (<=200 chars)
                  - key_details: short critical details (<=400 chars)
                  - (optional) subtasks: ["subtask1", "subtask2"]

                If no novel approach, set is_novel to false.
            """)
            be = cfg.pick_backend(depth)
            raw = await call_agent(cfg, prompt, depth=depth, backend=be)
            data = extract_json(raw)
            if data is None:
                ui.add(depth, f"Agent {i} returned invalid JSON, skipping")
                continue

            # support optional subtasks field for explicit decomposition to next layer
            subtasks = data.get("subtasks", []) or []
            if not isinstance(subtasks, list):
                subtasks = [str(subtasks)]

            if not data.get("is_novel", True) and len(proposals) >= cfg.min_competitors:
                ui.add(depth, f"Proposals converged after {len(proposals)} agents")
                break

            proposals.append(
                Proposal(
                    agent_id=f"d{depth}-c{i}-{uuid.uuid4().hex[:6]}",
                    plan=data.get("plan", ""),
                    key_details=data.get("key_details", ""),
                    backend=be,
                    subtasks=subtasks,
                )
            )
            ui.add(
                depth,
                f"Proposal {len(proposals)} [{be}]: {proposals[-1].plan[:SUMMARY_MAX_LEN]}{'...' if len(proposals[-1].plan) > SUMMARY_MAX_LEN else ''}",
                style="blue",
            )
        finally:
            pool.release()

    return proposals


async def synthesize_directions(
    cfg: Config,
    pool: AgentPool,
    task: str,
    proposals: list[Proposal],
    depth: int,
    call_tree: CallTree,
) -> list[Proposal]:
    """Phase 2: synthesize M distinct directions from proposals."""
    if len(proposals) <= cfg.min_selected:
        return proposals

    summary = "\n---\n".join(
        f"ID: {p.agent_id}\nPlan: {p.plan}\nDetails: {p.key_details}" for p in proposals
    )
    tree_context = call_tree.get_context_prompt()
    prompt = dedent(f"""\
        You are a manager agent. Produce exactly {cfg.min_selected} distinct, executable directions. Keep them focused and non-overlapping.
        Task: {task}
        {f"\n\n=== CALL TREE (Parent Context) ===\n{tree_context}\n=== END CALL TREE ===\n" if tree_context else ""}
        Proposals:
        {summary}

        Return a JSON list with exactly {cfg.min_selected} items. Each item must be:
          {{"plan": "short plan (<=200 chars)", "key_details": "short details (<=400 chars)", "subtasks": [optional list of short subtasks]}}
        Do NOT include extra text.
    """)

    if not await pool.acquire():
        return proposals[: cfg.min_selected]
    try:
        raw = await call_agent(cfg, prompt, depth=depth)
        directions = extract_json(raw, expect_list=True)
        if directions is None or not isinstance(directions, list):
            return proposals[: cfg.min_selected]

        synthesized = []
        for i, d in enumerate(directions[: cfg.min_selected]):
            if isinstance(d, dict):
                synthesized.append(
                    Proposal(
                        agent_id=f"d{depth}-s{i}-{uuid.uuid4().hex[:6]}",
                        plan=d.get("plan", ""),
                        key_details=d.get("key_details", ""),
                        backend=cfg.pick_backend(depth),
                        subtasks=d.get("subtasks", []) if isinstance(d, dict) else [],
                    )
                )
        if len(synthesized) < cfg.min_selected:
            synthesized.extend(proposals[: cfg.min_selected - len(synthesized)])
        return synthesized
    finally:
        pool.release()


async def execute_proposal(
    cfg: Config,
    pool: AgentPool,
    task: str,
    proposal: Proposal,
    depth: int,
    call_tree: CallTree,
) -> ExecutionResult:
    """Phase 3: execute a single proposal in its own worktree."""
    tag = proposal.agent_id
    task_hash = compute_task_hash(task, proposal.plan, proposal.key_details)

    cached = lookup_cache(task, proposal.plan, proposal.key_details)
    if cached and cached.success:
        ui.add(depth, f"Cache hit for {task_hash}, reusing result")
        return ExecutionResult(
            tag,
            proposal,
            Path(""),
            cached.output,
            True,
            call_tree=call_tree,
            task_hash=task_hash,
            test_passed=cached.test_passed,
            test_output=cached.test_output,
        )

    wt = await create_worktree(tag)
    ui.add(depth, f"Executing {proposal.agent_id} [{proposal.backend}] in {wt}")
    try:
        if depth + 1 < cfg.max_depth:
            # Recurse — support explicit subtasks dispatched to the next layer
            if proposal.subtasks:

                async def _run_sub(
                    sub: str, subtag: str, wt_sub: Path, child_tree: CallTree
                ) -> ExecutionResult:
                    try:
                        out = await run_layer(
                            cfg,
                            pool,
                            sub,
                            proposal.plan,
                            depth + 1,
                            cwd=str(wt_sub),
                            call_tree=child_tree,
                        )
                        return ExecutionResult(
                            subtag,
                            Proposal(
                                agent_id=subtag,
                                plan=sub,
                                key_details="",
                                backend=proposal.backend,
                            ),
                            wt_sub,
                            out[:2000],
                            True,
                            call_tree=child_tree,
                        )
                    except Exception as e:
                        return ExecutionResult(
                            subtag,
                            Proposal(
                                agent_id=subtag,
                                plan=sub,
                                key_details="",
                                backend=proposal.backend,
                            ),
                            wt_sub,
                            str(e)[:500],
                            False,
                            call_tree=child_tree,
                        )

                sub_args = []
                for j, sub in enumerate(proposal.subtasks):
                    subtag = f"{tag}-sub{j}"
                    wt_sub = await create_worktree(subtag)
                    child_tree = call_tree.spawn_child(task=sub, depth=depth + 1)
                    child_tree.mark_complete(
                        agent_id=subtag, plan=sub, backend=str(proposal.backend)
                    )
                    sub_args.append((sub, subtag, wt_sub, child_tree))

                results: list[ExecutionResult] = list(
                    await asyncio.gather(*[_run_sub(*args) for args in sub_args])
                )

                # Merge subtask results (merge agent will synthesize a combined output)
                merged = await merge_results(
                    cfg, pool, task, results, depth + 1, call_tree
                )

                # Clean up non-merged worktrees
                for r in results:
                    try:
                        if merged is None or r.worktree != merged.worktree:
                            await remove_worktree(r.worktree)
                    except Exception:
                        pass

                if merged is None:
                    call_tree.mark_complete(success=False)
                    return ExecutionResult(
                        tag,
                        proposal,
                        wt,
                        "all subtask executions failed"[:2000],
                        False,
                        call_tree=call_tree,
                    )

                output = merged.output
            else:
                child_tree = call_tree.spawn_child(task=task, depth=depth + 1)
                child_tree.mark_complete(
                    agent_id=tag, plan=proposal.plan, backend=str(proposal.backend)
                )
                output = await run_layer(
                    cfg,
                    pool,
                    task,
                    proposal.plan,
                    depth + 1,
                    cwd=str(wt),
                    call_tree=child_tree,
                )
        else:
            # Leaf — actually do the work
            prompt = dedent(f"""\
                You are an implementation agent working in your own git worktree.
                Implement the Plan directly by modifying files in the current directory. Do NOT add extra features.
                Task: {task}
                Plan: {proposal.plan}
                Key details: {proposal.key_details}

                If the Plan lists subtasks, implement them as separate, minimal changes.
            """)
            if not await pool.acquire():
                return ExecutionResult(tag, proposal, wt, "pool exhausted", False)
            try:
                output = await call_agent(
                    cfg, prompt, cwd=str(wt), backend=proposal.backend
                )
            finally:
                pool.release()

        test_passed, test_output = await run_tests(str(wt))
        ui.add(depth, f"Tests {'passed' if test_passed else 'failed'} for {tag}")

        cache_entry = CacheEntry(
            task_hash=task_hash,
            task=task,
            plan=proposal.plan,
            output=output[:2000],
            success=True,
            test_passed=test_passed,
            test_output=test_output,
        )
        save_cache(cache_entry)

        return ExecutionResult(
            tag,
            proposal,
            wt,
            output[:2000],
            True,
            call_tree=call_tree,
            task_hash=task_hash,
            test_passed=test_passed,
            test_output=test_output,
        )
    except Exception as e:
        call_tree.mark_complete(success=False)
        return ExecutionResult(
            tag, proposal, wt, str(e)[:500], False, call_tree=call_tree
        )


async def merge_results(
    cfg: Config,
    pool: AgentPool,
    task: str,
    results: list[ExecutionResult],
    depth: int,
    call_tree: CallTree,
) -> ExecutionResult | None:
    """Phase 4: merge all successful results into one."""
    successes = [r for r in results if r.success]
    if not successes:
        return None
    if len(successes) == 1:
        return successes[0]

    summary = "\n---\n".join(
        f"ID: {r.agent_id}\nPlan: {r.proposal.plan}\nOutput (excerpt): {r.output[:500]}"
        for r in successes
    )
    tree_context = call_tree.get_context_prompt()
    prompt = dedent(f"""\
        You are a merge agent. Merge the successful results into one focused solution. Do NOT add new features.
        Task: {task}
        {f"\n\n=== CALL TREE (Parent Context) ===\n{tree_context}\n=== END CALL TREE ===\n" if tree_context else ""}
        Results:
        {summary}

        Output the merged solution as plain text only.
    """)
    if not await pool.acquire():
        return successes[0]
    try:
        # Create a dedicated merge worktree for the merge agent
        merge_tag = f"d{depth}-merge-{uuid.uuid4().hex[:6]}"
        merge_wt = await create_worktree(merge_tag)
        worktrees_dir = merge_wt / ".worktrees"
        worktrees_dir.mkdir(parents=True, exist_ok=True)

        # Link each successful worktree into the merge worktree under .worktrees/<id>
        # Use symlinks to avoid copying heavy directories (node_modules etc.) and preserve filesystem semantics.
        for r in successes:
            dest = worktrees_dir / r.agent_id
            try:
                try:
                    rel_target = os.path.relpath(r.worktree, start=worktrees_dir)
                    dest.symlink_to(rel_target)
                except Exception:
                    dest.symlink_to(r.worktree)
            except Exception:
                # If symlink fails, fall back to best-effort copy of files (excluding .git)
                try:
                    shutil.copytree(
                        r.worktree, dest, ignore=shutil.ignore_patterns(".git")
                    )
                except Exception:
                    try:
                        dest.mkdir(parents=True, exist_ok=True)
                        for src_path in r.worktree.rglob("*"):
                            rel = src_path.relative_to(r.worktree)
                            target = dest / rel
                            if src_path.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                            else:
                                if src_path.name == ".git":
                                    continue
                                try:
                                    shutil.copy2(src_path, target)
                                except Exception:
                                    pass
                    except Exception:
                        pass

        # Run merge agent in the merge worktree so it can inspect .worktrees/
        merged_output = await call_agent(cfg, prompt, depth=depth, cwd=str(merge_wt))

        # After merge, clean up original worktrees and symlinks/copies
        await asyncio.gather(
            *[remove_worktree(r.worktree) for r in successes],
            return_exceptions=True,
        )

        for entry in worktrees_dir.iterdir():
            try:
                if entry.is_symlink():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except Exception:
                pass
        try:
            worktrees_dir.rmdir()
        except Exception:
            pass

        return ExecutionResult(
            agent_id=merge_tag,
            proposal=Proposal(
                agent_id="merge",
                plan="Merged solution from multiple directions",
                key_details="",
            ),
            worktree=merge_wt,
            output=merged_output,
            success=True,
            call_tree=call_tree,
        )
    finally:
        pool.release()


async def run_layer(
    cfg: Config,
    pool: AgentPool,
    task: str,
    context: str = "",
    depth: int = 0,
    cwd: str | None = None,
    call_tree: CallTree | None = None,
) -> str:
    """Main recursive entry: decompose → compete → execute → evaluate."""
    if call_tree is None:
        call_tree = CallTree()

    # show task in the status tree
    ui.add(
        depth,
        f"Task: {task[:SUMMARY_MAX_LEN]}{'...' if len(task) > SUMMARY_MAX_LEN else ''}",
    )

    full_task = (
        f"{task}\n\nAdditional context from upper layer:\n{context}"
        if context
        else task
    )

    # Phase 1: gather proposals
    proposals = await gather_proposals(cfg, pool, full_task, depth, call_tree)
    if not proposals:
        ui.add(depth, "No proposals generated, executing directly")
        if not await pool.acquire():
            return "pool exhausted"
        try:
            return await call_agent(
                cfg, f"Execute this task directly:\n{full_task}", depth=depth, cwd=cwd
            )
        finally:
            pool.release()

    # Check convergence — if all proposals are essentially the same, execute directly
    if len(proposals) >= cfg.min_competitors:
        convergence_prompt = dedent(f"""\
            Are these proposals essentially identical in approach? Answer ONLY "yes" or "no".
            Read the plans below and answer strictly with 'yes' or 'no'. Do NOT provide explanation.
            {chr(10).join(p.plan for p in proposals)}
        """)
        if await pool.acquire():
            try:
                answer = (
                    (await call_agent(cfg, convergence_prompt, depth=depth))
                    .strip()
                    .lower()
                )
            finally:
                pool.release()
            if "yes" in answer:
                ui.add(depth, "All proposals converged, executing directly")
                prompt = dedent(f"""\
                    Execute this task directly. Follow this plan:
                    Task: {full_task}
                    Plan: {proposals[0].plan}
                    Details: {proposals[0].key_details}
                """)
                if await pool.acquire():
                    try:
                        return await call_agent(cfg, prompt, depth=depth, cwd=cwd)
                    finally:
                        pool.release()

    # Phase 2: synthesize
    directions = await synthesize_directions(
        cfg, pool, full_task, proposals, depth, call_tree
    )
    ui.add(depth, f"Synthesized {len(directions)} directions for execution")

    # Phase 3: execute all directions in parallel
    results: list[ExecutionResult] = list(
        await asyncio.gather(
            *[
                execute_proposal(cfg, pool, full_task, p, depth, call_tree)
                for p in directions
            ]
        )
    )

    # Phase 4: merge
    merged = await merge_results(cfg, pool, full_task, results, depth, call_tree)
    if merged is None:
        ui.add(depth, "All executions failed")
        await asyncio.gather(
            *[remove_worktree(r.worktree) for r in results], return_exceptions=True
        )
        return "all executions failed"

    ui.add(depth, f"Merged {len([r for r in results if r.success])} results")

    # Clean up all worktrees, keep merged result
    await asyncio.gather(
        *[
            remove_worktree(r.worktree)
            for r in results
            if r.worktree != merged.worktree
        ],
        return_exceptions=True,
    )
    await merge_worktree(merged.worktree)

    return merged.output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_agent_spec(spec: str) -> tuple[list[int], AgentBackend]:
    """Parse 'depth=command template' into (depths, AgentBackend).

    Format: "DEPTH=CMD" or "DEPTH_RANGE=CMD" where CMD contains {prompt}.

    Depth can be:
        - Single value: "0=CMD" -> [0]
        - Range: "0-2=CMD" -> [0, 1, 2]
        - Comma-separated: "0,1,2=CMD" -> [0, 1, 2]
        - Mixed: "0,2-4=CMD" -> [0, 2, 3, 4]

    Examples:
        "0=opencode run -m claude {prompt}"
        "0-2=opencode run -m sonnet {prompt}"
        "opencode run {prompt}"     (implies depth=0)
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
        "{prompt} placeholder will be passed as final argument. "
        'Example: -a "0=opencode run -m opus {prompt}" '
        '-a "1=opencode run -m sonnet {prompt}"',
    ),
    max_depth: int = typer.Option(3, help="Maximum recursion depth"),
    min_competitors: int = typer.Option(3, "-n", help="Minimum competing proposals"),
    min_selected: int = typer.Option(2, "-m", help="Minimum synthesized directions"),
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
    pool = AgentPool(cfg.max_agents)
    ui.start()
    # header shows basic startup info; do not populate a meta section
    ui.set_header(
        f"LaYER starting: depth={cfg.max_depth}, N={cfg.min_competitors}, M={cfg.min_selected}, max_agents={cfg.max_agents} | Task: {task}"
    )

    try:
        result = asyncio.run(_run_async(cfg, pool, task))
        # show final result in header (concise)
        ui.set_header(
            ui.header
            + f" | Final result: {result[:SUMMARY_MAX_LEN]}{'...' if len(result) > SUMMARY_MAX_LEN else ''}"
        )
    finally:
        ui.stop()


async def _run_async(cfg: Config, pool: AgentPool, task: str) -> str:
    return await run_layer(cfg, pool, task)


if __name__ == "__main__":
    app()
