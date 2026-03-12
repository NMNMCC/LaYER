import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

_CACHE_DIR = Path.home() / ".layer" / "cache"


@dataclass
class CacheEntry:
    """A cached execution result."""

    task_hash: str
    task: str
    plan: str
    output: str
    success: bool
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    test_passed: bool | None = None
    test_output: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_hash": self.task_hash,
            "task": self.task,
            "plan": self.plan,
            "output": self.output,
            "success": self.success,
            "created_at": self.created_at,
            "test_passed": self.test_passed,
            "test_output": self.test_output,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(
            task_hash=data["task_hash"],
            task=data["task"],
            plan=data["plan"],
            output=data["output"],
            success=data["success"],
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            test_passed=data.get("test_passed"),
            test_output=data.get("test_output", ""),
            metrics=data.get("metrics", {}),
        )


def compute_task_hash(task: str, plan: str, context: str = "") -> str:
    """Compute a deterministic hash for a task+plan+context combination."""
    content = json.dumps(
        {"task": task, "plan": plan, "context": context}, sort_keys=True
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_cache_path(task_hash: str) -> Path:
    """Get the cache file path for a task hash."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{task_hash}.json"


def load_cache(task_hash: str) -> CacheEntry | None:
    """Load a cached entry if it exists."""
    cache_path = get_cache_path(task_hash)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        return CacheEntry.from_dict(data)
    except (json.JSONDecodeError, KeyError, Exception) as e:
        console.print(
            f"[yellow]Warning: Failed to load cache {task_hash}: {e}[/yellow]"
        )
        return None


def save_cache(entry: CacheEntry) -> None:
    """Save a cache entry to disk."""
    cache_path = get_cache_path(entry.task_hash)
    with open(cache_path, "w") as f:
        json.dump(entry.to_dict(), f, indent=2)
    console.print(f"[green]Cached result for task hash {entry.task_hash}[/green]")


def lookup_cache(task: str, plan: str, context: str = "") -> CacheEntry | None:
    """Look up a cached result by task+plan+context."""
    task_hash = compute_task_hash(task, plan, context)
    entry = load_cache(task_hash)
    if entry:
        console.print(
            f"[blue]Cache hit for task hash {task_hash} (created: {entry.created_at})[/blue]"
        )
    return entry


def update_cache_test_result(
    task_hash: str,
    test_passed: bool,
    test_output: str,
    metrics: dict[str, Any] | None = None,
) -> bool:
    """Update a cache entry with test results."""
    entry = load_cache(task_hash)
    if not entry:
        console.print(
            f"[yellow]Warning: Cannot update test result for unknown hash {task_hash}[/yellow]"
        )
        return False
    entry.test_passed = test_passed
    entry.test_output = test_output
    if metrics:
        entry.metrics.update(metrics)
    save_cache(entry)
    return True
