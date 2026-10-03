from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mlops_pipeline.exceptions import StageError


@dataclass
class StageResult:
    name: str
    ok: bool
    started_at: str
    finished_at: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    run_id: str
    values: dict[str, Any] = field(default_factory=dict)
    history: list[StageResult] = field(default_factory=list)

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise StageError(f"missing context key: {key}")
        return self.values[key]


def run_stage(name: str, ctx: PipelineContext, fn: Callable[[PipelineContext], dict[str, Any]]) -> StageResult:
    started = datetime.now(timezone.utc).isoformat()
    try:
        detail = fn(ctx)
        ctx.values.update(detail)
        result = StageResult(name, True, started, datetime.now(timezone.utc).isoformat(), detail)
    except Exception as exc:  # noqa: BLE001 — stage boundary captures and records
        result = StageResult(
            name,
            False,
            started,
            datetime.now(timezone.utc).isoformat(),
            {"error": str(exc), "type": type(exc).__name__},
        )
        ctx.history.append(result)
        raise StageError(f"{name} failed: {exc}") from exc
    ctx.history.append(result)
    return result
