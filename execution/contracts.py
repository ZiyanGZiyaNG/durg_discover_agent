from __future__ import annotations
from typing import Any, Literal, Protocol
from pydantic import BaseModel, Field


Status = Literal[
    "success",
    "partial_truncated",
    "partial_timeout",
    "partial_error",
    "no_records",
    "error",
    "not_executed",
]


class StepResult(BaseModel):
    step_id: str
    operation: str
    status: Status
    source: str
    payload: Any = None
    provides: list[str] = Field(default_factory=list)
    total_count: int | None = None
    fetched_count: int | None = None
    truncated: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0


class StepContext(BaseModel):
    step_id: str
    operation: str
    entities: dict[str, dict[str, Any]]
    board: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    ranking: dict[str, Any] | None = None
    aggregation: dict[str, Any] | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)


class Adapter(Protocol):
    source: str

    def run(self, ctx: StepContext) -> StepResult: ...



class ToolCall(BaseModel):
    """One executable call produced by compiling an approved plan step."""

    step_id: str
    operation: str
    adapter: str
    source: str
    declared_sources: list[str] = Field(default_factory=list)
    entities: dict[str, dict[str, Any]] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    ranking: dict[str, Any] | None = None
    aggregation: dict[str, Any] | None = None
    inputs: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    requires_all: list[str] = Field(default_factory=list)
    requires_any: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    output_model: str