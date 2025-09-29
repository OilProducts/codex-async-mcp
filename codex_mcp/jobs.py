"""Shared job state for the FastMCP async wrapper."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class JobState:
    job_id: str
    session_id: str | None
    detached: "DetachedSession"
    events: List[Dict[str, Any]] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=lambda: time.time())
    completed_at: float | None = None
    event_task: asyncio.Task[None] | None = None

    @property
    def next_cursor(self) -> int:
        return len(self.events)


class JobRegistry:
    """Track detached Codex sessions and their buffered events."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, detached: "DetachedSession") -> JobState:
        job_id = uuid.uuid4().hex
        session_id: str | None = getattr(detached, "conversation_id", None)
        state = JobState(job_id=job_id, session_id=session_id, detached=detached)
        async with self._lock:
            self._jobs[job_id] = state
        return state

    async def record_event(self, job_id: str, event: Dict[str, Any]) -> None:
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            state.events.append(event)
            msg = event.get("msg")
            if isinstance(msg, dict):
                event_type = msg.get("type")
                if event_type == "session_configured":
                    session_id = msg.get("session_id")
                    if isinstance(session_id, str) and session_id:
                        state.session_id = session_id
                    state.status = JobStatus.RUNNING
                elif event_type == "task_complete":
                    state.status = JobStatus.COMPLETED
                    state.completed_at = time.time()

    async def fail_job(self, job_id: str, error: str) -> None:
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            state.error = error
            state.status = JobStatus.FAILED
            state.completed_at = time.time()

    async def finish_job(self, job_id: str, result: Any) -> None:
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            state.result = result
            state.status = JobStatus.COMPLETED
            state.completed_at = time.time()

    async def get_snapshot(self, job_id: str) -> JobState | None:
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return None
            return JobState(
                job_id=state.job_id,
                session_id=state.session_id,
                detached=state.detached,
                events=list(state.events),
                result=state.result,
                error=state.error,
                status=state.status,
                created_at=state.created_at,
                completed_at=state.completed_at,
                event_task=state.event_task,
            )

    async def get_events(self, job_id: str, cursor: int | None) -> Tuple[List[Dict[str, Any]], int]:
        start = 0 if cursor is None else max(int(cursor), 0)
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return [], start
            slice_events = state.events[start:]
            next_cursor = state.next_cursor
            return list(slice_events), next_cursor

    async def get_state(self, job_id: str) -> JobState | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return
            state.status = JobStatus.RUNNING
            state.error = None


# Imported lazily to avoid circular imports at runtime.
from .client import DetachedSession  # noqa: E402  ( placed at end intentionally )
