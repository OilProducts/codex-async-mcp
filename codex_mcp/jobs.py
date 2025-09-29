"""Shared job state for the FastMCP async wrapper."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


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

    async def get_events(
        self,
        job_id: str,
        cursor: int | None = None,
        *,
        limit: int | None = None,
        event_types: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        start = 0 if cursor is None else max(int(cursor), 0)
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return [], start
            events = state.events
            total = len(events)

            collected: List[Dict[str, Any]] = []
            idx = start
            remaining = None if limit is None or limit <= 0 else limit

            normalized_types = None
            if event_types:
                normalized_types = {t for t in event_types if t}

            while idx < total:
                event = events[idx]
                idx += 1
                if normalized_types is not None:
                    msg = event.get("msg")
                    event_type = msg.get("type") if isinstance(msg, dict) else None
                    if event_type not in normalized_types:
                        continue
                collected.append(event)
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break

            next_cursor = idx
            return collected, next_cursor

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


class CodexJob:
    """Client-side view of an asynchronous Codex job managed by CodexJobManager."""

    def __init__(
        self,
        manager: "CodexJobManager",
        job_id: str,
        session: DetachedSession,
    ) -> None:
        self.manager = manager
        self.job_id = job_id
        self.session = session
        self.prompts: List[str] = []
        self.event_log: List[Dict[str, Any]] = []
        self.results: List[Any] = []
        self.error: Optional[str] = None
        self._loop = manager.loop
        self._events_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._events_drained = asyncio.Event()
        self._closed = asyncio.Event()
        self._event_task: asyncio.Task[None] | None = None
        self._result_task: asyncio.Task[Any] | None = None

    def _start(self) -> None:
        self._event_task = self._loop.create_task(self._pump_events())
        self._result_task = self._loop.create_task(self._watch_result())

    async def _pump_events(self) -> None:
        try:
            while True:
                event = await self.session.next_event()
                self.event_log.append(event)
                await self._events_queue.put(event)

                msg = event.get("msg")
                if isinstance(msg, dict) and msg.get("type") == "task_complete":
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.manager.logger.debug("Event pump for job %s failed: %s", self.job_id, exc)
        finally:
            self._events_drained.set()

    async def _watch_result(self) -> Any:
        try:
            result = await self.session.wait_result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - client gets the same exception
            self.error = str(exc)
            self.results.append(exc)
            raise
        else:
            self.results.append(result)
            return result
        finally:
            try:
                await asyncio.wait_for(
                    self._events_drained.wait(),
                    timeout=self.manager.event_poll_interval,
                )
            except asyncio.TimeoutError:
                if self._event_task and not self._event_task.done():
                    self._event_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._event_task
            else:
                if self._event_task and not self._event_task.done():
                    await self._event_task
            finally:
                self.manager._finalize_job(self)
                self._closed.set()

    async def next_event(self) -> Dict[str, Any]:
        return await self._events_queue.get()

    async def wait_result(self) -> Any:
        if self._result_task is None:
            raise RuntimeError("job has not been initialised")
        return await asyncio.shield(self._result_task)

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def send_followup(self, prompt: str) -> asyncio.Future[Any]:
        tokenised = self.manager._format_prompt(self.job_id, prompt)
        future = await self.manager.client.continue_detached_codex(self.session, tokenised)
        self.prompts.append(tokenised)

        def _record_result(fut: asyncio.Future[Any]) -> None:
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover - surfaced to caller
                self.results.append(exc)
                self.error = str(exc)
            else:
                self.results.append(result)

        future.add_done_callback(_record_result)
        return future


class CodexJobManager(Mapping[str, CodexJob]):
    """Manage detached Codex sessions and provide ergonomic access to their output."""

    def __init__(
        self,
        client: "CodexMCPClient",
        *,
        id_generator: Callable[[], str] | None = None,
        event_poll_interval: float = 0.1,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.client = client
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - fallback when called out of loop
                loop = asyncio.get_event_loop()
        self.loop = loop
        self.event_poll_interval = max(0.01, float(event_poll_interval))
        self.logger = logging.getLogger(__name__)
        self._id_generator = id_generator or (lambda: uuid.uuid4().hex)
        self._jobs: Dict[str, CodexJob] = {}

    def __getitem__(self, key: str) -> CodexJob:
        return self._jobs[key]

    def __iter__(self):
        return iter(self._jobs)

    def __len__(self) -> int:
        return len(self._jobs)

    def __contains__(self, key: object) -> bool:
        return key in self._jobs

    def _format_prompt(self, job_id: str, prompt: str) -> str:
        token = f"[job:{job_id}] "
        if prompt.startswith(token):
            return prompt
        return token + prompt

    def _finalize_job(self, job: CodexJob) -> None:
        self._jobs.pop(job.job_id, None)

    async def create_job(self, prompt: str, **kwargs: Any) -> CodexJob:
        job_id = self._id_generator()
        tokenised_prompt = self._format_prompt(job_id, prompt)
        session = await self.client.start_detached_codex(tokenised_prompt, **kwargs)

        job = CodexJob(self, job_id, session)
        job.prompts.append(tokenised_prompt)
        self._jobs[job_id] = job
        job._start()
        return job
