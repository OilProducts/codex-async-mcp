"""FastMCP server exposing asynchronous Codex MCP jobs."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping, Annotated

from fastmcp import FastMCP
from pydantic import Field

from .client import CodexMCPClient
from .jobs import JobRegistry, JobState

_LOGGER = logging.getLogger(__name__)

LOG_FILE_NAME = "codex_async_mcp.log"


def _configure_logging() -> None:
    log_path = Path.cwd() / LOG_FILE_NAME
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return

    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)


_configure_logging()

registry = JobRegistry()
_client: CodexMCPClient | None = None
_background_tasks: set[asyncio.Task[Any]] = set()
_EVENT_TRUNCATION = 1024


class NotificationHub:
    """Store job status notifications and support cursor-based consumption."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._offset: int = 0
        self._cond = asyncio.Condition()

    async def publish(self, payload: dict[str, Any]) -> int:
        async with self._cond:
            self._items.append(payload)
            self._cond.notify_all()
            return self._offset + len(self._items)

    async def fetch(self, cursor: int | None = None) -> tuple[list[dict[str, Any]], int]:
        async with self._cond:
            end = self._offset + len(self._items)
            absolute_cursor = self._normalize_cursor(cursor, end)
            idx = absolute_cursor - self._offset
            slice_items = self._items[idx:]
            next_cursor = self._offset + len(self._items)
            self._prune(absolute_cursor)
            return list(slice_items), next_cursor

    async def wait(self, cursor: int | None = None) -> tuple[list[dict[str, Any]], int]:
        async with self._cond:
            while True:
                end = self._offset + len(self._items)
                absolute_cursor = self._normalize_cursor(cursor, end)
                if end > absolute_cursor:
                    break
                await self._cond.wait()
                cursor = absolute_cursor
            idx = absolute_cursor - self._offset
            slice_items = self._items[idx:]
            next_cursor = self._offset + len(self._items)
            self._prune(absolute_cursor)
            return list(slice_items), next_cursor

    def _normalize_cursor(self, cursor: int | None, upper_bound: int) -> int:
        if cursor is None:
            return self._offset
        try:
            value = int(cursor)
        except (TypeError, ValueError):
            return self._offset
        if value < self._offset:
            return self._offset
        if value > upper_bound:
            return upper_bound
        return value

    def _prune(self, cutoff: int) -> None:
        if cutoff <= self._offset:
            return
        drop = min(cutoff - self._offset, len(self._items))
        if drop <= 0:
            return
        del self._items[:drop]
        self._offset += drop


notifications = NotificationHub()


async def _broadcast_notification(method: str, payload: Mapping[str, Any]) -> None:
    notifier = getattr(mcp, "notify", None)
    fallback = getattr(mcp, "notify_all", None)
    target = notifier if callable(notifier) else fallback if callable(fallback) else None
    if target is None:
        _LOGGER.debug("FastMCP notify method unavailable; skipping broadcast for %s", method)
        return

    try:
        result = target(method, payload)
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:  # pragma: no cover - defensive logging only
        _LOGGER.debug("Failed to broadcast notification %s: %s", method, exc)


async def _publish_job_update(job_id: str) -> None:
    snapshot = await registry.get_snapshot(job_id)
    if snapshot is None:
        return

    payload: dict[str, Any] = {
        "job_id": snapshot.job_id,
        "status": snapshot.status.value,
        "result": snapshot.result,
        "error": snapshot.error,
        "conversation_id": snapshot.session_id,
        "published_at": time.time(),
        "source": "job",
    }

    cursor = await notifications.publish(payload)
    payload_with_cursor = dict(payload, cursor=cursor)
    await _broadcast_notification("job_update", payload_with_cursor)


def _truncate_value(value: Any, limit: int) -> Any:
    if limit <= 0:
        return value
    if isinstance(value, str) and len(value) > limit:
        extra = len(value) - limit
        return f"{value[:limit]}... ({extra} more chars truncated)"
    if isinstance(value, list):
        return [_truncate_value(item, limit) for item in value]
    if isinstance(value, tuple):  # pragma: no cover - not expected but safe
        return tuple(_truncate_value(item, limit) for item in value)
    if isinstance(value, dict):
        return {key: _truncate_value(item, limit) for key, item in value.items()}
    return value


def _summarise_events(events: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return [dict(event) for event in events]
    summarised: list[dict[str, Any]] = []
    for event in events:
        summarised.append(_truncate_value(event, limit))
    return summarised


def _track(task: asyncio.Task[Any]) -> None:
    _background_tasks.add(task)

    def _cleanup(done: asyncio.Task[Any]) -> None:
        _background_tasks.discard(done)
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive logging only
            _LOGGER.exception("Background task failed: %s", exc)

    task.add_done_callback(_cleanup)


def _require_client() -> CodexMCPClient:
    if _client is None:
        raise RuntimeError("Codex MCP client is not initialised")
    return _client


def _ensure_event_pump(state: JobState) -> None:
    if state.event_task is not None and not state.event_task.done():
        return
    task = asyncio.create_task(_pump_events(state))
    state.event_task = task
    _track(task)


async def _pump_events(state: JobState) -> None:
    job_id = state.job_id
    queue = state.detached.events
    try:
        while True:
            event = await queue.get()
            await registry.record_event(job_id, event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - log and mark failure
        _LOGGER.exception("Event pump for %s failed: %s", job_id, exc)
        await registry.fail_job(job_id, f"event stream error: {exc}")
        await _publish_job_update(job_id)


async def _watch_future(job_id: str, fut: asyncio.Future[Any]) -> None:
    try:
        result = await fut
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - propagate as failure
        _LOGGER.exception("Codex job %s raised: %s", job_id, exc)
        await registry.fail_job(job_id, str(exc))
        await _publish_job_update(job_id)
    else:
        await registry.finish_job(job_id, result)
        await _publish_job_update(job_id)


def _job_payload(state: JobState, cursor: int, events: list[dict[str, Any]] | None = None) -> dict[
    str, Any]:
    payload = {
        "job_id": state.job_id,
        "status": state.status.value,
        "conversation_id": state.session_id,
        "cursor": cursor,
        "result": state.result,
        "error": state.error,
    }
    if events is not None:
        payload["events"] = events
    return payload


@asynccontextmanager
async def _lifespan(_app: FastMCP):
    global _client
    client = CodexMCPClient()
    await client.start()
    _client = client
    try:
        yield
    finally:
        tasks = list(_background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
        _client = None


mcp = FastMCP(name="Codex Async Wrapper", lifespan=_lifespan)


@mcp.tool(
    name="job_start",
    description=(
        "Start a long-lived Codex session—i.e. a Codex CLI coding conversation—in detached mode "
        "so the agent keeps working after your turn. Reach for this when you want the model to "
        "pursue an autonomous plan while you periodically check progress with the job-specific "
        "event stream (`job_events`) and the process-wide notification feed (`job_notifications` or "
        "`job_wait`), steering it via `job_reply`; the return includes the job_id and initial cursor "
        "into that job's private event log."
    ),
)
async def start(
    prompt: Annotated[
        str,
        Field(
            description="Initial user prompt that seeds the Codex conversation (required).",
        ),
    ],
    model: Annotated[
        str,
        Field(
            description="Override for the Codex model name (for example `o3`, `o4-mini`); default = None to use the server configuration.",
        ),
    ] = None,
    profile: Annotated[
        str,
        Field(
            description="Configuration profile defined in Codex `config.toml`; default = None defers to the server profile.",
        ),
    ] = None,
    cwd: Annotated[
        str,
        Field(
            description="Working directory for the session; relative paths resolve against the server cwd; default = None keeps the server default.",
        ),
    ] = None,
    approval_policy: Annotated[
        str,
        Field(
            description="Approval policy for shell commands (`untrusted`, `on-failure`, `never`); default = None keeps the Codex default.",
        ),
    ] = None,
    sandbox: Annotated[
        str,
        Field(
            description="Sandbox mode (`read-only`, `workspace-write`, or `danger-full-access`); default = None keeps the server default.",
        ),
    ] = None,
    config: Annotated[
        Mapping[str, Any],
        Field(
            description="Overrides for individual Codex config settings (mirrors the Codex Config struct); default = None sends no overrides.",
        ),
    ] = None,
    base_instructions: Annotated[
        str,
        Field(
            description="Custom system instructions that replace the default set; default = None keeps the built-in instructions.",
        ),
    ] = None,
    include_plan_tool: Annotated[
        bool,
        Field(
            description="Whether to include the Codex plan tool in the conversation; default = True matches the Codex server default.",
        ),
    ] = True,
    extra_arguments: Annotated[
        Mapping[str, Any],
        Field(
            description="Provider-specific arguments forwarded unchanged to the Codex backend; default = None omits extra parameters.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Proxy the Codex `codex` tool to launch a detached conversation and return its initial state."""
    client = _require_client()
    session = await client.start_detached_codex(
        prompt,
        model=model,
        profile=profile,
        cwd=cwd,
        approval_policy=approval_policy,
        sandbox=sandbox,
        config=config,
        base_instructions=base_instructions,
        include_plan_tool=include_plan_tool,
        extra_arguments=extra_arguments,
        await_ready=False,
    )

    state = await registry.create_job(session)
    _ensure_event_pump(state)
    await registry.mark_running(state.job_id)

    watcher = asyncio.create_task(_watch_future(state.job_id, session.result))
    _track(watcher)

    snapshot = await registry.get_snapshot(state.job_id)
    if snapshot is None:
        raise RuntimeError("failed to create Codex job")
    return _job_payload(snapshot, snapshot.next_cursor, events=[])


@mcp.tool(
    name="job_events",
    description=(
        "Collect buffered Codex streaming events for a detached job starting from the cursor you "
        "already processed in that job's private event log. This feed is intentionally verbose and "
        "noisy; prefer `job_wait` (or `job_notifications`) whenever you just need to know when the "
        "job finishes, and reach for `job_events` only when you must inspect the granular stream."
    ),
)
async def fetch_events(
    job_id: Annotated[str, Field(description="Target job identifier returned by job_start")],
    cursor: Annotated[int, Field(
        description="Number of events already consumed from this job_id's event list; use 0 or omit for the first call, default = None")] = None,
    limit: Annotated[int, Field(gt=0,
                                description="Maximum events to return in this page, default = 20")] = 20,
    event_types: Annotated[list[str], Field(
        description="Optional whitelist of Codex event types to include, default = None")] = None,
    truncate: Annotated[int, Field(gt=0,
                                   description="Maximum characters per string field before truncation, default = _EVENT_TRUNCATION")] = _EVENT_TRUNCATION,
) -> dict[str, Any]:
    """Return Codex events after *cursor*, honoring optional limits and filters."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    events, next_cursor = await registry.get_events(
        job_id,
        cursor,
        limit=limit,
        event_types=event_types,
    )
    truncated = _summarise_events(events, truncate)
    snapshot = await registry.get_snapshot(job_id)
    if snapshot is None:
        raise ValueError(f"unknown job_id {job_id}")
    payload = _job_payload(snapshot, next_cursor, events=truncated)
    payload["returned"] = len(truncated)
    payload["requested_limit"] = limit
    payload["filter_types"] = event_types
    if truncate is not None and truncate > 0:
        payload["truncate"] = truncate
    return payload


@mcp.tool(
    name="job_reply",
    description=(
        "Push an additional user prompt into an existing detached Codex job and advance the cursor "
        "within that job's private event stream. Use this when the background agent needs more "
        "guidance or clarification without restarting the session."
    ),
)
async def reply(
    job_id: Annotated[
        str,
        Field(
            description="Existing Codex conversation identifier returned by `job_start`; required to continue the session.",
        ),
    ],
    prompt: Annotated[
        str,
        Field(
            description="Next user prompt to extend the Codex conversation (required).",
        ),
    ],
) -> dict[str, Any]:
    """Proxy the Codex `codex-reply` tool by sending a follow-up prompt and refreshing the detached job snapshot."""
    client = _require_client()
    state = await registry.get_state(job_id)
    if state is None:
        raise ValueError(f"unknown job_id {job_id}")
    if state.session_id is None:
        raise RuntimeError("Codex conversation is not ready yet")

    _ensure_event_pump(state)
    await registry.mark_running(job_id)

    future = await client.continue_detached_codex(state.detached, prompt)
    watcher = asyncio.create_task(_watch_future(job_id, future))
    _track(watcher)

    snapshot = await registry.get_snapshot(job_id)
    if snapshot is None:
        raise RuntimeError("failed to refresh job state")
    return _job_payload(snapshot, snapshot.next_cursor)


@mcp.tool(
    name="job_notifications",
    description=(
        "Poll the process-wide notification feed for any completion or failure notices that "
        "arrived after your last cursor. This is a non-blocking peek and may return nothing; when "
        "you actually want to wait for work to finish, prefer `job_wait` so you are resumed only "
        "once a notification exists."
    ),
)
async def fetch_notifications(
    cursor: Annotated[int, Field(
        description="Notification index already processed from the shared notification queue; omit or use 0 initially, default = None."
    )] = None,
) -> dict[str, Any]:
    """Fetch completion notifications that occur after *cursor*."""
    items, next_cursor = await notifications.fetch(cursor)
    return {"notifications": items, "cursor": next_cursor}


@mcp.tool(
    name="job_wait",
    description=(
        "Block on the shared notification queue until any detached job reports completion or "
        "failure beyond the supplied cursor. Reach for this when you want to go idle and be woken "
        "up; it prevents you from filling the context window by repeatedly polling with "
        "`job_events` or `job_notifications`."
    ),
)
async def wait_notifications(
    cursor: Annotated[int, Field(
        description="Notification index already processed from the shared notification queue; omit or use 0 initially, default = None")] = None,
) -> dict[str, Any]:
    """Block until a notification beyond *cursor* arrives, then return it."""
    items, next_cursor = await notifications.wait(cursor)
    return {"notifications": items, "cursor": next_cursor}


def main() -> None:
    """Run the FastMCP server with the default stdio transport."""

    mcp.run()


if __name__ == "__main__":
    main()

__all__ = ["mcp", "registry", "main"]
