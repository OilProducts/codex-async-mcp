"""FastMCP server exposing asynchronous Codex MCP jobs."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Mapping

from fastmcp import FastMCP

from .client import CodexMCPClient
from .jobs import JobRegistry, JobState


_LOGGER = logging.getLogger(__name__)

registry = JobRegistry()
_client: CodexMCPClient | None = None
_background_tasks: set[asyncio.Task[Any]] = set()


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


async def _watch_future(job_id: str, fut: asyncio.Future[Any]) -> None:
    try:
        result = await fut
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - propagate as failure
        _LOGGER.exception("Codex job %s raised: %s", job_id, exc)
        await registry.fail_job(job_id, str(exc))
    else:
        await registry.finish_job(job_id, result)


def _job_payload(state: JobState, cursor: int, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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


@mcp.tool(name="codex_async.start", description="Start an asynchronous Codex job")
async def start(
    prompt: str,
    model: str | None = None,
    profile: str | None = None,
    cwd: str | None = None,
    approval_policy: str | None = None,
    sandbox: str | None = None,
    config: Mapping[str, Any] | None = None,
    base_instructions: str | None = None,
    include_plan_tool: bool | None = None,
    extra_arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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


@mcp.tool(name="codex_async.events", description="Fetch buffered Codex MCP events for a job")
async def fetch_events(job_id: str, cursor: int | None = None) -> dict[str, Any]:
    events, next_cursor = await registry.get_events(job_id, cursor)
    snapshot = await registry.get_snapshot(job_id)
    if snapshot is None:
        raise ValueError(f"unknown job_id {job_id}")
    return _job_payload(snapshot, next_cursor, events=events)


@mcp.tool(name="codex_async.reply", description="Send a follow-up prompt to an existing Codex job")
async def reply(job_id: str, prompt: str) -> dict[str, Any]:
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


def main() -> None:
    """Run the FastMCP server with the default stdio transport."""

    mcp.run()


if __name__ == "__main__":
    main()


__all__ = ["mcp", "registry", "main"]
