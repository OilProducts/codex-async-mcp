"""Async MCP client wrapper around ``codex mcp serve``.

This module provides :class:`CodexMCPClient`, a small asyncio-based helper that
launches the Codex MCP server over stdio, performs the handshake described in
``codex-rs/mcp-server/tests/common/mcp_process.rs`` and offers convenience
helpers for issuing JSON-RPC requests.

The wrapper is transport-agnostic beyond stdio and does not depend on
``mcp-types``; all payloads are plain ``dict`` instances that follow the server's
expectations.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from asyncio import QueueEmpty, QueueFull
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import Any, Awaitable, Callable, Dict, Mapping, MutableMapping, Sequence

_LOGGER = logging.getLogger(__name__)

JSONValue = Any
NotificationHandler = Callable[[JSONValue], Awaitable[None] | None]
DEFAULT_PROTOCOL_VERSION = os.environ.get("MCP_SCHEMA_VERSION", "2025-06-18")


@dataclass(slots=True)
class JSONRPCResponse:
    """Container for a JSON-RPC response."""

    id: Any
    result: Any | None = None
    error: Any | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class DetachedSession:
    """Represents a Codex tool invocation running asynchronously."""

    conversation_id: str
    request_id: int
    result: asyncio.Future[Any]
    events: asyncio.Queue[dict[str, Any]]

    def done(self) -> bool:
        return self.result.done()

    async def wait_result(self) -> Any:
        return await asyncio.shield(self.result)

    async def next_event(self) -> dict[str, Any]:
        return await self.events.get()


@dataclass(slots=True)
class _DetachedWaiter:
    queue: asyncio.Queue[dict[str, Any]]
    session_id: asyncio.Future[str]
    result: asyncio.Future[Any]


class CodexMCPClient:
    """Manage a ``codex mcp serve`` subprocess and speak MCP over stdio."""

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._command: tuple[str, ...] = tuple(command or ("codex", "mcp", "serve"))
        self._env = dict(env or {})
        self._cwd = cwd
        self._protocol_version = protocol_version
        self._loop = loop or asyncio.get_event_loop()
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: MutableMapping[Any, asyncio.Future[Any]] = {}
        self._handlers: Dict[str, list[NotificationHandler]] = defaultdict(list)
        self._id_iter = count(1)
        self._write_lock = asyncio.Lock()
        self._shutdown = asyncio.Event()
        self._detached_waiters: Dict[str, _DetachedWaiter] = {}
        self.server_info: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Launch the MCP server process and perform the initialization handshake."""

        if self._proc is not None:
            raise RuntimeError("CodexMCPClient already started")

        env = os.environ.copy()
        env.update(self._env)

        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self._cwd,
        )

        assert self._proc.stdout and self._proc.stdin and self._proc.stderr

        self._reader_task = self._loop.create_task(self._stdout_reader(), name="codex-mcp-stdout")
        self._stderr_task = self._loop.create_task(self._stderr_logger(), name="codex-mcp-stderr")

        await self._initialize()

    async def close(self) -> None:
        """Terminate the MCP server and wait for background tasks to exit."""

        if self._proc is None:
            return

        if self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.close()

        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            _LOGGER.warning("Timed out waiting for codex MCP process; killing")
            self._proc.kill()
            await self._proc.wait()

        for task in (self._reader_task, self._stderr_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await self._abort_outstanding("Codex MCP client closed")

        self._proc = None
        self._reader_task = None
        self._stderr_task = None
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_request(self, method: str, params: JSONValue | None = None) -> Any:
        """Send a JSON-RPC request and await its response."""

        request_id, future, message = self._prepare_request(method, params)
        await self._write(message)
        return await future

    async def send_notification(self, method: str, params: JSONValue | None = None) -> None:
        if self._proc is None:
            raise RuntimeError("Codex MCP process is not running")

        message = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params

        await self._write(message)

    def on_notification(self, method: str, handler: NotificationHandler) -> None:
        """Register a coroutine or function to receive notifications for *method*."""

        self._handlers[method].append(handler)

    async def start_detached_codex(
        self,
        prompt: str,
        *,
        model: str | None = None,
        profile: str | None = None,
        cwd: str | None = None,
        approval_policy: str | None = None,
        sandbox: str | None = None,
        config: Mapping[str, Any] | None = None,
        base_instructions: str | None = None,
        include_plan_tool: bool | None = None,
        extra_arguments: Mapping[str, Any] | None = None,
    ) -> DetachedSession:
        """Launch a Codex MCP tool invocation without waiting for completion."""

        arguments: Dict[str, Any] = {"prompt": prompt}
        if model:
            arguments["model"] = model
        if profile:
            arguments["profile"] = profile
        if cwd:
            arguments["cwd"] = cwd
        if approval_policy:
            arguments["approval-policy"] = approval_policy
        if sandbox:
            arguments["sandbox"] = sandbox
        if config:
            arguments["config"] = dict(config)
        if base_instructions:
            arguments["base-instructions"] = base_instructions
        if include_plan_tool is not None:
            arguments["include-plan-tool"] = include_plan_tool
        if extra_arguments:
            arguments.update(extra_arguments)

        params = {
            "name": "codex",
            "arguments": arguments,
        }

        request_id, result_future, message = self._prepare_request("tools/call", params)

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=128)
        session_future: asyncio.Future[str] = self._loop.create_future()
        waiter = _DetachedWaiter(queue=queue, session_id=session_future, result=result_future)
        self._detached_waiters[str(request_id)] = waiter
        result_future.add_done_callback(lambda fut, rid=str(request_id): self._on_request_done(rid, fut))

        await self._write(message)

        conversation_id = await session_future
        return DetachedSession(
            conversation_id=conversation_id,
            request_id=request_id,
            result=result_future,
            events=queue,
        )

    async def continue_detached_codex(
        self,
        session: DetachedSession,
        prompt: str,
    ) -> asyncio.Future[Any]:
        """Send a follow-up ``codex-reply`` prompt for an existing conversation."""

        args = {
            "name": "codex-reply",
            "arguments": {
                "conversationId": session.conversation_id,
                "prompt": prompt,
            },
        }
        request_id, result_future, message = self._prepare_request("tools/call", args)

        waiter = _DetachedWaiter(
            queue=session.events,
            session_id=self._loop.create_future(),
            result=result_future,
        )
        waiter.session_id.set_result(session.conversation_id)
        key = str(request_id)
        self._detached_waiters[key] = waiter
        result_future.add_done_callback(lambda fut, rid=key: self._on_request_done(rid, fut))

        await self._write(message)
        return result_future

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_request(
        self,
        method: str,
        params: JSONValue | None,
    ) -> tuple[int, asyncio.Future[Any], Mapping[str, Any]]:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Codex MCP process is not running")

        request_id = next(self._id_iter)
        future: asyncio.Future[Any] = self._loop.create_future()
        self._pending[request_id] = future

        message: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        return request_id, future, message

    async def _initialize(self) -> None:
        capabilities = {"tools": {"listChanged": True}}
        client_info = {
            "name": "codex-mcp-wrapper",
            "title": "Codex MCP Wrapper",
            "version": "0.1.0",
        }
        params = {
            "capabilities": capabilities,
            "clientInfo": client_info,
            "protocolVersion": self._protocol_version,
        }

        response = await self.send_request("initialize", params)
        if isinstance(response, dict):
            self.server_info = response.get("serverInfo")
        _LOGGER.debug("Initialized MCP server: %s", response)

        await self.send_notification("notifications/initialized")

    async def _write(self, message: Mapping[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        line = json.dumps(message, separators=(",", ":")) + "\n"
        async with self._write_lock:
            self._proc.stdin.write(line.encode("utf-8"))
            await self._proc.stdin.drain()

    async def _stdout_reader(self) -> None:
        assert self._proc and self._proc.stdout
        while not self._proc.stdout.at_eof():
            try:
                line = await self._proc.stdout.readline()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.error("Error reading MCP stdout: %s", exc)
                break

            if not line:
                break

            try:
                payload = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                _LOGGER.warning("Discarding invalid JSON from MCP server: %r", line)
                continue

            await self._dispatch(payload)

        await self._abort_outstanding("Codex MCP server closed its stdout")
        self._shutdown.set()

    async def _stderr_logger(self) -> None:
        assert self._proc and self._proc.stderr
        while not self._proc.stderr.at_eof():
            try:
                line = await self._proc.stderr.readline()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                _LOGGER.error("Error reading MCP stderr: %s", exc)
                break

            if not line:
                break

            _LOGGER.debug("[codex-mcp stderr] %s", line.decode("utf-8", errors="replace").rstrip())

    async def _dispatch(self, message: Mapping[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            await self._handle_response(message)
            return

        method = message.get("method")
        if method:
            params = message.get("params")
            if method == "codex/event":
                await self._handle_codex_event(params)

            for handler in self._handlers.get(method, ()):  # type: ignore[arg-type]
                try:
                    result = handler(params)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # pragma: no cover - log only
                    _LOGGER.exception("Notification handler for %s failed: %s", method, exc)
            return

        _LOGGER.debug("Unhandled MCP message: %s", message)

    async def _handle_response(self, payload: Mapping[str, Any]) -> None:
        response = JSONRPCResponse(
            id=payload.get("id"),
            result=payload.get("result"),
            error=payload.get("error"),
        )
        future = self._pending.pop(response.id, None)
        if future is None:
            _LOGGER.debug("No pending call for response id %r", response.id)
            return

        if response.error is not None:
            future.set_exception(RuntimeError(response.error))
        else:
            future.set_result(response.result)

    async def _handle_codex_event(self, params: JSONValue) -> None:
        if not isinstance(params, dict):
            return

        meta = params.get("_meta")
        if not isinstance(meta, dict):
            return

        request_id = meta.get("requestId")
        if request_id is None:
            return

        key = str(request_id)
        waiter = self._detached_waiters.get(key)
        if waiter is None:
            return

        await waiter.queue.put(params)

        msg = params.get("msg")
        if isinstance(msg, dict):
            event_type = msg.get("type")
            if event_type == "session_configured" and not waiter.session_id.done():
                conversation_id = msg.get("session_id")
                if isinstance(conversation_id, str) and conversation_id:
                    waiter.session_id.set_result(conversation_id)
                elif not waiter.session_id.done():
                    waiter.session_id.set_exception(
                        RuntimeError("session_configured event missing session_id")
                    )
                if event_type == "task_complete":
                    # Remove once the task finishes to avoid leaking queues.
                    self._detached_waiters.pop(key, None)

    def _on_request_done(self, request_key: str, future: asyncio.Future[Any]) -> None:
        waiter = self._detached_waiters.pop(request_key, None)
        if waiter and not waiter.session_id.done():
            if future.cancelled():
                waiter.session_id.cancel()
            else:
                exc = future.exception()
                if exc is not None:
                    waiter.session_id.set_exception(exc)
                else:
                    waiter.session_id.set_exception(
                        RuntimeError("Codex session completed without a session_configured event")
                    )

    async def _abort_outstanding(self, reason: str) -> None:
        """Fail any pending requests and queues with *reason*."""

        abort_error = RuntimeError(reason)

        for request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(abort_error)
            self._pending.pop(request_id, None)

        for key, waiter in list(self._detached_waiters.items()):
            if not waiter.result.done():
                waiter.result.set_exception(abort_error)
            if not waiter.session_id.done():
                waiter.session_id.set_exception(abort_error)
            self._drain_queue_with_abort(waiter.queue, reason)
            self._detached_waiters.pop(key, None)

    def _drain_queue_with_abort(
        self, queue: asyncio.Queue[dict[str, Any]], reason: str
    ) -> None:
        sentinel = {"msg": {"type": "job_aborted", "reason": reason}}

        while True:
            try:
                queue.put_nowait(sentinel)
            except QueueFull:
                try:
                    queue.get_nowait()
                except QueueEmpty:
                    break
            else:
                break

    # ------------------------------------------------------------------
    # Await helpers
    # ------------------------------------------------------------------
    async def wait_closed(self) -> None:
        """Await process termination and reader shutdown."""

        await self._shutdown.wait()
