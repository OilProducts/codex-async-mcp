import asyncio

import pytest

pytest.importorskip("mcp.server.fastmcp")

from codex_async_mcp.jobs import CodexJobManager
from codex_async_mcp.client import DetachedSession


class FakeClient:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.prompts: list[tuple[str, dict]] = []
        self.followups: list[tuple[DetachedSession, str, asyncio.Future]] = []
        self.notifications: list[tuple[str, dict | None]] = []
        self.session: DetachedSession | None = None

    async def start_detached_codex(self, prompt: str, **kwargs):
        self.prompts.append((prompt, kwargs))
        result: asyncio.Future = self.loop.create_future()
        queue: asyncio.Queue = asyncio.Queue()
        session = DetachedSession(
            conversation_id="conv-1",
            request_id=1,
            result=result,
            events=queue,
        )
        self.session = session
        return session

    async def continue_detached_codex(self, session: DetachedSession, prompt: str):
        future: asyncio.Future = self.loop.create_future()
        self.followups.append((session, prompt, future))
        return future

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        self.notifications.append((method, params or {}))


@pytest.mark.asyncio
async def test_job_token_injection_and_event_tracking():
    loop = asyncio.get_running_loop()
    client = FakeClient(loop)
    manager = CodexJobManager(client, id_generator=lambda: "abcd1234", event_poll_interval=0.05)

    job = await manager.create_job("Do the thing")
    assert client.prompts[0][0] == "[job:abcd1234] Do the thing"
    assert job.prompts == ["[job:abcd1234] Do the thing"]

    assert client.session is not None
    await client.session.events.put({"msg": {"type": "status", "payload": "working"}})
    await client.session.events.put({"msg": {"type": "assistant_response", "content": "done"}})

    await asyncio.sleep(0.05)
    event = await asyncio.wait_for(job.next_event(), timeout=0.1)
    assert event["msg"]["type"] == "assistant_response"
    assert job.event_log[0]["msg"]["type"] == "status"
    full_history = job.iter_events(include_non_responses=True)
    assert full_history[0]["msg"]["type"] == "status"

    await client.session.events.put({"msg": {"type": "task_complete"}})
    client.session.result.set_result({"ok": True})

    result = await asyncio.wait_for(job.wait_result(), timeout=0.1)
    assert result == {"status": "ok", "payload": {"ok": True}}

    await asyncio.wait_for(job.wait_closed(), timeout=0.2)
    assert job.results[-1] == {"status": "ok", "payload": {"ok": True}}
    assert job.event_log[-1]["msg"]["type"] == "task_complete"
    assert "abcd1234" not in manager


@pytest.mark.asyncio
async def test_followup_reuses_token_and_records_result():
    loop = asyncio.get_running_loop()
    client = FakeClient(loop)
    manager = CodexJobManager(client, id_generator=lambda: "job1", event_poll_interval=0.05)

    job = await manager.create_job("Initial work")

    follow_future = await job.send_followup("Need clarification")
    assert client.followups[0][1] == "[job:job1] Need clarification"
    assert job.prompts[-1] == "[job:job1] Need clarification"

    follow_future.set_result({"ok": "done"})
    await asyncio.sleep(0)
    assert job.results[-1] == {"status": "ok", "payload": {"ok": "done"}}

    # Finish session so background task exits cleanly
    assert client.session is not None
    await client.session.events.put({"msg": {"type": "task_complete"}})
    client.session.result.set_result({"status": "complete"})
    await asyncio.wait_for(job.wait_closed(), timeout=0.2)
    assert "job1" not in manager


@pytest.mark.asyncio
async def test_job_timeout_triggers_cancel_and_cleanup():
    loop = asyncio.get_running_loop()
    client = FakeClient(loop)
    manager = CodexJobManager(
        client,
        id_generator=lambda: "timeout",
        event_poll_interval=0.05,
        result_timeout=0.05,
    )

    job = await manager.create_job("Take your time")

    result = await asyncio.wait_for(job.wait_result(), timeout=0.2)

    assert result == {
        "status": "timeout",
        "message": "Codex job timeout timed out after 0.05 seconds",
        "retryable": True,
    }
    assert job.error == result["message"]
    assert job.results[-1] == result

    timeout_event = await asyncio.wait_for(job.next_event(), timeout=0.2)
    assert timeout_event["msg"]["type"] == "job_timeout"
    assert timeout_event["msg"]["reason"] == result["message"]

    await asyncio.wait_for(job.wait_closed(), timeout=0.2)
    assert "timeout" not in manager
    assert ("rpc.cancel", {"id": 1}) in client.notifications
