import asyncio

import pytest


@pytest.mark.asyncio
async def test_timeout_notification_includes_event_cursor(monkeypatch):
    from codex_async_mcp.client import DetachedSession
    from codex_async_mcp.server import (
        _publish_job_update,
        notifications,
        registry,
    )

    async def _noop_broadcast(method, payload):  # pragma: no cover - replacement in test only
        return None

    monkeypatch.setattr("codex_async_mcp.server._broadcast_notification", _noop_broadcast)

    # Drain any existing notifications to isolate the test.
    _, cursor = await notifications.fetch(None)

    loop = asyncio.get_running_loop()
    session = DetachedSession(
        conversation_id="conv-test",
        request_id=42,
        result=loop.create_future(),
        events=asyncio.Queue(),
    )

    state = await registry.create_job(session)

    # Record a couple of events so the cursor advances beyond zero.
    await registry.record_event(state.job_id, {"msg": {"type": "status"}})
    await registry.record_event(state.job_id, {"msg": {"type": "status"}})

    timeout_result = {
        "status": "timeout",
        "message": "Job timed out after 1.0 seconds",
        "retryable": True,
    }
    await registry.finish_job(state.job_id, timeout_result)

    await _publish_job_update(state.job_id)

    notifications_list, next_cursor = await notifications.fetch(cursor)
    assert notifications_list, "timeout notification should be present"
    note = notifications_list[-1]

    assert note["result"]["status"] == "timeout"
    assert note["result"]["next_event_cursor"] == 2
    assert "next job_events cursor: 2" in note["result"]["message"]
    assert next_cursor >= 0

    # Drop processed notifications and clean up shared registry state for other tests.
    await notifications.fetch(next_cursor)
    async with registry._lock:  # type: ignore[attr-defined]
        registry._jobs.pop(state.job_id, None)  # type: ignore[attr-defined]
