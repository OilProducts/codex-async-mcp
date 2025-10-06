import asyncio
import types

import pytest

from codex_async_mcp.client import CodexMCPClient


@pytest.mark.asyncio
async def test_stdout_reader_handles_large_frames():
    client = CodexMCPClient(command=("python3", "-c", "pass"))

    captured: list[dict] = []

    async def fake_dispatch(payload):
        captured.append(payload)

    client._dispatch = fake_dispatch  # type: ignore[attr-defined]

    reader = asyncio.StreamReader()
    client._proc = types.SimpleNamespace(stdout=reader)

    large_value = "x" * 70_000
    message = f'{{"foo":"{large_value}"}}'.encode("utf-8")

    reader.feed_data(message + b"\n")
    reader.feed_eof()

    await client._stdout_reader()

    assert len(captured) == 1
    assert captured[0]["foo"] == large_value


@pytest.mark.asyncio
async def test_stdout_reader_handles_content_length_frames():
    client = CodexMCPClient(command=("python3", "-c", "pass"))

    captured: list[dict] = []

    async def fake_dispatch(payload):
        captured.append(payload)

    client._dispatch = fake_dispatch  # type: ignore[attr-defined]

    reader = asyncio.StreamReader()
    client._proc = types.SimpleNamespace(stdout=reader)

    payload = b'{"event":"job_update"}'
    frame = b"Content-Length: %d\r\n\r\n" % len(payload) + payload + b"\r\n"

    reader.feed_data(frame)
    reader.feed_eof()

    await client._stdout_reader()

    assert captured == [{"event": "job_update"}]
