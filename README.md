# Codex MCP Wrapper

Async Python utilities to manage `codex mcp serve`, launch detached Codex agents,
and supervise their MCP event streams.

```python
import asyncio
from codex_mcp import CodexMCPClient

async def main():
    client = CodexMCPClient()
    await client.start()

    session = await client.start_detached_codex("Summarise the latest logs")

    async def watcher():
        while True:
            event = await session.next_event()
            print("event", event.get("msg", {}).get("type"))
            if session.done():
                break

    watcher_task = asyncio.create_task(watcher())
    result = await session.wait_result()
    print("final result", result)

    watcher_task.cancel()
    await client.close()

asyncio.run(main())
```

## FastMCP server

Run the asynchronous wrapper as an MCP server with `codex-mcp-async` (exposes
stdio by default). The server provides three tools:

- `codex_async.start` launches a Codex job and returns `{job_id, cursor, status}`
  immediately.
- `codex_async.events` accepts `job_id` and `cursor`, replaying queued
  `codex/event` payloads beyond that cursor and exposing the latest status,
  result, and error state.
- `codex_async.reply` sends a follow-up prompt in the same Codex session and
  resets the job status to `running` for the next turn.

The cursor reflects how many events you have already consumed. Each call to
`codex_async.events` returns the next cursor; pass that value back on the next
poll to resume streaming without duplication.
