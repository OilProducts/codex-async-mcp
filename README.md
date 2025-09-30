# Codex Async MCP

Async Python utilities to manage `codex mcp serve`, launch detached Codex agents,
and supervise their MCP event streams.

```python
import asyncio
from codex_async_mcp import CodexMCPClient

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

Run the asynchronous wrapper as an MCP server with `codex-async-mcp` (exposes
stdio by default). The server provides five tools:

- `codex_async_start` launches a Codex job and returns `{job_id, cursor, status}`;
  carry the cursor into later streaming calls.
- `codex_async_events` accepts `job_id` and `cursor`, returning the next batch
  of `codex/event` payloads beyond that cursor. Provide `limit`,
  `event_types`, or `truncate` arguments to cap batch size and trim verbose
  fields by default.
- `codex_async_reply` sends a follow-up prompt in the same Codex session and
  resets the job status to `running` for the next turn.
- `codex_async_notifications` returns job completion notifications after the
  cursor you supply so agents can resume only when work has finished.
- `codex_async_wait` blocks until notifications beyond the given cursor arrive;
  omit `cursor` (or use `0`) on the first call, then reuse the returned value on
  subsequent waits.

The cursor reflects how many events you have already consumed. Each call to
`codex_async_events` returns the next cursor; pass that value back on the next
poll to resume streaming without duplication.

When a job completes (successfully or with an error) the server broadcasts a
`codex_async/job_update` notification to connected FastMCP clients. Agents that
support the MCP notification channel can rely on the push signal, while others
can poll `codex_async_notifications` or yield on `codex_async_wait` to resume
work only when new updates are available. Event payloads returned by
`codex_async_events` are truncated to 1,024 characters per string field by
default; override `truncate` to adjust or disable this safeguard.
