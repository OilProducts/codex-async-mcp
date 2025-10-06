# Codex Async MCP

Codex Async MCP wraps the `codex mcp-server` CLI so you can run Codex coding agents as long-lived background jobs through the Model Context Protocol (MCP). It launches the Codex server over stdio, relays streaming events, and exposes them via FastMCP so any MCP-compatible client can supervise and steer Codex remotely.

## Highlights
- Start Codex conversations in detached mode and keep them alive after your turn completes.
- Stream Codex event logs and receive completion notifications using cursor-based pagination.
- Resume conversations by posting follow-up prompts without restarting the Codex process.
- Choose among all FastMCP transports (`stdio`, `sse`, `streamable-http`).

## Prerequisites
- Python 3.11 or later.
- The `codex` CLI on your `PATH` (the server shells out to `codex mcp-server`).
- Access to the Codex configuration/profile you expect the MCP to run with (credentials, repository access, approval policy, etc.).

## Installation
I recommend using `pipx` for isolation:
```
pipx install codex-async-mcp
```

## Running the MCP server
The package installs the `codex-async-mcp` console script. By default it serves the MCP over stdio because thats what Codex expects, but you can choose any FastMCP transport.
```
codex-async-mcp
```

Available transports:
- `stdio` *(default)* – run as a local subprocess.
- `sse` – expose a Server-Sent Events endpoint. Provide `--mount-path=/mcp` (or your preferred path) when embedding behind an HTTP server.
- `streamable-http` – enable FastMCP's experimental streaming HTTP transport.

Example:
```
codex-async-mcp --transport sse --mount-path /mcp
```

Pass `--debug` to persist detailed logs (including Codex stderr passthrough) to `codex_async_mcp.log` in the current working directory.

## Sample Codex MCP configuration
Codex discovers MCP servers via the `[mcp_servers]` table in `~/.codex/config.toml`. The entry below launches this package through the stdio transport, matching the defaults described in the Codex MCP guide:

```toml
# ~/.codex/config.toml
[mcp_servers.codex_async]
command = "codex-async-mcp"
args = []  # For debug logging, use ["--debug"] or append "--debug" to the list.
```

If Codex cannot resolve `codex-async-mcp` on your `PATH`, replace the command with the absolute location that `pipx` printed after installation (for example `~/.local/bin/codex-async-mcp`).

## Tool catalogue
`codex-async-mcp` exposes five MCP tools. All responses are JSON-serialisable and designed to be forwarded verbatim to MCP clients.

### `job_start`
Launch a detached Codex conversation.
- **Required input:** `prompt` – initial user instruction.
- **Optional inputs:** `model`, `profile`, `cwd`, `approval_policy`, `sandbox`, `config`, `base_instructions`, `include_plan_tool`, `extra_arguments`.
- **Returns:** `{ job_id, status, conversation_id, cursor, events }`.

Use the returned `job_id` and `cursor` across subsequent calls.

### `job_events`
Fetch buffered streaming events for a specific job.
- **Inputs:** `job_id`, optional `cursor` (defaults to 0), `limit` (default 20), `event_types`, `truncate` (default 1024 characters).
- **Returns:** Event batch plus an updated cursor and snapshot metadata (`status`, `result`, `error`).

Events are truncated to keep payloads small. Pass `truncate=0` to disable trimming.

### `job_reply`
Post an additional prompt to an existing Codex job.
- **Inputs:** `job_id`, `prompt`.
- **Returns:** Updated job snapshot with the new cursor and status.

The job status switches back to `running`, and a new result watcher is queued automatically.

### `job_notifications`
Peek at the process-wide notification queue for completions or failures.
- **Input:** optional `cursor` (defaults to the beginning of the queue).
- **Returns:** `{ notifications, cursor }`. This call never blocks; use it for non-blocking polling.

### `job_wait`
Block until any job posts a completion notification beyond the supplied cursor.
- **Input:** optional `cursor`.
- **Returns:** Same schema as `job_notifications`, but only after at least one new notification arrives.

## Typical MCP workflow
1. Call `job_start` with your initial prompt and capture `{ job_id, cursor }`.
2. Wait for progress with `job_wait` (blocking) or `job_notifications` (non-blocking).
3. Inspect the detailed stream via `job_events`, always passing the latest cursor you have consumed.
4. If the agent needs more guidance, call `job_reply` with the same `job_id` and your follow-up prompt, then repeat from step 2.
5. When the job snapshot reports `status == "completed"` or `"failed"`, read `result` or `error` from the payload.

All cursors are monotonically increasing integers. Reuse the most recent cursor to avoid processing the same data twice.

## Working with cursors and logs
- **Job event log:** Each detached job buffers every Codex event it has emitted. `job_events` paginates through this buffer and applies string truncation to limit payload size.
- **Notification feed:** The notification queue is global across all jobs. Store the cursor returned by `job_notifications` or `job_wait` and supply it on your next call to receive only new items.
- **Logging:** Run the server with `--debug` to write `codex_async_mcp.log`, then tail it while troubleshooting handshake issues or Codex runtime errors.

## Using the Python helpers directly
You can embed the same behaviour in your own asyncio application without running FastMCP:
```python
import asyncio
from codex_async_mcp import CodexMCPClient, CodexJobManager

async def main():
    client = CodexMCPClient()
    await client.start()

    manager = CodexJobManager(client)
    job = await manager.create_job("Inventory the failing tests in repo X")
    result = await job.wait_result()
    print("result:", result)

    await client.close()

asyncio.run(main())
```

`CodexJobManager` keeps per-job event queues and result futures in sync with the detached Codex sessions managed by `CodexMCPClient`.

## Development
- Run test suites with `pytest`.
- Integration tests expect the `codex` CLI to be available; skip or mock those pieces when running in environments without Codex.
- No formatter is enforced. Follow your team's conventions when contributing patches.

## License
Distributed under the MIT License. See `LICENSE` for details.
