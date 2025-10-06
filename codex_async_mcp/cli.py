"""CLI entry point for running the Codex Async MCP server."""

from __future__ import annotations

import argparse
from typing import Sequence

from .server import enable_debug_logging, mcp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-async-mcp",
        description="Run the codex-async-mcp FastMCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="Transport to use when serving MCP (default: stdio).",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Mount path when using the SSE transport.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write detailed logs to codex_async_mcp.log in the current working directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.debug:
        enable_debug_logging()
    if args.transport == "sse":
        mcp.run(transport="sse", mount_path=args.mount_path)
    else:
        mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
