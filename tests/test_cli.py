import pytest

pytest.importorskip("mcp.server.fastmcp")

from codex_async_mcp import CodexJobManager, CodexMCPClient, mcp  # noqa: F401 ensure public exports
from codex_async_mcp.cli import build_parser


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.transport == "stdio"
    assert args.mount_path is None


def test_build_parser_custom_transport():
    parser = build_parser()
    args = parser.parse_args(["--transport", "sse", "--mount-path", "/codex"])
    assert args.transport == "sse"
    assert args.mount_path == "/codex"
