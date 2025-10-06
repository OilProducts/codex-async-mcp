import pytest

pytest.importorskip("mcp.server.fastmcp")

from codex_async_mcp import CodexJobManager, CodexMCPClient, mcp  # noqa: F401 ensure public exports
from codex_async_mcp.cli import build_parser


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.transport == "stdio"
    assert args.mount_path is None
    assert args.debug is False


def test_build_parser_custom_transport():
    parser = build_parser()
    args = parser.parse_args(["--transport", "sse", "--mount-path", "/codex"])
    assert args.transport == "sse"
    assert args.mount_path == "/codex"


def test_build_parser_debug_flag():
    parser = build_parser()
    args = parser.parse_args(["--debug"])
    assert args.debug is True
