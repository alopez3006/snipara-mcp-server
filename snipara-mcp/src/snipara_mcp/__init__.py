"""Snipara MCP Server - Context optimization for LLMs."""

__version__ = "2.6.1"
__all__ = ["main", "get_snipara_tools"]


def main(*args, **kwargs):
    """Run the MCP server without importing server dependencies at package import time."""
    from .server import main as _main

    return _main(*args, **kwargs)


def get_snipara_tools(*args, **kwargs):
    """Get Snipara tools for rlm-runtime integration.

    This is a lazy import to avoid requiring rlm-runtime
    when using snipara-mcp as a standalone MCP server.

    See rlm_tools.get_snipara_tools for full documentation.
    """
    from .rlm_tools import get_snipara_tools as _get_snipara_tools
    return _get_snipara_tools(*args, **kwargs)
