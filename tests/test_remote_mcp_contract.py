"""Opt-in remote MCP contract tests against a hosted deployment.

This test is disabled by default. Enable it explicitly with:

    SNIPARA_REMOTE_CONTRACT=1
    SNIPARA_REMOTE_MCP_API_KEY=rlm_...
    SNIPARA_REMOTE_PROJECT=snipara

Optional overrides:
    SNIPARA_REMOTE_MCP_URL=https://api.snipara.com/mcp/snipara
    SNIPARA_REMOTE_BASE_URL=https://api.snipara.com
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import httpx
import pytest

from tests.mcp_contract_surface import ESSENTIAL_TOOL_SURFACE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNIPARA_MCP_SRC = PROJECT_ROOT / "apps/mcp-server/snipara-mcp/src"

if str(SNIPARA_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(SNIPARA_MCP_SRC))

import snipara_mcp.server as mcp_server


def _remote_contract_config() -> tuple[str, str]:
    """Resolve remote MCP endpoint and API key for opt-in contract checks."""
    if os.environ.get("SNIPARA_REMOTE_CONTRACT") != "1":
        pytest.skip("Set SNIPARA_REMOTE_CONTRACT=1 to run hosted MCP contract checks.")

    api_key = os.environ.get("SNIPARA_REMOTE_MCP_API_KEY")
    if not api_key:
        pytest.skip("SNIPARA_REMOTE_MCP_API_KEY is required for hosted MCP contract checks.")

    remote_url = os.environ.get("SNIPARA_REMOTE_MCP_URL")
    if remote_url:
        return remote_url, api_key

    project = (
        os.environ.get("SNIPARA_REMOTE_PROJECT")
        or os.environ.get("SNIPARA_REMOTE_PROJECT_SLUG")
        or os.environ.get("SNIPARA_REMOTE_PROJECT_ID")
    )
    if not project:
        pytest.skip(
            "Set SNIPARA_REMOTE_MCP_URL or SNIPARA_REMOTE_PROJECT for hosted MCP contract checks."
        )

    base_url = os.environ.get("SNIPARA_REMOTE_BASE_URL", "https://api.snipara.com").rstrip("/")
    return f"{base_url}/mcp/{project}", api_key


async def _fetch_remote_tools(remote_url: str, api_key: str) -> dict[str, dict]:
    """Fetch tools/list from the hosted MCP endpoint."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(remote_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    tools = (((data or {}).get("result") or {}).get("tools") or [])
    return {tool["name"]: tool for tool in tools if tool.get("name")}


async def _call_remote_tool(
    remote_url: str,
    api_key: str,
    tool_name: str,
    arguments: dict | None = None,
) -> dict:
    """Call a hosted MCP tool via JSON-RPC."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments or {}},
    }
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(remote_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def _remote_rest_config() -> tuple[str, str]:
    """Resolve hosted REST /v1 MCP endpoint and API key for opt-in checks."""
    if os.environ.get("SNIPARA_REMOTE_CONTRACT") != "1":
        pytest.skip("Set SNIPARA_REMOTE_CONTRACT=1 to run hosted MCP contract checks.")

    api_key = os.environ.get("SNIPARA_REMOTE_MCP_API_KEY")
    if not api_key:
        pytest.skip("SNIPARA_REMOTE_MCP_API_KEY is required for hosted MCP contract checks.")

    project = (
        os.environ.get("SNIPARA_REMOTE_PROJECT")
        or os.environ.get("SNIPARA_REMOTE_PROJECT_SLUG")
        or os.environ.get("SNIPARA_REMOTE_PROJECT_ID")
    )
    if not project:
        pytest.skip("SNIPARA_REMOTE_PROJECT is required for hosted REST /v1 MCP contract checks.")

    base_url = os.environ.get("SNIPARA_REMOTE_BASE_URL", "https://api.snipara.com").rstrip("/")
    return f"{base_url}/v1/{project}/mcp", api_key


async def _call_remote_rest_tool(
    remote_url: str,
    api_key: str,
    tool_name: str,
    params: dict | None = None,
) -> dict:
    """Call a hosted REST /v1 MCP tool."""
    payload = {
        "tool": tool_name,
        "params": params or {},
    }
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(remote_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


@pytest.mark.asyncio
async def test_remote_mcp_surface_matches_critical_local_contract():
    """Hosted MCP should expose the critical workflow surface and its key schema props."""
    remote_url, api_key = _remote_contract_config()

    local_tools = {tool.name: tool for tool in await mcp_server.list_tools()}
    remote_tools = await _fetch_remote_tools(remote_url, api_key)

    assert ESSENTIAL_TOOL_SURFACE <= set(local_tools), "Local critical MCP surface is incomplete."

    missing_remote = sorted(ESSENTIAL_TOOL_SURFACE - set(remote_tools))
    assert not missing_remote, (
        f"Hosted MCP at {remote_url} is missing critical tools: {missing_remote}"
    )

    for tool_name in sorted(ESSENTIAL_TOOL_SURFACE):
        local_schema = local_tools[tool_name].inputSchema
        remote_schema = remote_tools[tool_name].get("inputSchema", {})

        local_props = set((local_schema.get("properties") or {}).keys())
        remote_props = set((remote_schema.get("properties") or {}).keys())
        assert local_props <= remote_props, (
            f"Hosted MCP tool {tool_name} at {remote_url} is missing schema properties: "
            f"{sorted(local_props - remote_props)}"
        )

        local_required = set(local_schema.get("required") or [])
        remote_required = set(remote_schema.get("required") or [])
        assert local_required <= remote_required, (
            f"Hosted MCP tool {tool_name} at {remote_url} is missing required params: "
            f"{sorted(local_required - remote_required)}"
        )


@pytest.mark.asyncio
async def test_remote_mcp_stats_smoke_call_succeeds():
    """Hosted MCP should handle an authenticated tools/call after deploy."""
    remote_url, api_key = _remote_contract_config()

    data = await _call_remote_tool(remote_url, api_key, "rlm_stats", {})
    assert not data.get("error"), f"Hosted MCP tools/call returned error: {data.get('error')}"

    content = (((data or {}).get("result") or {}).get("content") or [])
    text_items = [item.get("text", "") for item in content if item.get("type") == "text"]
    assert any(text.strip() for text in text_items), "Hosted MCP rlm_stats returned empty content"


@pytest.mark.asyncio
async def test_remote_rest_mcp_stats_smoke_call_succeeds():
    """Hosted REST /v1 MCP should handle an authenticated tool call after deploy."""
    remote_url, api_key = _remote_rest_config()

    data = await _call_remote_rest_tool(remote_url, api_key, "rlm_stats", {})
    assert data.get("success") is True, (
        f"Hosted REST /v1 MCP returned error: {data.get('error') or data}"
    )
    assert data.get("result") is not None, "Hosted REST /v1 MCP returned no result payload"
