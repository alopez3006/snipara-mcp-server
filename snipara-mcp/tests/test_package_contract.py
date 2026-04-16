"""Package-local regression tests for snipara-mcp."""

from pathlib import Path

from typer.testing import CliRunner

import snipara_mcp
import snipara_mcp.server as mcp_server
from snipara_mcp.cli import app
from snipara_mcp.tool_contract import TOOL_DEFINITIONS


def _package_version_from_pyproject() -> str:
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_package_version_matches_pyproject() -> None:
    """The installed package should report the same version as pyproject.toml."""
    assert snipara_mcp.__version__ == _package_version_from_pyproject()


def test_cli_version_reports_package_version() -> None:
    """The CLI version command should reuse the package version."""
    runner = CliRunner()

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert f"Snipara CLI v{snipara_mcp.__version__}" in result.stdout
    assert f"snipara-mcp: {snipara_mcp.__version__}" in result.stdout


def test_cli_tools_list_renders_tool_names(monkeypatch) -> None:
    """The generic tools list command should surface the hosted tool inventory."""
    runner = CliRunner()

    async def fake_call_mcp_jsonrpc(api_url, project_slug, auth_header, method, params):
        assert method == "tools/list"
        assert project_slug == "snipara"
        return {
            "result": {
                "tools": [
                    {"name": "rlm_help", "description": "Recommend the right tool for a task."},
                    {"name": "rlm_session_memories", "description": "Load tiered session memories."},
                ]
            }
        }

    monkeypatch.setattr("snipara_mcp.cli.load_auth_header", lambda: "rlm_test")
    monkeypatch.setattr("snipara_mcp.cli.call_mcp_jsonrpc", fake_call_mcp_jsonrpc)

    result = runner.invoke(app, ["tools", "list", "--slug", "snipara"])

    assert result.exit_code == 0
    assert "Available MCP tools: 2" in result.stdout
    assert "rlm_help" in result.stdout
    assert "rlm_session_memories" in result.stdout


def test_cli_tools_call_decodes_json_text_payload(monkeypatch) -> None:
    """The generic tools call command should pretty-print decoded JSON text payloads."""
    runner = CliRunner()

    async def fake_call_mcp_tool(api_url, project_slug, auth_header, tool_name, arguments):
        assert tool_name == "rlm_help"
        assert arguments == {"query": "session automation"}
        return {
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": '{"recommended_tools":["rlm_session_memories","rlm_remember_if_novel"]}',
                    }
                ]
            }
        }

    monkeypatch.setattr("snipara_mcp.cli.load_auth_header", lambda: "rlm_test")
    monkeypatch.setattr("snipara_mcp.cli.call_mcp_tool", fake_call_mcp_tool)

    result = runner.invoke(
        app,
        [
            "tools",
            "call",
            "rlm_help",
            "--slug",
            "snipara",
            "--args",
            '{"query":"session automation"}',
        ],
    )

    assert result.exit_code == 0
    assert '"recommended_tools"' in result.stdout
    assert "rlm_session_memories" in result.stdout


async def test_list_tools_matches_generated_contract() -> None:
    """The packaged MCP server should expose the generated tool contract verbatim."""
    listed_tools = {tool.name: tool.inputSchema for tool in await mcp_server.list_tools()}
    contract_tools = {tool["name"]: tool["inputSchema"] for tool in TOOL_DEFINITIONS}

    assert listed_tools == contract_tools
